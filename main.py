import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from datetime import  timedelta
import json
from pathlib import Path
import requests
import redis
from typing import List, Dict, Any
import logging
import threading
from datetime import datetime

# Đánh giá mô hình
from model_evaluator import ModelEvaluator, create_test_cases
from text_processing import TextProcessor
from models import db, Job
from caching import RedisJobCache
from job_recommender import TransformerJobRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
global_model = None
model_lock = threading.Lock()

class Config:
    # Database Configuration
    DB_USERNAME = os.environ.get('DB_USERNAME', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '1234')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'jobsayhi_db')

    # Redis Configuration
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_CACHE_EXPIRY = int(os.environ.get('REDIS_CACHE_EXPIRY', 3600))

    # Model Configuration
    MAIN_MODEL = os.environ.get(
        'MAIN_MODEL',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )

    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')

    # CORS Configuration
    ALLOWED_ORIGINS = os.environ.get(
        'ALLOWED_ORIGINS',
        'http://localhost:3000,http://localhost:8080'
    ).split(',')

# Logging Setup
def setup_logging():
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def create_app():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create Flask app
    app = Flask(__name__)

    # Configure SQLAlchemy
    sqlalchemy_database_uri = (
        f"mysql://{Config.DB_USERNAME}:{Config.DB_PASSWORD}@"
        f"{Config.DB_HOST}/{Config.DB_NAME}"
    )
    app.config['SQLALCHEMY_DATABASE_URI'] = sqlalchemy_database_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Configure CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": Config.ALLOWED_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True
        }
    })

    # Initialize extensions
    db.init_app(app)

    # Redis client
    redis_client = redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        decode_responses=False
    )

    def download_google_drive_file (file_url: str) -> bytes:
        try:
            file_id = file_url.split('/d/')[1].split('/view')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'

            response = requests.get(download_url)
            response.raise_for_status()

            return response.content
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise ValueError(f"Failed to download file: {str(e)}")

    @app.route('/api/recommend/drive-url-file', methods=['POST'])
    def recommend_url_drive_pdf ():
        try:
            file_url = request.form.get('file_url')
            if not file_url:
                return jsonify({'error': 'No file URL provided'}), 400

            # Get pagination and filtering params
            limit = int(request.form.get('limit', 20))
            page = int(request.form.get('page', 1))
            keyword = request.form.get('keyword')
            order_by = request.form.get('orderBy', 'similarity')

            pdf_bytes = download_google_drive_file(file_url)
            cv_text = TextProcessor.extract_text_from_pdf_bytes(pdf_bytes)

            recommender = TransformerJobRecommender(redis_client,global_model)
            recommendations = recommender.recommend_jobs(
                cv_text,
                limit=limit,
                page=page,
                keyword=keyword,
                order_by=order_by
            )

            return jsonify({
                'status': 'success',
                **recommendations
            })

        except Exception as e:
            logger.error(f"Error in recommendation endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recommend/file', methods=['POST'])
    def recommend_by_file_pdf ():
        try:
            if 'cv' not in request.files:
                return jsonify({'error': 'No CV file provided'}), 400

            cv_file = request.files['cv']
            if cv_file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Get pagination and filtering params
            limit = int(request.args.get('limit', 10))
            page = int(request.args.get('page', 1))
            keyword = request.args.get('keyword')
            order_by = request.args.get('orderBy', 'similarity')

            cv_text = TextProcessor.extract_text_from_pdf(cv_file)

            recommender = TransformerJobRecommender(redis_client,global_model)
            recommendations = recommender.recommend_jobs(
                cv_text,
                limit=limit,
                page=page,
                keyword=keyword,
                order_by=order_by
            )

            return jsonify({
                'status': 'success',
                **recommendations
            })

        except Exception as e:
            logger.error(f"Error in recommendation endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # Add this method to the routes section of the Flask application
    @app.route('/api/jobs/related/<job_id>', methods=['GET'])
    def get_related_jobs_endpoint (job_id):
        try:
            # Get pagination parameters
            limit = int(request.args.get('limit', 10))
            page = int(request.args.get('page', 1))

            # Verify job exists
            job = Job.query.filter_by(id=job_id).first()
            if not job:
                return jsonify({'error': 'Job not found'}), 404

            recommender = TransformerJobRecommender(redis_client,global_model)
            related_jobs = recommender.get_related_jobs(
                job_id,
                limit=limit,
                page=page
            )

            return jsonify({
                'status': 'success',
                **related_jobs
            })

        except Exception as e:
            logger.error(f"Error in related jobs endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/cache/add-job/<job_id>', methods=['POST'])
    def add_job_to_cache (job_id):
        try:
            recommender = TransformerJobRecommender(redis_client,global_model)
            recommender.cache.add_job_embedding_to_cache(job_id, recommender.model.encode(
                [Job.query.filter_by(id=job_id).first().combined_text])[0])
            return jsonify({'status': 'success', 'message': f'Job {job_id} added to cache'})
        except Exception as e:
            logger.error(f"Error adding job to cache: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/jobs/generate-embedding', methods=['POST'])
    def generate_job_embedding ():
        try:
            # Get job data from request
            data = request.get_json()
            job_id = data.get('jobId')
            job_text = data.get('jobText')

            # Prepare the embedding
            language = TextProcessor.detect_language(job_text)
            processed_text = TextProcessor.preprocess_text(job_text, language)

            # Generate embedding using the model
            embedding = global_model.encode([processed_text])[0]

            # Add to Redis cache
            cache = RedisJobCache(redis_client)
            cache.add_job_embedding_to_cache(job_id, embedding)

            embedding_str = json.dumps(embedding.tolist())

            return jsonify({
                'status': 'success',
                'embedding': embedding_str
            }), 200

        except Exception as e:
            logger.error(f"Error generating job embedding: {str(e)}")
            return jsonify({'error': str(e)}), 500

    # API Test hiệu năng mô hình
    @app.route('/api/evaluation/run-full', methods=['POST'])
    def run_full_evaluation ():
        try:
            # Get evaluation parameters from request
            data = request.get_json()
            n_test_cases = data.get('n_test_cases', 100)
            n_latency_runs = data.get('n_latency_runs', 50)

            # Get jobs from database for test cases
            jobs = Job.query.all()
            if not jobs:
                return jsonify({'error': 'No jobs available for evaluation'}), 400

            # Create test cases
            job_dicts = [
                {
                    'id': job.id,
                    'category_id': job.category_id,
                    'combined_text': job.combined_text
                }
                for job in jobs
            ]

            test_cases = create_test_cases(job_dicts, n_cases=n_test_cases)
            test_cvs = [job.combined_text for job in jobs[:10]]  # Use 10 jobs as test CVs

            # Initialize recommender and evaluator
            recommender = TransformerJobRecommender(redis_client,global_model)
            evaluator = ModelEvaluator(recommender)

            # Generate evaluation report
            report = evaluator.generate_test_report(test_cases, test_cvs)

            # Save report
            save_evaluation_report(report)

            return jsonify({
                'status': 'success',
                'report': report
            })

        except Exception as e:
            logger.error(f"Error in evaluation endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/evaluation/latency-test', methods=['POST'])
    def test_latency ():
        try:
            data = request.get_json()
            n_runs = data.get('n_runs', 50)

            # Get sample CVs for testing
            jobs = Job.query.limit(10).all()  # Use 10 jobs as test CVs
            test_cvs = [job.combined_text for job in jobs]

            recommender = TransformerJobRecommender(redis_client,global_model)
            evaluator = ModelEvaluator(recommender)

            latency_metrics = evaluator.evaluate_latency(test_cvs, n_runs=n_runs)

            return jsonify({
                'status': 'success',
                'metrics': latency_metrics
            })

        except Exception as e:
            logger.error(f"Error in latency test endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/evaluation/relevance-test', methods=['POST'])
    def test_relevance ():
        try:
            data = request.get_json()
            n_test_cases = data.get('n_test_cases', 100)

            # Get jobs and create test cases
            jobs = Job.query.all()
            job_dicts = [
                {
                    'id': job.id,
                    'category_id': job.category_id,
                    'combined_text': job.combined_text
                }
                for job in jobs
            ]

            test_cases = create_test_cases(job_dicts, n_cases=n_test_cases)

            recommender = TransformerJobRecommender(redis_client,global_model)
            evaluator = ModelEvaluator(recommender)

            relevance_metrics = evaluator.evaluate_relevance_metrics(test_cases)

            return jsonify({
                'status': 'success',
                'metrics': relevance_metrics
            })

        except Exception as e:
            logger.error(f"Error in relevance test endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/evaluation/reports', methods=['GET'])
    def get_evaluation_reports ():
        try:
            # Get query parameters
            days = int(request.args.get('days', 7))  # Get reports from last 7 days by default

            # Load and filter reports
            reports = load_evaluation_reports(days)

            return jsonify({
                'status': 'success',
                'reports': reports
            })

        except Exception as e:
            logger.error(f"Error retrieving evaluation reports: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def save_evaluation_report (report: Dict[str, Any]) -> None:
        """Save evaluation report to file"""
        reports_dir = Path('evaluation_reports')
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f'report_{timestamp}.json'

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def load_evaluation_reports (days: int) -> List[Dict[str, Any]]:
        """Load evaluation reports from the last N days"""
        reports_dir = Path('evaluation_reports')
        if not reports_dir.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        reports = []

        for report_file in reports_dir.glob('report_*.json'):
            try:
                # Get report date from filename
                report_date = datetime.strptime(report_file.stem.split('_')[1], '%Y%m%d')

                if report_date >= cutoff_date:
                    with open(report_file) as f:
                        report = json.load(f)
                        reports.append(report)
            except Exception as e:
                logger.error(f"Error loading report {report_file}: {str(e)}")
                continue

        return sorted(reports, key=lambda x: x['timestamp'], reverse=True)

    @app.route('/api/cache/clear', methods=['POST'])
    def clear_cache ():
        try:
            cache = RedisJobCache(redis_client)
            success = cache.clear_all_cache()

            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Cache cleared successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to clear cache'
                }), 500

        except Exception as e:
            logger.error(f"Error in clear cache endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500
    return app, db, redis_client

# Global model initialization with thread safety
def initialize_model(model_path=None):
    """Initialize the model in a separate thread"""
    global global_model
    if global_model is None:
        with model_lock:
            if global_model is None:  # Double-check lock
                logger.info("Initializing model...")
                model_path = model_path or Config.MAIN_MODEL
                global_model = SentenceTransformer(model_path)
                logger.info("Model initialized successfully")
def initCaching(redis_client):
    RedisJobCache(redis_client)

def setup_app():
    # Initialize model on startup
    threading.Thread(target=initialize_model).start()

    app, db, redis_client = create_app()

    return app, db, redis_client

# WSGI Entry Point for cPanel
application, db, redis_client = setup_app()

# Modify main block for WSGI compatibility
if __name__ == '__main__':
    application.run(host='127.0.0.1', port=5000, threaded=True)