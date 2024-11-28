# Requirements: pip install flask flask-sqlalchemy mysql-connector-python nltk scikit-learn PyMuPDF langdetect underthesea

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import fitz
import langdetect
from underthesea import word_tokenize
import re
from datetime import datetime
import pickle
import os
from flask import Flask
from flask_cors import CORS
import requests

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:1234@localhost/jobsayhi_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model Cache
MODEL_CACHE_PATH = 'model_cache/'
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)


# Database Models
class Job(db.Model):
	__tablename__ = 'job'

	id = db.Column(db.String, primary_key=True)
	title = db.Column(db.String(255), nullable=False)
	description = db.Column(db.Text, nullable=False)
	job_requirements = db.Column(db.Text)
	location = db.Column(db.String(255))
	created_at = db.Column(db.DateTime, default=datetime.utcnow)

	@property
	def combined_text (self):
		return f"{self.title} {self.description} {self.job_requirements or ''}"

	def to_dict (self):
		return {
			'id': self.id,
			'title': self.title,
			'description': self.description,
			'job_requirements': self.job_requirements,
			'location': self.location,
			'created_at': self.created_at.isoformat()
		}


class ModelCache:
	def __init__ (self):
		self.vectorizer_path = os.path.join(MODEL_CACHE_PATH, 'tfidf_vectorizer.pkl')
		self.svd_path = os.path.join(MODEL_CACHE_PATH, 'svd_model.pkl')
		self.matrix_path = os.path.join(MODEL_CACHE_PATH, 'tfidf_matrix.pkl')
		self.vectorizer = None
		self.svd_model = None
		self.tfidf_matrix = None
		self.last_update = None
		self.cache_duration = 3600  # 1 hour cache duration

	def needs_update (self):
		if not all([os.path.exists(p) for p in [self.vectorizer_path, self.svd_path, self.matrix_path]]):
			return True
		if not self.last_update or (datetime.now() - self.last_update).total_seconds() > self.cache_duration:
			return True
		return False

	def save_models (self, vectorizer, svd_model, tfidf_matrix):
		with open(self.vectorizer_path, 'wb') as f:
			pickle.dump(vectorizer, f)
		with open(self.svd_path, 'wb') as f:
			pickle.dump(svd_model, f)
		with open(self.matrix_path, 'wb') as f:
			pickle.dump(tfidf_matrix, f)
		self.vectorizer = vectorizer
		self.svd_model = svd_model
		self.tfidf_matrix = tfidf_matrix
		self.last_update = datetime.now()

	def load_models (self):
		if os.path.exists(self.vectorizer_path):
			with open(self.vectorizer_path, 'rb') as f:
				self.vectorizer = pickle.load(f)
			with open(self.svd_path, 'rb') as f:
				self.svd_model = pickle.load(f)
			with open(self.matrix_path, 'rb') as f:
				self.tfidf_matrix = pickle.load(f)
			self.last_update = datetime.now()
			return True
		return False


model_cache = ModelCache()


class TextProcessor:
	@staticmethod
	def detect_language (text):
		try:
			return langdetect.detect(text)
		except:
			return 'en'  # Default to English if detection fails

	@staticmethod
	def preprocess_text (text, language):
		if language == 'en':
			lemmatizer = WordNetLemmatizer()
			text = re.sub(r'\W', ' ', text).lower()
			words = text.split()
			words = [lemmatizer.lemmatize(word) for word in words
			         if word not in stopwords.words('english')
			         and word not in ENGLISH_STOP_WORDS
			         and len(word) > 2]
		elif language == 'vi':
			words = word_tokenize(text.lower(), format="text").split()
			words = [word for word in words
			         if not re.match(r'\W', word)
			         and len(word) > 1]
		return ' '.join(words)

	@staticmethod
	def extract_text_from_pdf (pdf_bytes):
		text = ""
		try:
			with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
				for page in pdf:
					text += page.get_text()
		except Exception as e:
			raise ValueError(f"Error processing PDF: {str(e)}")
		return text


class JobRecommender:
	def __init__ (self):
		self.vectorizer = None
		self.svd_model = None
		self.tfidf_matrix = None

	def prepare_models (self):
		if model_cache.needs_update():
			# Get all jobs from database
			jobs = Job.query.all()
			combined_texts = [job.combined_text for job in jobs]

			# Initialize and fit TF-IDF vectorizer
			self.vectorizer = TfidfVectorizer(
				max_features=5000,
				ngram_range=(1, 2),
				strip_accents='unicode',
				min_df=2,
				max_df=0.95
			)
			self.tfidf_matrix = self.vectorizer.fit_transform(combined_texts)

			# Initialize and fit SVD model
			self.svd_model = TruncatedSVD(n_components=min(100, self.tfidf_matrix.shape[1] - 1))
			self.tfidf_matrix = self.svd_model.fit_transform(self.tfidf_matrix)

			# Cache the models
			model_cache.save_models(self.vectorizer, self.svd_model, self.tfidf_matrix)
		else:
			# Load from cache
			model_cache.load_models()
			self.vectorizer = model_cache.vectorizer
			self.svd_model = model_cache.svd_model
			self.tfidf_matrix = model_cache.tfidf_matrix

	def recommend_jobs (self, cv_text, top_n=10):
		# Transform CV text using cached vectorizer
		cv_vector = self.vectorizer.transform([cv_text])
		cv_vector_reduced = self.svd_model.transform(cv_vector)

		# Calculate similarities
		similarities = cosine_similarity(cv_vector_reduced, self.tfidf_matrix).flatten()

		# Get top N similar jobs
		top_indices = np.argsort(similarities)[-top_n:][::-1]

		# Get jobs from database
		jobs = Job.query.all()

		recommendations = []
		for idx in top_indices:
			job = jobs[idx]
			recommendations.append({
				**job.to_dict(),
				'similarity_score': float(similarities[idx])
			})

		return recommendations


def download_google_drive_file (file_url):
	# Extract file ID from the URL
	file_id = file_url.split('/d/')[1].split('/view')[0]

	# Construct direct download URL
	download_url = f'https://drive.google.com/uc?id={file_id}'

	try:
		# Download file
		response = requests.get(download_url)
		response.raise_for_status()

		return response.content
	except requests.RequestException as e:
		raise ValueError(f"Failed to download file: {str(e)}")

# API Routes
@app.route('/api/recommend', methods=['POST'])
def recommend ():
	try:
		file_url = request.form.get('file_url')
		if not file_url:
			return jsonify({'error': 'No file URL provided'}), 400

		# Download file content
		pdf_bytes = download_google_drive_file(file_url)

		# Process CV
		cv_text = TextProcessor.extract_text_from_pdf(pdf_bytes)
		language = TextProcessor.detect_language(cv_text)
		processed_cv = TextProcessor.preprocess_text(cv_text, language)

		# Initialize recommender and get recommendations
		recommender = JobRecommender()
		recommender.prepare_models()
		recommendations = recommender.recommend_jobs(processed_cv)

		return jsonify({
			'status': 'success',
			'recommendations': recommendations
		})

	except Exception as e:
		return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
	with app.app_context():
		db.create_all()
	app.run(debug=True)