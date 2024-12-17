from sklearn.metrics.pairwise import cosine_similarity
from typing import  Dict, Any
import logging
import time
from caching import RedisJobCache
from text_processing import TextProcessor
from models import Job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerJobRecommender:
	def __init__ (self,redis_client,global_model):
		start_time = time.time()
		while global_model is None and time.time() - start_time < 30:
			time.sleep(0.1)

		if global_model is None:
			raise RuntimeError("Model initialization timeout")

		self.model = global_model
		self.cache = RedisJobCache(redis_client)
		self.job_embeddings = None
		self.job_ids = None

	def prepare_embeddings (self) -> None:
		try:
			logger.info(f"Preparing embeddings. Total jobs in database: {Job.query.count()}")
			cached_data = self.cache.load_cache()
			if cached_data[0] is not None:
				self.job_embeddings, self.job_ids = cached_data
				return

			self.cache.initialize_cache()
			cached_data = self.cache.load_cache()
			if cached_data[0] is not None:
				self.job_embeddings, self.job_ids = cached_data
		except Exception as e:
			logger.error(f"Error preparing embeddings: {str(e)}")
			raise

	def needs_cache_refresh (self):
		total_jobs = Job.query.count()

		if total_jobs != len(self.job_ids or []):
			return True
		return False

	def recommend_jobs (self, cv_text: str, limit: int = 20, page: int = 1, keyword: str = None,
	                    order_by: str = "similarity") -> Dict[str, Any]:
		try:
			if self.job_embeddings is None:
				self.prepare_embeddings()

			if self.job_embeddings is None or self.job_ids is None:
				raise ValueError("Failed to prepare job embeddings")

			language = TextProcessor.detect_language(cv_text)
			processed_cv = TextProcessor.preprocess_text(cv_text, language)
			cv_embedding = self.model.encode([processed_cv])[0]

			similarities = cosine_similarity([cv_embedding], self.job_embeddings)[0]

			jobs_with_scores = []
			for idx, job_id in enumerate(self.job_ids):
				job = Job.query.filter_by(id=job_id).first()
				if job is None:
					continue

				if keyword:
					if not (keyword.lower() in job.title.lower() or
					        (job.description and keyword.lower() in job.description.lower())):
						continue

				jobs_with_scores.append({
					'job': job,
					'score': float(similarities[idx])
				})

			if order_by == "date":
				jobs_with_scores.sort(key=lambda x: x['job'].created_at, reverse=True)
			else:
				jobs_with_scores.sort(key=lambda x: x['score'], reverse=True)



			total_items = len(jobs_with_scores)
			total_pages = (total_items + limit - 1) // limit
			start_idx = (page - 1) * limit
			end_idx = min(start_idx + limit, total_items)

			# Get paginated results
			paginated_jobs = jobs_with_scores[start_idx:end_idx]

			recommendations = [
				{
					**job_score['job'].to_dict(),
					'similarityScore': job_score['score']
				}
				for job_score in paginated_jobs
			]

			return {
				'data': recommendations,
				'totalItems': total_items,
				'totalPages': total_pages,
				'currentPage': page
			}

		except Exception as e:
			logger.error(f"Error generating recommendations: {str(e)}")
			raise

	def get_related_jobs (self, job_id: str, limit: int = 10, page: int = 1) -> Dict[str, Any]:
		try:
			# Prepare embeddings if not already done
			if self.job_embeddings is None:
				self.prepare_embeddings()

			if self.job_embeddings is None or self.job_ids is None:
				raise ValueError("Failed to prepare job embeddings")

			# Get the reference job from database
			reference_job = Job.query.filter_by(id=job_id).first()
			if not reference_job:
				raise ValueError(f"Job with ID {job_id} not found in database")

			# Try to find the job in cached embeddings
			try:
				ref_job_index = self.job_ids.index(job_id)
				ref_job_embedding = self.job_embeddings[ref_job_index]
				logger.info(f"Found job {job_id} in cache")
			except ValueError:
				# Job not in cache, generate embedding from job content
				logger.info(f"Job {job_id} not found in cache, generating new embedding")
				job_text = f"{reference_job.title} {reference_job.description or ''}"
				language = TextProcessor.detect_language(job_text)
				processed_job_text = TextProcessor.preprocess_text(job_text, language)
				ref_job_embedding = self.model.encode([processed_job_text])[0]

			# Calculate similarities with other jobs
			similarities = cosine_similarity([ref_job_embedding], self.job_embeddings)[0]

			# Create list of related jobs with their similarity scores
			seen_job_ids = set()  # Track seen job IDs
			related_jobs_with_scores = []

			for idx, other_job_id in enumerate(self.job_ids):
				# Skip if it's the same job or if we've seen this job before
				if other_job_id == job_id or other_job_id in seen_job_ids:
					continue

				job = Job.query.filter_by(id=other_job_id).first()
				if job is None:
					continue

				seen_job_ids.add(other_job_id)  # Mark this job ID as seen
				related_jobs_with_scores.append({
					'job': job,
					'score': float(similarities[idx])
				})

			# Sort related jobs by similarity score in descending order
			related_jobs_with_scores.sort(key=lambda x: x['score'], reverse=True)

			# Calculate pagination
			total_items = len(related_jobs_with_scores)
			total_pages = (total_items + limit - 1) // limit
			start_idx = (page - 1) * limit
			end_idx = min(start_idx + limit, total_items)

			# Get paginated results
			paginated_jobs = related_jobs_with_scores[start_idx:end_idx]

			recommendations = [
				{
					**job_score['job'].to_dict(),
					'similarityScore': job_score['score']
				}
				for job_score in paginated_jobs
			]

			return {
				'data': recommendations,
				'totalItems': total_items,
				'totalPages': total_pages,
				'currentPage': page
			}

		except Exception as e:
			logger.error(f"Error generating related jobs: {str(e)}")
			raise