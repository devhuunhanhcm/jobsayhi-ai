import logging
import pickle
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
from text_processing import TextProcessor
from models import Job

logger = logging.getLogger(__name__)

class RedisJobCache:
	def __init__ (self, redis_client):
		self.redis = redis_client
		self.embeddings_key = 'job_embeddings'
		self.job_ids_key = 'job_ids'
		self.last_update_key = 'last_update'

	def clear_all_cache (self) -> bool:
		try:
			# Delete specific keys
			keys_to_delete = [
				self.embeddings_key,
				self.job_ids_key
			]

			# Delete all keys
			deleted = self.redis.delete(*keys_to_delete)

			logger.info(f"Successfully cleared {deleted} cache keys")
			return True

		except Exception as e:
			logger.error(f"Error clearing cache: {str(e)}")
			return False

	def save_cache (self, embeddings: np.ndarray, job_ids: List[str]) -> None:
		"""
		Save job embeddings and related job IDs to Redis cache

		Args:
				embeddings (np.ndarray): Array of job embeddings
				job_ids (List[str]): Corresponding job IDs
		"""
		try:
			pipeline = self.redis.pipeline()
			pipeline.set(self.embeddings_key, pickle.dumps(embeddings))
			pipeline.set(self.job_ids_key, pickle.dumps(job_ids))
			pipeline.set(self.last_update_key, str(datetime.now().timestamp()))
			pipeline.execute()
			logger.info(f"Cached {len(job_ids)} job embeddings")
		except Exception as e:
			logger.error(f"Error saving cache: {str(e)}")

	def load_cache (self) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
		"""
		Load job embeddings and job IDs from Redis cache

		Returns:
				Tuple of embeddings array and job IDs list
		"""
		try:
			embeddings_bytes = self.redis.get(self.embeddings_key)
			job_ids_bytes = self.redis.get(self.job_ids_key)

			if embeddings_bytes and job_ids_bytes:
				return (
					pickle.loads(embeddings_bytes),
					pickle.loads(job_ids_bytes)
				)
			return None, None
		except Exception as e:
			logger.error(f"Error loading cache: {str(e)}")
			return None, None

	def add_job_embedding_to_cache (self, job_id: str, embedding: np.ndarray) -> None:
		try:
			# Load existing cache
			current_embeddings, current_job_ids = self.load_cache()

			if current_embeddings is None or current_job_ids is None:
				# Initialize cache if it doesn't exist
				current_embeddings = embedding.reshape(1, -1)
				current_job_ids = [job_id]
			else:
				# Append new embedding and job ID
				current_embeddings = np.vstack([current_embeddings, embedding])
				current_job_ids.append(job_id)

			# Save updated cache
			self.save_cache(current_embeddings, current_job_ids)

		except Exception as e:
			logger.error(f"Error adding job embedding to cache: {str(e)}")

	def initialize_cache (self) -> None:
		try:
			logger.info("Starting cache initialization...")

			# Fetch all jobs from the database
			jobs = Job.query.all()
			logger.info(f"Total jobs found: {len(jobs)}")

			embeddings = []
			job_ids = []

			for job in jobs:
				try:
					if hasattr(job, 'job_embedding') and job.job_embedding:
							existing_embedding = json.loads(job.job_embedding)
							embedding = np.array(existing_embedding)
					else:
						continue

					embeddings.append(embedding)
					job_ids.append(job.id)

				except Exception as job_error:
					logger.error(f"Error processing job {job.id}: {str(job_error)}")
					continue

			# Convert embeddings to numpy array
			if embeddings:
				embeddings_array = np.array(embeddings)

				# Save embeddings and job IDs to Redis cache
				pipeline = self.redis.pipeline()
				pipeline.set(self.embeddings_key, pickle.dumps(embeddings_array))
				pipeline.set(self.job_ids_key, pickle.dumps(job_ids))
				pipeline.set(self.last_update_key, str(datetime.now().timestamp()))
				pipeline.execute()

				logger.info(f"Cache initialized with {len(job_ids)} job embeddings")
			else:
				logger.warning("No job embeddings could be generated")

		except Exception as e:
			logger.error(f"Error initializing cache: {str(e)}")
			raise


