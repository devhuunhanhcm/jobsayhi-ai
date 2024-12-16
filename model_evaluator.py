from sklearn.metrics import precision_score, recall_score, ndcg_score, average_precision_score
from typing import List, Dict, Tuple
import numpy as np
import time
from datetime import datetime


class ModelEvaluator:
	def __init__ (self, recommender):
		self.recommender = recommender
		self.metrics = {}
		self.latency_stats = []

	def evaluate_relevance_metrics (self, test_cases: List[Dict]) -> Dict:
		"""
		Evaluate recommendation relevance using standard IR metrics

		test_cases: List of dicts containing {'cv_text': str, 'relevant_job_ids': List[str]}
		"""
		precisions = []
		recalls = []
		ndcg_scores = []
		map_scores = []

		for test_case in test_cases:
			# Get recommendations
			recommendations = self.recommender.recommend_jobs(
				test_case['cv_text'],
				limit=20
			)

			recommended_ids = [job['id'] for job in recommendations['data']]
			relevant_ids = set(test_case['relevant_job_ids'])

			# Calculate binary relevance
			y_true = [1 if job_id in relevant_ids else 0 for job_id in recommended_ids]
			y_score = [job['similarityScore'] for job in recommendations['data']]

			# Precision and Recall
			precisions.append(precision_score(y_true, [1] * len(y_score), zero_division=0))
			recalls.append(recall_score(y_true, [1] * len(y_score), zero_division=0))

			# NDCG
			if len(relevant_ids) > 0:
				ndcg_scores.append(ndcg_score([y_true], [y_score]))

			# Mean Average Precision
			map_scores.append(average_precision_score(y_true, y_score))

		return {
			'precision': np.mean(precisions),
			'recall': np.mean(recalls),
			'ndcg': np.mean(ndcg_scores),
			'map': np.mean(map_scores)
		}

	def evaluate_latency (self, test_cvs: List[str], n_runs: int = 100) -> Dict:
		"""
		Evaluate model latency metrics
		"""
		latencies = []

		for _ in range(n_runs):
			for cv_text in test_cvs:
				start_time = time.time()
				self.recommender.recommend_jobs(cv_text)
				latency = time.time() - start_time
				latencies.append(latency)

		return {
			'mean_latency': np.mean(latencies),
			'p95_latency': np.percentile(latencies, 95),
			'p99_latency': np.percentile(latencies, 99),
			'min_latency': np.min(latencies),
			'max_latency': np.max(latencies)
		}

	def evaluate_memory_usage (self) -> Dict:
		"""
		Evaluate model memory usage
		"""
		import psutil
		import os

		process = psutil.Process(os.getpid())
		memory_info = process.memory_info()

		return {
			'rss_memory_mb': memory_info.rss / (1024 * 1024),
			'vms_memory_mb': memory_info.vms / (1024 * 1024)
		}

	def generate_test_report (self, test_cases: List[Dict], test_cvs: List[str]) -> Dict:
		"""
		Generate comprehensive test report
		"""
		relevance_metrics = self.evaluate_relevance_metrics(test_cases)
		latency_metrics = self.evaluate_latency(test_cvs)
		memory_metrics = self.evaluate_memory_usage()

		return {
			'timestamp': datetime.now().isoformat(),
			'relevance_metrics': relevance_metrics,
			'latency_metrics': latency_metrics,
			'memory_metrics': memory_metrics
		}

def create_test_cases (job_db: List[Dict], n_cases: int = 100) -> List[Dict]:
	"""
	Create test cases from existing job database
	"""
	test_cases = []

	for _ in range(n_cases):
		# Randomly select a job as the "CV"
		cv_job = np.random.choice(job_db)

		# Find similar jobs based on category, skills, etc.
		relevant_jobs = [
			job['id'] for job in job_db
			if job['category_id'] == cv_job['category_id']
			   and job['id'] != cv_job['id']
		]

		test_cases.append({
			'cv_text': cv_job['combined_text'],
			'relevant_job_ids': relevant_jobs
		})

	return test_cases