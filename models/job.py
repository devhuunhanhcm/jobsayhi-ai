from .base import db, BaseEntity
from typing import Dict, Any

class Job(BaseEntity):
	__tablename__ = 'job'

	title = db.Column(db.String(255), nullable=False)
	salary = db.Column(db.String(255))
	location = db.Column(db.String(255))
	working_location = db.Column(db.String(255))
	experience = db.Column(db.String(255))
	position = db.Column(db.String(255))
	status = db.Column(db.String(50))
	description = db.Column(db.Text)
	job_requirements = db.Column(db.Text)
	benefits = db.Column(db.Text)
	working_time = db.Column(db.String(255))
	deadline = db.Column(db.DateTime)
	job_embedding = db.Column(db.Text,nullable=True)

	# Foreign Keys
	user_id = db.Column(db.String(36), db.ForeignKey('c_user.id'), nullable=False)
	category_id = db.Column(db.String(36), db.ForeignKey('category.id'))

	# Relationships
	category = db.relationship('Category')

	@property
	def combined_text (self) -> str:
		return f"{self.title} {self.description} {self.job_requirements or ''} {self.location or ''}"

	def to_dict (self, similarity_score: float = None) -> Dict[str, Any]:
		result = {
			'id': self.id,
			'categoryId': self.category_id,
			'company': self.user.company.to_dict() if self.user and self.user.company else None,
			'createAt': self.created_at.isoformat() if self.created_at else None,
			'lastModifiedAt': self.last_modified_at.isoformat() if self.last_modified_at else None,
			'title': self.title,
			'salary': self.salary,
			'location': self.location,
			'experience': self.experience,
			'position': self.position,
			'status': self.status,
			'description': self.description,
			'requirements': self.job_requirements,
			'benefits': self.benefits,
			'workingTime': self.working_time,
			'workingLocation': self.working_location,
			'deadline': self.deadline.isoformat() if self.deadline else None
		}

		if similarity_score is not None:
			result['similarity_score'] = similarity_score

		return result
