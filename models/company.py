from .base import db
from typing import Dict, Any

class Company(db.Model):
	__tablename__ = 'company'

	id = db.Column(db.String(36), primary_key=True)
	owner_id = db.Column(db.String(36), db.ForeignKey('c_user.id'))
	avatar_url = db.Column(db.String(255))
	name = db.Column(db.String(255), nullable=False)
	address = db.Column(db.Text)
	email = db.Column(db.String(255))
	no_employees = db.Column(db.String(255))
	introduction = db.Column(db.Text)
	establish_date = db.Column(db.DateTime)
	is_verified = db.Column(db.Boolean, default=False)
	phone = db.Column(db.String(13))

	def to_dict (self) -> Dict[str, Any]:
		return {
			'name': self.name,
			'avatarUrl': self.avatar_url or '',
			'address': self.address,
			'email': self.email,
			'noEmployees': self.no_employees,
			'introduction': self.introduction,
			'establishDate': self.establish_date.isoformat() if self.establish_date else None,
			'phone': self.phone,
			'verified': self.is_verified
		}
