from .base import db, BaseEntity

class CUser(BaseEntity):
	__tablename__ = 'c_user'

	username = db.Column(db.String(100), unique=True, nullable=False)
	password = db.Column(db.String(255), nullable=False)
	email = db.Column(db.String(255), nullable=False, unique=True)
	display_name = db.Column(db.String(255))
	phone = db.Column(db.String(13))
	avatar = db.Column(db.String(255))
	is_active = db.Column(db.Boolean, default=True)

	# Relationships
	company = db.relationship('Company', backref='owner', uselist=False)
	jobs = db.relationship('Job', backref='user', lazy='dynamic')
