from .base import db

class Category(db.Model):
	__tablename__ = 'category'

	id = db.Column(db.String(36), primary_key=True)
	name = db.Column(db.String(255), nullable=False)
