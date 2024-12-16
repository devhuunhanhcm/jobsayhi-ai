from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class BaseEntity(db.Model):
	__abstract__ = True

	id = db.Column(db.String(36), primary_key=True)
	created_at = db.Column(db.DateTime, default=datetime.utcnow)
	last_modified_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
