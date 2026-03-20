import json
from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    submissions = db.relationship("Submission", backref="user", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(255), nullable=True)
    source_type = db.Column(db.String(20), nullable=False)
    source_url = db.Column(db.Text, nullable=True)
    raw_text = db.Column(db.Text, nullable=False)
    processed_text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(30), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    credibility_score = db.Column(db.Float, nullable=False)
    explanation_json = db.Column(db.Text, nullable=False)
    chart_json = db.Column(db.Text, nullable=False)
    model_breakdown_json = db.Column(db.Text, nullable=False)
    report_summary = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    @property
    def explanation(self):
        return json.loads(self.explanation_json)

    @property
    def chart_data(self):
        return json.loads(self.chart_json)

    @property
    def model_breakdown(self):
        return json.loads(self.model_breakdown_json)
