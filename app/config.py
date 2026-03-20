import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'instance' / 'fake_news.db'}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_ARTIFACTS_DIR = str(BASE_DIR / "app" / "models_artifacts")
    DATASET_PATH = str(BASE_DIR / "app" / "data" / "sample_news.csv")
    REPORTS_DIR = str(BASE_DIR / "reports")
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    WTF_CSRF_TIME_LIMIT = None
