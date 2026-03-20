from pathlib import Path

from .ml_pipeline import HybridFakeNewsService


def bootstrap_application(app) -> None:
    Path(app.config["MODEL_ARTIFACTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["REPORTS_DIR"]).mkdir(parents=True, exist_ok=True)
    HybridFakeNewsService.from_app(app).ensure_model_artifacts()
