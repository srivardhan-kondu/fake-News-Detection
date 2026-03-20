from app import create_app
from app.services.ml_pipeline import HybridFakeNewsService


app = create_app()


with app.app_context():
    HybridFakeNewsService.from_app(app).train_models()
    print("Training complete.")
