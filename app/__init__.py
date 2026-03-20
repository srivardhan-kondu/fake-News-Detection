from flask import Flask

from .config import Config
from .extensions import csrf, db, login_manager
from .routes import main_bp
from .services.bootstrap import bootstrap_application


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    app.register_blueprint(main_bp)

    # Exempt JSON API endpoints from CSRF (session cookie auth still applies)
    from .routes import api_analyze, api_history, api_metrics, api_submission
    csrf.exempt(api_analyze)
    csrf.exempt(api_history)
    csrf.exempt(api_metrics)
    csrf.exempt(api_submission)

    with app.app_context():
        db.create_all()
        bootstrap_application(app)

    return app
