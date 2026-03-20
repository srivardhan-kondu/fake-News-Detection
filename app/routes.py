import json

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user

from .extensions import csrf, db
from .forms import LoginForm, RegisterForm
from .models import Submission, User
from .services.ml_pipeline import HybridFakeNewsService
from .services.reporting import build_submission_summary, export_submission_csv, export_submission_pdf
from .services.scraper import extract_article


main_bp = Blueprint("main", __name__)


def get_detector() -> HybridFakeNewsService:
    return HybridFakeNewsService.from_app(current_app)


def serialize_submission(submission: Submission) -> dict:
    return {
        "id": submission.id,
        "title": submission.title or "Untitled submission",
        "source_type": submission.source_type,
        "source_url": submission.source_url,
        "predicted_label": submission.predicted_label,
        "confidence_score": submission.confidence_score,
        "credibility_score": submission.credibility_score,
        "explanation": submission.explanation,
        "charts": submission.chart_data,
        "model_breakdown": submission.model_breakdown,
        "report_summary": submission.report_summary,
        "created_at": submission.created_at.strftime("%d %b %Y %H:%M"),
    }


# ---------------------------------------------------------------------------
# Page routes (server-rendered shells)
# ---------------------------------------------------------------------------

@main_bp.route("/")
def index():
    metrics = get_detector().get_metrics()
    return render_template("index.html", metrics=metrics)


@main_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = RegisterForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data.strip(),
            email=form.email.data.strip().lower(),
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Log in to continue.", "success")
        return redirect(url_for("main.login"))
    return render_template("register.html", form=form)


@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.strip().lower()).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html", form=form)


@main_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.index"))


@main_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@main_bp.route("/history")
@login_required
def history():
    return render_template("history.html")


# ---------------------------------------------------------------------------
# JSON API endpoints (consumed by client-side JS)
# ---------------------------------------------------------------------------

@main_bp.route("/api/metrics")
@login_required
def api_metrics():
    return jsonify(get_detector().get_metrics())


@main_bp.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    article_text = (data.get("article_text") or "").strip()
    article_url = (data.get("article_url") or "").strip()

    if not article_text and not article_url:
        return jsonify({"error": "Provide article text or a valid URL."}), 400

    source_type = "manual"
    source_url = None
    raw_text = article_text

    if article_url:
        source_type = "url"
        source_url = article_url
        try:
            extracted = extract_article(source_url)
        except Exception as exc:
            return jsonify({"error": f"Could not extract article: {exc}"}), 400
        title = title or extracted["title"]
        raw_text = extracted["text"]

    if len(raw_text) < 50:
        return jsonify({"error": "Article text must be at least 50 characters."}), 400

    detector = get_detector()
    try:
        analysis = detector.analyze(title=title, raw_text=raw_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    submission = Submission(
        user_id=current_user.id,
        title=title,
        source_type=source_type,
        source_url=source_url,
        raw_text=raw_text,
        processed_text=analysis["processed_text"],
        predicted_label=analysis["predicted_label"],
        confidence_score=analysis["confidence_score"],
        credibility_score=analysis["credibility_score"],
        explanation_json=json.dumps(analysis["explanation"]),
        chart_json=json.dumps(analysis["charts"]),
        model_breakdown_json=json.dumps(analysis["model_breakdown"]),
        report_summary=build_submission_summary(title, analysis),
    )
    db.session.add(submission)
    db.session.commit()

    return jsonify(serialize_submission(submission)), 201


@main_bp.route("/api/history")
@login_required
def api_history():
    submissions = (
        Submission.query.filter_by(user_id=current_user.id)
        .order_by(Submission.created_at.desc())
        .all()
    )
    return jsonify([serialize_submission(s) for s in submissions])


@main_bp.route("/api/submission/<int:submission_id>")
@login_required
def api_submission(submission_id: int):
    submission = Submission.query.filter_by(id=submission_id, user_id=current_user.id).first_or_404()
    return jsonify(serialize_submission(submission))


# ---------------------------------------------------------------------------
# File exports (still server-side)
# ---------------------------------------------------------------------------

@main_bp.route("/report/<int:submission_id>/csv")
@login_required
def download_report_csv(submission_id: int):
    submission = Submission.query.filter_by(id=submission_id, user_id=current_user.id).first_or_404()
    export_path = export_submission_csv(current_app.config["REPORTS_DIR"], submission)
    return send_file(export_path, as_attachment=True)


@main_bp.route("/report/<int:submission_id>/pdf")
@login_required
def download_report_pdf(submission_id: int):
    submission = Submission.query.filter_by(id=submission_id, user_id=current_user.id).first_or_404()
    export_path = export_submission_pdf(current_app.config["REPORTS_DIR"], submission)
    return send_file(export_path, as_attachment=True)


@main_bp.route("/health")
def health():
    return Response("ok", mimetype="text/plain")
