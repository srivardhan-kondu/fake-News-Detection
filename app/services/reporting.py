import csv
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def build_submission_summary(title: str, analysis: dict) -> str:
    summary_title = title or "Untitled article"
    return (
        f"{summary_title}: classified as {analysis['predicted_label']} with "
        f"{analysis['confidence_score']:.2f}% confidence and "
        f"{analysis['credibility_score']:.2f}% credibility score."
    )


def build_distribution_data(submissions) -> dict:
    fake_count = sum(1 for item in submissions if item.predicted_label == "Fake News")
    real_count = sum(1 for item in submissions if item.predicted_label == "Real News")
    return {
        "labels": ["Fake News", "Real News"],
        "counts": [fake_count, real_count],
    }


def export_submission_csv(report_dir: str, submission) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    export_path = Path(report_dir) / f"submission_{submission.id}.csv"
    explanation = submission.explanation
    with export_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Field", "Value"])
        writer.writerow(["Title", submission.title or "Untitled submission"])
        writer.writerow(["Source Type", submission.source_type])
        writer.writerow(["Source URL", submission.source_url or ""])
        writer.writerow(["Predicted Label", submission.predicted_label])
        writer.writerow(["Confidence Score", f"{submission.confidence_score:.2f}"])
        writer.writerow(["Credibility Score", f"{submission.credibility_score:.2f}"])
        writer.writerow(["Summary", submission.report_summary])
        writer.writerow(["Influential Terms", ", ".join(explanation.get("influential_terms", []))])
        writer.writerow(["Explanation", " | ".join(explanation.get("insights", []))])
    return str(export_path)


def export_submission_pdf(report_dir: str, submission) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    export_path = Path(report_dir) / f"submission_{submission.id}.pdf"
    pdf = canvas.Canvas(str(export_path), pagesize=A4)
    width, height = A4
    y = height - 50
    lines = [
        "Hybrid Fake News Detection Report",
        f"Submission ID: {submission.id}",
        f"Title: {submission.title or 'Untitled submission'}",
        f"Source Type: {submission.source_type}",
        f"Source URL: {submission.source_url or 'N/A'}",
        f"Predicted Label: {submission.predicted_label}",
        f"Confidence Score: {submission.confidence_score:.2f}%",
        f"Credibility Score: {submission.credibility_score:.2f}%",
        f"Summary: {submission.report_summary}",
        "Insights:",
    ]

    for insight in submission.explanation.get("insights", []):
        lines.append(f"- {insight}")

    for line in lines:
        pdf.drawString(50, y, line[:110])
        y -= 18
        if y < 60:
            pdf.showPage()
            y = height - 50
    pdf.save()
    return str(export_path)
