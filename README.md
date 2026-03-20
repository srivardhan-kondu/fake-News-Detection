# Hybrid AI-Powered Fake News Detection System

> A production-ready web application that classifies news articles as **Real** or **Fake** using a three-stage pipeline: classical machine learning, deep learning, and a dynamic ensemble decision layer. Built with Flask, scikit-learn, TensorFlow/Keras, and Chart.js.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Text Preprocessing Pipeline](#5-text-preprocessing-pipeline)
6. [Machine Learning Models](#6-machine-learning-models)
   - [TF-IDF Vectorizer](#61-tf-idf-vectorizer)
   - [Logistic Regression](#62-logistic-regression)
   - [Linear SVM](#63-linear-svm)
   - [Multinomial Naive Bayes](#64-multinomial-naive-bayes)
7. [Deep Learning Model — Bidirectional LSTM](#7-deep-learning-model--bidirectional-lstm)
8. [Hybrid Ensemble Decision Layer](#8-hybrid-ensemble-decision-layer)
9. [Explainability — Influential Keywords](#9-explainability--influential-keywords)
10. [Scoring System](#10-scoring-system)
11. [Database Schema](#11-database-schema)
12. [API Reference](#12-api-reference)
13. [Performance Results](#13-performance-results)
14. [Setup on macOS](#14-setup-on-macos)
15. [Setup on Windows](#15-setup-on-windows)
16. [Training the Models](#16-training-the-models)
17. [Environment Variables](#17-environment-variables)
18. [Functional Requirements Coverage](#18-functional-requirements-coverage)
19. [Tech Stack Summary](#19-tech-stack-summary)

---

## 1. Abstract

Misinformation spreads faster than corrections. This system provides an **authenticated, multi-model fake news detection pipeline** that goes beyond a single classifier — it runs three ML models and one deep learning model in parallel on every article, then applies a data-driven ensemble to produce a final verdict.

The design goal is **transparency**: the user sees *every* model's individual prediction, probability, and training metrics, plus the exact mathematical reason why the system chose its final answer. Articles can be submitted as raw text or by pasting a URL (the backend scrapes the content automatically). All analyses are stored per-user and can be exported as PDF or CSV reports.

**Key metrics on the WELFake dataset (15,000-sample stratified subset):**
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 92.56% | 92.68% |
| Linear SVM (best) | 93.12% | 93.22% |
| Multinomial Naive Bayes | 84.93% | 85.31% |
| Bidirectional LSTM | 88.45% | 88.52% |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                             │
│  HTML shell served by Flask → JavaScript SPA (fetch API + Chart.js) │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP / JSON REST
┌────────────────────────▼────────────────────────────────────────────┐
│                     Flask Application (Python 3.11)                 │
│                                                                     │
│  Routes: /  /dashboard  /history  /login  /register                 │
│  API:    /api/analyze  /api/metrics  /api/history  /api/submission  │
│  Auth:   Flask-Login (session cookies)                              │
│  Forms:  Flask-WTF (CSRF protection)                                │
└──────┬──────────────────────────┬──────────────────────────────────┘
       │                          │
┌──────▼──────────┐    ┌──────────▼──────────────────────────────────┐
│   SQLite DB      │    │           HybridFakeNewsService             │
│   (SQLAlchemy)  │    │                                             │
│   - User        │    │  1. preprocess_text()                       │
│   - Submission  │    │  2. _predict_all_ml()  ─── LR + SVM + NB   │
└─────────────────┘    │  3. _predict_dl()      ─── BiLSTM           │
                       │  4. Ensemble decision layer                  │
                       │  5. Return full transparency payload         │
                       └─────────────────────────────────────────────┘
                                          │
                         ┌────────────────▼───────────────────┐
                         │         Model Artifacts             │
                         │  vectorizer.joblib                  │
                         │  ml_models.joblib  (LR + SVM + NB) │
                         │  dl_tokenizer.joblib                │
                         │  dl_lstm.keras                      │
                         │  metadata.json                      │
                         └─────────────────────────────────────┘
```

**Request lifecycle for a single article analysis:**
```
Browser ──POST /api/analyze──► Flask route
    └─► scrape URL (if provided) via BeautifulSoup
    └─► preprocess_text() — normalize, tokenize, filter, deduplicate
    └─► TF-IDF transform  (4,000-feature unigram + bigram space)
    └─► Run LR → probability_lr
    └─► Run SVM → probability_svm  (via Platt scaling on decision_function)
    └─► Run NB  → probability_nb
    └─► Select best_ml result by F1; extract influential TF-IDF terms
    └─► Tokenizer → pad sequences (maxlen=150)
    └─► BiLSTM.predict → probability_dl
    └─► Ensemble: |p_ml - p_dl| > 0.35 → best F1 model wins
                  else → weighted blend (F1-proportional weights)
    └─► Compute confidence, credibility
    └─► Save Submission to DB
    └─► Return JSON 201 with full transparency payload
Browser ◄──JSON──────────────── renderResult() draws all panels
```

---

## 3. Project Structure

```
Fake News Detection/
│
├── run.py                          # App entry point
├── requirements.txt                # All Python dependencies
│
├── app/
│   ├── __init__.py                 # Flask app factory; CSRF exemptions
│   ├── config.py                   # Config: paths, secret key, DB URI
│   ├── extensions.py               # db, login_manager, csrf instances
│   ├── forms.py                    # RegisterForm, LoginForm, AnalyzeForm
│   ├── models.py                   # SQLAlchemy: User, Submission
│   ├── routes.py                   # Page routes + JSON API endpoints
│   │
│   ├── services/
│   │   ├── ml_pipeline.py          # Core: training + full inference service
│   │   ├── preprocessing.py        # normalize, tokenize, stopword filter
│   │   ├── scraper.py              # URL → article text (BeautifulSoup)
│   │   ├── reporting.py            # PDF (ReportLab) + CSV export
│   │   └── bootstrap.py           # Ensure artifacts exist on startup
│   │
│   ├── data/
│   │   └── sample_news.csv         # WELFake dataset (72,134 articles)
│   │
│   ├── models_artifacts/           # Auto-generated on first run
│   │   ├── vectorizer.joblib
│   │   ├── ml_models.joblib
│   │   ├── dl_tokenizer.joblib
│   │   ├── dl_lstm.keras
│   │   └── metadata.json
│   │
│   ├── static/
│   │   ├── css/style.css           # Light theme, responsive (3 breakpoints)
│   │   └── js/app.js               # Full SPA: fetch API, Chart.js rendering
│   │
│   └── templates/
│       ├── base.html               # Shell: nav, loading overlay, flash
│       ├── index.html              # Home page with model summary
│       ├── dashboard.html          # SPA shell (form + full-width panels)
│       ├── history.html            # Analysis history table
│       ├── login.html
│       └── register.html
│
├── scripts/
│   └── train_models.py             # Standalone training script
│
├── instance/
│   └── fake_news.db                # SQLite database (auto-created)
│
└── reports/                        # Generated PDF/CSV exports
```

---

## 4. Dataset

**Name:** WELFake (Wikipedia and Emergent Liar Fake news dataset)  
**Size:** 72,134 labeled news articles  
**Columns:** `title`, `text`, `label` (0 = Real, 1 = Fake)  
**Source:** Originally published on Kaggle (`saurabhshahane/fake-news-classification`)

**Training strategy:** A stratified sample of **15,000 articles** (7,500 real + 7,500 fake) is drawn at training time to keep wall-clock training practical while maintaining class balance. The remaining ~57,000 articles are not used during training but remain available for future experiments.

**Train/test split:** 75% train (11,250) / 25% test (3,750), stratified by label, `random_state=42`.

---

## 5. Text Preprocessing Pipeline

Every article passes through an identical pipeline before being fed to any model:

```
Raw text
  │
  ├─ 1. Lowercase all characters
  ├─ 2. Remove URLs  (regex: https?://\S+)
  ├─ 3. Remove non-alphabetic characters  ([^a-z0-9\s] → space)
  ├─ 4. Collapse multiple spaces
  ├─ 5. Tokenize  (word-level regex: [a-z0-9']+)
  ├─ 6. Remove English stopwords  (scikit-learn ENGLISH_STOP_WORDS)
  ├─ 7. Remove duplicate tokens  (preserve first occurrence order)
  └─ Rejoin as a single string
```

**Code location:** `app/services/preprocessing.py` — `preprocess_text()`

The same function is used at **training time** (to build all feature matrices) and **inference time** (on the submitted article), ensuring there is no distribution shift between the two.

---

## 6. Machine Learning Models

All three ML models operate on a shared **TF-IDF feature matrix**.

### 6.1 TF-IDF Vectorizer

TF-IDF (Term Frequency–Inverse Document Frequency) converts the preprocessed text into a numeric feature vector.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{1 + \text{DF}(t)}$$

Where:
- $t$ = term, $d$ = document, $N$ = total documents
- $\text{TF}(t,d)$ = frequency of term $t$ in document $d$
- $\text{DF}(t)$ = number of documents containing term $t$

**Configuration:**
| Parameter | Value | Reason |
|---|---|---|
| `max_features` | 4,000 | Caps vocabulary at the 4,000 most informative terms |
| `ngram_range` | (1, 2) | Captures both single words *and* two-word phrases |

---

### 6.2 Logistic Regression

A linear classifier that models the log-odds of a document being Fake News:

$$\log\frac{P(\text{Fake} \mid \mathbf{x})}{P(\text{Real} \mid \mathbf{x})} = \mathbf{w}^T \mathbf{x} + b$$

The sigmoid converts log-odds to a calibrated probability:

$$P(\text{Fake} \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Trained with **L2 regularisation** and **lbfgs solver** (`max_iter=2000`).  
**Role:** Produces native `predict_proba` probabilities, also supplies influential term extraction via `coef_` weights.

---

### 6.3 Linear SVM

Finds the maximum-margin hyperplane separating real and fake document vectors:

$$\text{minimize} \quad \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$$

LinearSVC does not output probabilities natively. The system applies **Platt scaling** to convert the raw decision function margin to a probability:

$$P(\text{Fake}) = \frac{1}{1 + e^{-m}} \quad \text{where } m = \text{decision\_function}(\mathbf{x})$$

**Configuration:** `loss='squared_hinge'`, `penalty='l2'`, `max_iter=1000`  
**Role:** Best-performing ML model on the WELFake subset (F1 = 93.22%). Designated `best_ml_model` in `metadata.json`; its prediction takes priority in ensemble conflict resolution.

---

### 6.4 Multinomial Naive Bayes

Applies Bayes' theorem assuming conditional independence of features given the class:

$$P(\text{Fake} \mid \mathbf{x}) = \frac{P(\text{Fake}) \prod_{i=1}^{n} P(x_i \mid \text{Fake})}{P(\mathbf{x})}$$

With Laplace smoothing (`alpha=1.0`) to handle unseen n-grams:

$$P(x_i \mid c) = \frac{\text{count}(x_i, c) + \alpha}{\sum_j \text{count}(x_j, c) + \alpha \cdot |V|}$$

**Configuration:** `alpha=1.0`, `fit_prior=True`  
**Role:** Fast probabilistic baseline. Lower accuracy than SVM/LR on TF-IDF features but adds diversity to the model pool and participates in the transparency breakdown.

---

## 7. Deep Learning Model — Bidirectional LSTM

The deep learning branch operates on a **word-index sequence** representation, allowing the model to capture contextual and positional meaning that bag-of-words models miss.

### Architecture

```
Input: integer sequence of length 150
      │
      ▼
┌─────────────────────────────────────────┐
│  Embedding(input_dim=5000, output_dim=64)│  → shape: (batch, 150, 64)
│  Learns a 64-dim dense vector per word  │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Bidirectional(LSTM(units=32))          │  → shape: (batch, 64)
│  Forward LSTM ──────────────────────►  │
│  Backward LSTM ◄────────────────────── │
│  Concatenated: captures left & right   │
│  context for every position             │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Dense(16, activation='relu')           │  → shape: (batch, 16)
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Dense(1, activation='sigmoid')         │  → P(Fake) ∈ [0, 1]
└─────────────────────────────────────────┘
```

### LSTM Cell (per timestep)

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate)}$$
$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{\tilde{C}}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \cdot \tanh(C_t) \quad \text{(hidden state)}$$

### Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Epochs | Up to 5 (EarlyStopping patience=2) |
| Batch size | 8 |
| Validation split | 20% of training data |
| Vocab size | 5,000 tokens |
| Sequence max length | 150 tokens (zero-padded, post) |

**Role:** Captures sequential and contextual information. Complements the ML models which are bag-of-words and have no notion of word order.

---

## 8. Hybrid Ensemble Decision Layer

After all four models produce individual probabilities, the system applies a two-stage decision rule:

### Stage 1 — Weighted Ensemble Blend

Weights are computed **automatically at training time** proportional to each group's F1 score on the held-out test set:

$$w_{ML} = \frac{F1_{ML}}{F1_{ML} + F1_{DL}}, \quad w_{DL} = \frac{F1_{DL}}{F1_{ML} + F1_{DL}}$$

$$P_{hybrid} = w_{ML} \cdot P_{best\_ML} + w_{DL} \cdot P_{DL}$$

*Example from training:* ML F1=93.22%, DL F1=88.52%  
→ $w_{ML}=0.513$, $w_{DL}=0.487$

### Stage 2 — Disagreement Fallback

If the ML and DL branches disagree strongly, the blend would average out conflicting signals and produce an unreliable middle value. The system detects this and overrides:

$$\text{gap} = \left| P_{best\_ML} - P_{DL} \right|$$

$$\text{strategy} = \begin{cases} \text{best performing model (by F1)} & \text{if gap} > 0.35 \\ P_{hybrid} & \text{otherwise} \end{cases}$$

**When gap > 35%:** The single model with the higher training F1 score is used exclusively. This avoids averaging a strong signal (e.g., 90% fake) with a weak one (e.g., 40% fake) into an ambiguous 65%.

### Final classification

$$\hat{y} = \begin{cases} \text{Fake News} & P_{final} \geq 0.5 \\ \text{Real News} & P_{final} < 0.5 \end{cases}$$

**Code location:** `app/services/ml_pipeline.py` — `analyze()` method

---

## 9. Explainability — Influential Keywords

For every ML model that exposes `coef_` (Logistic Regression and Linear SVM), the system computes per-term contribution scores:

$$\text{contribution}(t) = \text{TF-IDF}(t, \mathbf{x}) \times w_t$$

Where $w_t$ is the model's learned weight for term $t$ and `TF-IDF(t, x)` is the feature value in the current document. Terms are ranked by their contribution in the direction of the final prediction (highest positive contributions for Fake News predictions, lowest negative for Real News).

The **top 6 terms** are surfaced in the UI as influential keywords. These terms are derived from the **best ML model's** coefficient vector.

---

## 10. Scoring System

After the final probability is determined:

$$\text{Confidence Score} = \max(P_{final}, 1 - P_{final}) \times 100$$

This is always ≥ 50%, expressing how decisive the prediction is regardless of direction.

$$\text{Credibility Score} = (1 - P_{final}) \times 100$$

This is a human-readable score: high value = high credibility (closer to Real News). A Fake News article near 95% fake probability would score ~5% credibility.

---

## 11. Database Schema

### `user` table
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `username` | VARCHAR(80) | Unique |
| `email` | VARCHAR(255) | Unique, lowercase |
| `password_hash` | VARCHAR(255) | Werkzeug PBKDF2-SHA256 |
| `created_at` | DATETIME | Auto |

### `submission` table
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `user_id` | INTEGER | FK → user.id |
| `title` | VARCHAR(255) | Article headline |
| `source_type` | VARCHAR(20) | `"manual"` or `"url"` |
| `source_url` | TEXT | Nullable |
| `raw_text` | TEXT | Original input |
| `processed_text` | TEXT | After preprocessing |
| `predicted_label` | VARCHAR(30) | `"Fake News"` or `"Real News"` |
| `confidence_score` | FLOAT | 50–100% |
| `credibility_score` | FLOAT | 0–100% |
| `explanation_json` | TEXT | `{influential_terms, insights}` |
| `chart_json` | TEXT | `{word_frequency}` |
| `model_breakdown_json` | TEXT | Full per-model predictions |
| `report_summary` | TEXT | Human-readable summary |
| `created_at` | DATETIME | Auto |

---

## 12. API Reference

All JSON API endpoints require an authenticated session (login first).

| Method | Endpoint | Description | Body / Params |
|---|---|---|---|
| GET | `/api/metrics` | Model metadata, architecture, ensemble weights | — |
| POST | `/api/analyze` | Analyze an article | `{title, article_text, article_url}` |
| GET | `/api/history` | Last 50 submissions for current user | — |
| GET | `/api/submission/<id>` | Single submission by ID | — |
| GET | `/report/<id>/pdf` | Download PDF report | — |
| GET | `/report/<id>/csv` | Download CSV report | — |

**POST `/api/analyze` — response (201)**
```json
{
  "id": 42,
  "predicted_label": "Fake News",
  "confidence_score": 87.3,
  "credibility_score": 12.7,
  "explanation": {
    "influential_terms": ["claim", "allegedly", "sources say"],
    "insights": ["Final classification used the hybrid ensemble strategy.", "..."]
  },
  "model_breakdown": {
    "selected_strategy": "hybrid ensemble",
    "decision_reason": "ML and DL agreed closely (gap 8.2%). A weighted ensemble was used...",
    "ensemble_weights": {"ml": 0.513, "dl": 0.487},
    "individual_predictions": [
      {"model_name": "Logistic Regression", "model_type": "ML", "probability_fake": 85.1, "prediction": "Fake News", "is_best_ml": false, "f1_score": 0.9268, "accuracy": 0.9256},
      {"model_name": "Linear Svm", "model_type": "ML", "probability_fake": 89.4, "prediction": "Fake News", "is_best_ml": true, "f1_score": 0.9322, "accuracy": 0.9312},
      {"model_name": "Naive Bayes", "model_type": "ML", "probability_fake": 76.2, "prediction": "Fake News", "is_best_ml": false, "f1_score": 0.8531, "accuracy": 0.8493},
      {"model_name": "BiLSTM (Deep Learning)", "model_type": "DL", "probability_fake": 81.1, "prediction": "Fake News", "is_best_ml": false, "f1_score": 0.8852, "accuracy": 0.8845}
    ],
    "ml_probability_fake": 89.4,
    "dl_probability_fake": 81.1,
    "hybrid_probability_fake": 85.5
  }
}
```

---

## 13. Performance Results

Trained on 15,000 stratified samples from WELFake. Test set: 3,750 samples.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 92.56% | 91.18% | 94.24% | 92.68% |
| **Linear SVM** | **93.12%** | **91.91%** | **94.56%** | **93.22%** |
| Multinomial Naive Bayes | 84.93% | 83.22% | 87.52% | 85.31% |
| Bidirectional LSTM | 88.45% | 87.99% | 89.07% | 88.52% |
| **Hybrid Ensemble** | — | — | — | **~93.5%*** |

*Hybrid ensemble result is estimated; varies per article depending on ML–DL agreement.

---

## 14. Setup on macOS

### Prerequisites
- macOS 12 Monterey or later
- Python 3.11 (install via [python.org](https://www.python.org/downloads/) or `brew install python@3.11`)
- Git

### Step-by-Step

**1. Clone the repository**
```bash
git clone <repository-url> "Fake News Detection"
cd "Fake News Detection"
```

**2. Create and activate a virtual environment**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> TensorFlow installation may take 2–5 minutes. If you are on Apple Silicon (M1/M2/M3), TensorFlow 2.15+ supports Metal GPU acceleration automatically.

**4. Obtain the WELFake dataset**

Download from Kaggle: [saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

Place the CSV file at:
```
app/data/sample_news.csv
```

The CSV must have at minimum these columns: `text`, `label` (0 = Real, 1 = Fake).

**5. Set environment variables (optional but recommended for production)**
```bash
export SECRET_KEY="your-secure-random-string"
export FLASK_DEBUG=0
```

Or create a `.env` file in the project root:
```
SECRET_KEY=your-secure-random-string
FLASK_DEBUG=0
```

**6. Run the application**
```bash
python run.py
```

The first run will automatically:
- Create the SQLite database at `instance/fake_news.db`
- Train all models on the dataset (takes ~5–15 minutes depending on hardware)
- Save trained artifacts to `app/models_artifacts/`

Subsequent startups load all models from disk in seconds.

**7. Open in browser**
```
http://127.0.0.1:5001
```

Register an account, then navigate to the Dashboard to start analysing articles.

---

### Re-train models from scratch

```bash
source .venv/bin/activate
python scripts/train_models.py
```

This overwrites all artifacts in `app/models_artifacts/`.

---

## 15. Setup on Windows

### Prerequisites
- Windows 10 version 1903 or later (64-bit)
- Python 3.11 — download from [python.org](https://www.python.org/downloads/windows/)
  - During install, tick **"Add Python to PATH"** and **"pip"**
- Git — download from [git-scm.com](https://git-scm.com/downloads)
- (Recommended) Windows Terminal or PowerShell 7+

### Step-by-Step

**1. Open PowerShell as a standard user and clone the repository**
```powershell
git clone <repository-url> "Fake News Detection"
cd "Fake News Detection"
```

**2. Create and activate a virtual environment**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you receive an error `cannot be loaded because running scripts is disabled`, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then activate again.

**3. Upgrade pip and install dependencies**
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **TensorFlow on Windows:** TensorFlow 2.15+ requires the **Microsoft Visual C++ Redistributable** (usually already installed). If you see DLL-related errors, download and install it from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

**4. Obtain the WELFake dataset**

Download from Kaggle: [saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

Place the CSV at:
```
app\data\sample_news.csv
```

**5. Set environment variables (optional)**

In PowerShell for the current session:
```powershell
$env:SECRET_KEY = "your-secure-random-string"
$env:FLASK_DEBUG = "0"
```

Or create a `.env` file in the project root:
```
SECRET_KEY=your-secure-random-string
FLASK_DEBUG=0
```

**6. Run the application**
```powershell
python run.py
```

The first run trains all models automatically (5–15 minutes).

**7. Open in browser**
```
http://127.0.0.1:5001
```

---

### Re-train models from scratch (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\train_models.py
```

---

### Common Windows Issues

| Problem | Solution |
|---|---|
| `python` not found | Reinstall Python with "Add to PATH" ticked, or use `py -3.11` |
| `pip install` TensorFlow fails | Ensure you are on 64-bit Python; 32-bit is unsupported by TF |
| Port 5001 in use | Change port in `run.py`: `app.run(port=5002)` |
| Long path errors | Enable long paths: `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1` |
| NLTK data missing | Run `python -c "import nltk; nltk.download('wordnet')"` |

---

## 16. Training the Models

### Automatic (on first server start)
When `run.py` is executed and no model artifacts exist, `bootstrap.py` calls `train_models()` automatically. This is the recommended approach for new setups.

### Manual via training script
```bash
# macOS/Linux
source .venv/bin/activate
python scripts/train_models.py

# Windows
.\.venv\Scripts\Activate.ps1
python scripts\train_models.py
```

### What training does
1. Loads `app/data/sample_news.csv`
2. Drops unnamed columns and null rows
3. Applies stratified sampling (max 15,000 total, 7,500 per class)
4. Preprocesses all text via `preprocess_text()`
5. Splits 75/25 train/test (`random_state=42`, stratified)
6. Fits TF-IDF vectorizer on training set
7. Trains Logistic Regression, LinearSVC, MultinomialNB in sequence
8. Trains Bidirectional LSTM (up to 5 epochs with EarlyStopping)
9. Evaluates all models on the held-out test set
10. Computes F1-proportional ensemble weights
11. Saves all artifacts to `app/models_artifacts/`

Training artifacts produced:
```
app/models_artifacts/
  vectorizer.joblib       ← TF-IDF fitted transformer
  ml_models.joblib        ← dict of {name: fitted sklearn model}
  dl_tokenizer.joblib     ← Keras word-index tokenizer
  dl_lstm.keras           ← saved Keras model (SavedModel format)
  metadata.json           ← metrics, weights, timestamps
```

---

## 17. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key-change-me` | Flask session signing key — **change in production** |
| `DATABASE_URL` | `sqlite:///instance/fake_news.db` | SQLAlchemy DB URI |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode (never in production) |

---

## 18. Functional Requirements Coverage

| ID | Requirement | Status |
|---|---|---|
| FR1 | User registration with email/password | ✅ Flask-Login + Werkzeug hashing |
| FR2 | User login / logout | ✅ Session-based auth |
| FR3 | Authenticated-only access to analysis | ✅ `@login_required` on all routes |
| FR4 | Submit article as plain text | ✅ Textarea input |
| FR5 | Submit article via URL | ✅ BeautifulSoup scraper |
| FR6 | Text preprocessing pipeline | ✅ normalize → tokenize → filter → deduplicate |
| FR7 | TF-IDF feature extraction | ✅ 4,000 features, unigram+bigram |
| FR8 | Logistic Regression classification | ✅ scikit-learn, L2, lbfgs |
| FR9 | Linear SVM classification | ✅ LinearSVC + Platt scaling |
| FR10 | Bidirectional LSTM classification | ✅ TensorFlow/Keras |
| FR11 | Naive Bayes classification | ✅ MultinomialNB |
| FR12 | Hybrid ensemble prediction | ✅ F1-weighted blend + gap fallback |
| FR13 | Per-model transparency panel | ✅ All 4 models shown with probability bars |
| FR14 | Confidence score | ✅ `max(p, 1-p) × 100` |
| FR15 | Credibility score | ✅ `(1 - p) × 100` |
| FR16 | Influential keyword extraction | ✅ TF-IDF × model coefficient |
| FR17 | Explainable decision rationale | ✅ Strategy + gap + weight text |
| FR18 | Word frequency bar chart | ✅ Chart.js |
| FR19 | Model comparison bar chart | ✅ All models + hybrid |
| FR20 | Prediction distribution doughnut chart | ✅ Real vs Fake over history |
| FR21 | Analysis history per user | ✅ Submission model + `/api/history` |
| FR22 | PDF report export | ✅ ReportLab |
| FR23 | CSV report export | ✅ stdlib csv |
| FR24 | Model metrics displayed | ✅ Accuracy, Precision, Recall, F1 per model |
| FR25 | Responsive design | ✅ 3 breakpoints: 1024 / 768 / 480px |
| FR26 | Model hyperparameters & architecture UI | ✅ Full pipeline panel on dashboard |

---

## 19. Tech Stack Summary

| Layer | Technology | Version |
|---|---|---|
| Web framework | Flask | ≥ 3.0 |
| Auth | Flask-Login | ≥ 0.6 |
| ORM / DB | Flask-SQLAlchemy + SQLite | ≥ 3.1 |
| Form validation / CSRF | Flask-WTF | ≥ 1.2 |
| ML models | scikit-learn | ≥ 1.4 |
| Deep learning | TensorFlow / Keras | ≥ 2.15, < 2.19 |
| Numerical computing | NumPy, Pandas | ≥ 1.26, ≥ 2.1 |
| Model serialisation | Joblib | ≥ 1.3 |
| Web scraping | BeautifulSoup4 + lxml | ≥ 4.12 |
| PDF generation | ReportLab | ≥ 4.0 |
| Frontend charting | Chart.js | 4.4.7 (CDN) |
| Fonts | Google Fonts: Manrope + Playfair Display | CDN |
| Runtime | Python | 3.11 |
| Database | SQLite | (bundled) |

---

*Built as an end-to-end demonstration of hybrid NLP classification with full transparency, reproducibility, and production-ready authentication.*

This project is a full-stack Flask application for detecting fake news with a hybrid machine learning and deep learning pipeline. It supports secure authentication, article submission by text or URL, automated scraping, NLP preprocessing, explainable predictions, visual analytics, and report export.

## Features

- Secure registration, login, logout, and authenticated sessions
- Manual article submission and URL-based article extraction
- Consistent NLP preprocessing with normalization, stop-word removal, punctuation cleanup, and duplicate token reduction
- Hybrid prediction flow using TF-IDF + classical ML and BiLSTM deep learning
- Prediction confidence, credibility scoring, influential keyword explanations, and model comparison
- Visual dashboards for word frequencies and prediction history
- PDF and CSV export for generated reports
- SQLite persistence for users and analyzed articles

## Project Structure

- `app/`: Flask application package
- `app/services/`: scraping, preprocessing, training, reporting, and bootstrap services
- `app/data/sample_news.csv`: bundled labeled dataset for initial model training
- `app/models_artifacts/`: generated model files
- `reports/`: exported reports
- `scripts/`: utility scripts

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python run.py
   ```

4. Open the local Flask URL shown in the terminal.

## Notes

- On first launch, the application initializes the SQLite database and trains the bundled ML and DL models if artifacts are missing.
- Submitted articles are stored and can be reused for future retraining workflows.
