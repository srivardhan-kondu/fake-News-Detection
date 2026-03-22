from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .preprocessing import get_word_frequency, preprocess_text


class HybridFakeNewsService:
    def __init__(self, app=None):
        self.app = None
        self.artifact_dir = None
        self.dataset_path = None
        # In-memory caches — loaded once, reused across requests
        self._vectorizer = None
        self._ml_models = None
        self._dl_tokenizer = None
        self._dl_model = None
        self._metadata = None
        if app is not None:
            self.init_app(app)

    @classmethod
    def from_app(cls, app) -> "HybridFakeNewsService":
        detector = app.extensions.get("hybrid_detector")
        if detector is None:
            detector = cls(app)
            app.extensions["hybrid_detector"] = detector
        return detector

    def init_app(self, app) -> None:
        self.app = app
        self.artifact_dir = Path(app.config["MODEL_ARTIFACTS_DIR"])
        self.dataset_path = Path(app.config["DATASET_PATH"])

    def ensure_model_artifacts(self) -> None:
        required_files = [
            self.artifact_dir / "vectorizer.joblib",
            self.artifact_dir / "ml_models.joblib",
            self.artifact_dir / "dl_tokenizer.joblib",
            self.artifact_dir / "dl_lstm.keras",
            self.artifact_dir / "metadata.json",
        ]
        if all(path.exists() for path in required_files):
            return
        self.train_models()

    def train_models(self) -> None:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer

        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        MAX_TRAINING_SAMPLES = 15000

        dataset = pd.read_csv(self.dataset_path)
        # Drop unnamed index columns from WELFake-style CSVs
        dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]
        dataset = dataset.dropna(subset=["text", "label"])
        dataset["label"] = dataset["label"].astype(int)

        # Stratified sample to keep training time practical
        if len(dataset) > MAX_TRAINING_SAMPLES:
            dataset = (
                dataset.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), MAX_TRAINING_SAMPLES // 2), random_state=42))
                .reset_index(drop=True)
            )

        dataset["processed_text"] = dataset["text"].astype(str).apply(preprocess_text)

        x_train, x_test, y_train, y_test = train_test_split(
            dataset["processed_text"],
            dataset["label"].astype(int),
            test_size=0.25,
            random_state=42,
            stratify=dataset["label"].astype(int),
        )

        vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        logistic_model = LogisticRegression(max_iter=2000)
        logistic_model.fit(x_train_tfidf, y_train)

        svm_model = LinearSVC()
        svm_model.fit(x_train_tfidf, y_train)

        nb_model = MultinomialNB()
        nb_model.fit(x_train_tfidf, y_train)

        ml_models = {
            "logistic_regression": logistic_model,
            "linear_svm": svm_model,
            "naive_bayes": nb_model,
        }

        ml_results = {}
        for name, model in ml_models.items():
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x_test_tfidf)[:, 1]
            else:
                margins = model.decision_function(x_test_tfidf)
                probabilities = 1 / (1 + np.exp(-margins))
            predictions = (probabilities >= 0.5).astype(int)
            ml_results[name] = self._compute_metrics(y_test, predictions)

        best_ml_name = max(ml_results, key=lambda name: ml_results[name]["f1_score"])

        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(x_train.tolist())
        x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train.tolist()), maxlen=150, padding="post")
        x_test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test.tolist()), maxlen=150, padding="post")

        dl_model = Sequential(
            [
                Embedding(input_dim=5000, output_dim=64, input_length=150),
                Bidirectional(LSTM(32, return_sequences=False)),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        dl_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        dl_model.fit(
            x_train_seq,
            y_train,
            epochs=5,
            batch_size=8,
            validation_split=0.2,
            verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        )
        dl_probabilities = dl_model.predict(x_test_seq, verbose=0).flatten()
        dl_predictions = (dl_probabilities >= 0.5).astype(int)
        dl_results = self._compute_metrics(y_test, dl_predictions)

        # Compute per-model F1 weights for full 4-model ensemble
        all_f1 = {name: ml_results[name]["f1_score"] for name in ml_results}
        all_f1["bilstm"] = dl_results["f1_score"]
        total_f1 = max(sum(all_f1.values()), 0.001)
        per_model_weights = {name: f1 / total_f1 for name, f1 in all_f1.items()}

        ml_f1 = ml_results[best_ml_name]["f1_score"]
        dl_f1 = dl_results["f1_score"]
        total_weight = max(ml_f1 + dl_f1, 0.001)
        metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "dataset_size": int(len(dataset)),
            "labels": {"real": 0, "fake": 1},
            "best_ml_model": best_ml_name,
            "ml_models": ml_results,
            "dl_model": "bilstm",
            "performance": {
                "ml": ml_results[best_ml_name],
                "dl": dl_results,
            },
            "ensemble_weights": {
                "ml": ml_f1 / total_weight,
                "dl": dl_f1 / total_weight,
            },
            "per_model_weights": per_model_weights,
        }

        joblib.dump(vectorizer, self.artifact_dir / "vectorizer.joblib")
        joblib.dump(ml_models, self.artifact_dir / "ml_models.joblib")
        joblib.dump(tokenizer, self.artifact_dir / "dl_tokenizer.joblib")
        dl_model.save(self.artifact_dir / "dl_lstm.keras")
        (self.artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _load_artifacts(self) -> None:
        """Load all model artifacts into memory once."""
        if self._metadata is not None:
            return
        from tensorflow.keras.models import load_model as keras_load_model

        self._vectorizer = joblib.load(self.artifact_dir / "vectorizer.joblib")
        self._ml_models = joblib.load(self.artifact_dir / "ml_models.joblib")
        self._dl_tokenizer = joblib.load(self.artifact_dir / "dl_tokenizer.joblib")
        self._dl_model = keras_load_model(self.artifact_dir / "dl_lstm.keras")
        self._metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )

    def get_metrics(self) -> dict:
        self.ensure_model_artifacts()
        self._load_artifacts()
        meta = dict(self._metadata)

        # Enrich with model architecture details and hyperparameters
        model_details = []

        # --- ML models ---
        ml_param_map = {
            "logistic_regression": {
                "display_name": "Logistic Regression",
                "type": "ML — Linear Classifier",
                "description": "Probabilistic linear model that learns a decision boundary using log-odds. Outputs calibrated probabilities via sigmoid.",
                "params": {"max_iter": 2000, "solver": "lbfgs", "penalty": "l2"},
            },
            "linear_svm": {
                "display_name": "Linear SVM",
                "type": "ML — Support Vector Machine",
                "description": "Finds the maximum-margin hyperplane separating fake and real classes. Probabilities estimated via Platt scaling on decision_function.",
                "params": {"loss": "squared_hinge", "penalty": "l2", "max_iter": 1000},
            },
            "naive_bayes": {
                "display_name": "Multinomial Naive Bayes",
                "type": "ML — Probabilistic Classifier",
                "description": "Applies Bayes' theorem assuming feature independence. Naturally suited for word-count / TF-IDF features.",
                "params": {"alpha": 1.0, "fit_prior": True},
            },
        }

        for name, metrics in meta.get("ml_models", {}).items():
            info = ml_param_map.get(name, {})
            model_details.append({
                "key": name,
                "display_name": info.get("display_name", name),
                "model_type": info.get("type", "ML"),
                "description": info.get("description", ""),
                "hyperparameters": info.get("params", {}),
                "is_best_ml": name == meta.get("best_ml_model"),
                "metrics": metrics,
            })

        # --- DL model ---
        model_details.append({
            "key": "bilstm",
            "display_name": "Bidirectional LSTM",
            "model_type": "DL — Deep Learning",
            "description": "Sequential neural network: Embedding → Bidirectional LSTM → Dense layers. Captures long-range contextual dependencies in text.",
            "hyperparameters": {
                "embedding_dim": 64,
                "vocab_size": 5000,
                "max_sequence_length": 150,
                "lstm_units": 32,
                "dense_units": 16,
                "activation": "sigmoid",
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "epochs": "5 (EarlyStopping patience=2)",
                "batch_size": 8,
            },
            "is_best_ml": False,
            "metrics": meta.get("performance", {}).get("dl", {}),
        })

        # --- Preprocessing / Vectorizer config ---
        preprocessing_config = {
            "vectorizer": "TfidfVectorizer",
            "max_features": 4000,
            "ngram_range": "unigrams + bigrams (1, 2)",
            "steps": [
                "Lowercase & strip HTML",
                "Remove URLs, emails, digits",
                "Tokenize (word-level)",
                "Remove English stopwords",
                "Lemmatize (WordNet)",
                "Remove duplicate tokens",
            ],
        }

        # --- Ensemble decision factors ---
        weights = meta.get("ensemble_weights", {})
        per_model_w = meta.get("per_model_weights", {})
        decision_factors = {
            "method": "F1-weighted ensemble blending (all 4 models)",
            "ml_weight": round(weights.get("ml", 0), 4),
            "dl_weight": round(weights.get("dl", 0), 4),
            "per_model_weights": {k: round(v, 4) for k, v in per_model_w.items()},
            "fallback_rule": "If the max probability gap among all 4 models > 35%, the system selects the single best-performing model (by F1) instead of blending.",
            "best_model_selection": "The ML model with the highest F1 score on the held-out test set is designated best_ml_model.",
        }

        meta["model_details"] = model_details
        meta["preprocessing_config"] = preprocessing_config
        meta["decision_factors"] = decision_factors
        return meta

    def analyze(self, title: str, raw_text: str) -> dict:
        self.ensure_model_artifacts()
        processed_text = preprocess_text(raw_text)
        if not processed_text:
            raise ValueError("The submitted article does not contain enough analyzable text.")

        all_ml_results = self._predict_all_ml(processed_text)
        dl_result = self._predict_dl(processed_text)
        metadata = self.get_metrics()
        weights = metadata["ensemble_weights"]

        best_ml_name = metadata["best_ml_model"]
        ml_result = all_ml_results[best_ml_name]

        # --- Full 4-model weighted ensemble ---
        per_model_weights = metadata.get("per_model_weights", {})
        if per_model_weights:
            weighted_sum = 0.0
            for name, res in all_ml_results.items():
                w = per_model_weights.get(name, 0)
                weighted_sum += w * res["probability"]
            weighted_sum += per_model_weights.get("bilstm", 0) * dl_result["probability"]
            hybrid_probability = weighted_sum
        else:
            hybrid_probability = (weights["ml"] * ml_result["probability"]) + (weights["dl"] * dl_result["probability"])

        # Compute min/max individual probability to detect large disagreement
        all_probs = [res["probability"] for res in all_ml_results.values()] + [dl_result["probability"]]
        probability_gap = max(all_probs) - min(all_probs)
        use_best_model = probability_gap > 0.35

        if use_best_model:
            if metadata["performance"]["ml"]["f1_score"] >= metadata["performance"]["dl"]["f1_score"]:
                final_probability = ml_result["probability"]
                selected_strategy = f"best performing model: {best_ml_name}"
                decision_reason = (f"Models disagreed significantly (gap {probability_gap*100:.1f}%). "
                                   f"The best ML model ({best_ml_name}) was selected based on higher F1 score.")
            else:
                final_probability = dl_result["probability"]
                selected_strategy = "best performing model: bilstm"
                decision_reason = (f"Models disagreed significantly (gap {probability_gap*100:.1f}%). "
                                   f"The BiLSTM deep learning model was selected based on higher F1 score.")
        else:
            final_probability = hybrid_probability
            selected_strategy = "weighted ensemble (all 4 models)"
            weight_parts = [f"{n.replace('_', ' ').title()} {per_model_weights.get(n, 0):.2f}" for n in all_ml_results]
            weight_parts.append(f"BiLSTM {per_model_weights.get('bilstm', 0):.2f}")
            decision_reason = (f"All models agreed closely (gap {probability_gap*100:.1f}%). "
                               f"A weighted ensemble of all 4 models was used: {', '.join(weight_parts)}.")

        predicted_label = "Fake News" if final_probability >= 0.5 else "Real News"
        confidence_score = round(max(final_probability, 1 - final_probability) * 100, 2)
        credibility_score = round((1 - final_probability) * 100, 2)

        # --- Explainability: split terms into fake-supporting vs real-supporting ---
        fake_supporting_terms = ml_result["fake_supporting_terms"]
        real_supporting_terms = ml_result["real_supporting_terms"]

        # Build per-model individual predictions for transparency
        individual_predictions = []
        for name, res in all_ml_results.items():
            prob = res["probability"]
            label = "Fake News" if prob >= 0.5 else "Real News"
            trained_metrics = metadata["ml_models"].get(name, {})
            individual_predictions.append({
                "model_name": name.replace("_", " ").title(),
                "model_type": "ML",
                "probability_fake": round(prob * 100, 2),
                "prediction": label,
                "is_best_ml": name == best_ml_name,
                "f1_score": trained_metrics.get("f1_score", 0),
                "accuracy": trained_metrics.get("accuracy", 0),
                "ensemble_weight": round(per_model_weights.get(name, 0), 4),
            })

        dl_prob = dl_result["probability"]
        dl_label = "Fake News" if dl_prob >= 0.5 else "Real News"
        individual_predictions.append({
            "model_name": "BiLSTM (Deep Learning)",
            "model_type": "DL",
            "probability_fake": round(dl_prob * 100, 2),
            "prediction": dl_label,
            "is_best_ml": False,
            "f1_score": metadata["performance"]["dl"].get("f1_score", 0),
            "accuracy": metadata["performance"]["dl"].get("accuracy", 0),
            "ensemble_weight": round(per_model_weights.get("bilstm", 0), 4),
        })

        insights = [
            f"Final classification used the {selected_strategy} strategy.",
            decision_reason,
        ]
        for name, res in all_ml_results.items():
            w = per_model_weights.get(name, 0)
            insights.append(f"{name.replace('_', ' ').title()} (weight {w:.2f}) estimated {res['probability']*100:.2f}% fake probability.")
        insights.extend([
            f"BiLSTM (weight {per_model_weights.get('bilstm', 0):.2f}) estimated {dl_prob*100:.2f}% fake probability.",
            f"The weighted ensemble probability was {hybrid_probability*100:.2f}%.",
            "Influential keywords are derived from TF-IDF × model coefficient contributions.",
            f"Top words pushing toward Fake: {', '.join(fake_supporting_terms) if fake_supporting_terms else 'none identified'}.",
            f"Top words pushing toward Real: {', '.join(real_supporting_terms) if real_supporting_terms else 'none identified'}.",
        ])

        return {
            "title": title,
            "processed_text": processed_text,
            "predicted_label": predicted_label,
            "confidence_score": confidence_score,
            "credibility_score": credibility_score,
            "explanation": {
                "influential_terms": fake_supporting_terms + real_supporting_terms,
                "fake_supporting_terms": fake_supporting_terms,
                "real_supporting_terms": real_supporting_terms,
                "insights": insights,
            },
            "charts": {
                "word_frequency": get_word_frequency(processed_text),
            },
            "model_breakdown": {
                "selected_strategy": selected_strategy,
                "decision_reason": decision_reason,
                "ensemble_weights": weights,
                "per_model_weights": per_model_weights,
                "ml_model": ml_result["model_name"],
                "ml_probability_fake": round(ml_result["probability"] * 100, 2),
                "dl_probability_fake": round(dl_prob * 100, 2),
                "hybrid_probability_fake": round(hybrid_probability * 100, 2),
                "ml_metrics": metadata["performance"]["ml"],
                "dl_metrics": metadata["performance"]["dl"],
                "individual_predictions": individual_predictions,
            },
        }

    def _predict_all_ml(self, processed_text: str) -> dict:
        """Run every ML model and return per-model results with split fake/real terms."""
        self._load_artifacts()
        vectorizer = self._vectorizer
        models = self._ml_models
        features = vectorizer.transform([processed_text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        non_zero_indices = features.nonzero()[1]

        results = {}
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(features)[0][1])
            else:
                margin = float(model.decision_function(features)[0])
                probability = 1 / (1 + math.exp(-margin))

            fake_supporting_terms = []
            real_supporting_terms = []
            if hasattr(model, "coef_"):
                contribution_values = []
                coef_weights = model.coef_[0]
                feature_values = features.toarray()[0]
                for index in non_zero_indices:
                    contribution_values.append((feature_names[index], feature_values[index] * coef_weights[index]))
                # Positive contributions push toward fake (label 1)
                sorted_by_score = sorted(contribution_values, key=lambda item: item[1], reverse=True)
                fake_supporting_terms = [term for term, score in sorted_by_score if score > 0][:6]
                # Negative contributions push toward real (label 0)
                sorted_by_score_asc = sorted(contribution_values, key=lambda item: item[1])
                real_supporting_terms = [term for term, score in sorted_by_score_asc if score < 0][:6]

            results[name] = {
                "model_name": name,
                "probability": probability,
                "fake_supporting_terms": fake_supporting_terms,
                "real_supporting_terms": real_supporting_terms,
                "influential_terms": fake_supporting_terms + real_supporting_terms,
            }
        return results

    def _predict_ml(self, processed_text: str) -> dict:
        self._load_artifacts()
        vectorizer = self._vectorizer
        models = self._ml_models
        best_ml_name = self._metadata["best_ml_model"]
        model = models[best_ml_name]
        features = vectorizer.transform([processed_text])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])
        else:
            margin = float(model.decision_function(features)[0])
            probability = 1 / (1 + math.exp(-margin))

        feature_names = np.array(vectorizer.get_feature_names_out())
        non_zero_indices = features.nonzero()[1]
        fake_supporting_terms = []
        real_supporting_terms = []
        if hasattr(model, "coef_"):
            contribution_values = []
            coef_weights = model.coef_[0]
            feature_values = features.toarray()[0]
            for index in non_zero_indices:
                contribution_values.append((feature_names[index], feature_values[index] * coef_weights[index]))
            sorted_desc = sorted(contribution_values, key=lambda item: item[1], reverse=True)
            fake_supporting_terms = [term for term, score in sorted_desc if score > 0][:6]
            sorted_asc = sorted(contribution_values, key=lambda item: item[1])
            real_supporting_terms = [term for term, score in sorted_asc if score < 0][:6]

        return {
            "model_name": best_ml_name,
            "probability": probability,
            "fake_supporting_terms": fake_supporting_terms,
            "real_supporting_terms": real_supporting_terms,
            "influential_terms": fake_supporting_terms + real_supporting_terms,
        }

    def _predict_dl(self, processed_text: str) -> dict:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self._load_artifacts()
        sequence = self._dl_tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=150, padding="post")
        probability = float(self._dl_model.predict(padded_sequence, verbose=0)[0][0])
        return {"probability": probability}

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }
