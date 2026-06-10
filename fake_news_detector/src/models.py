"""
Model training, evaluation, and comparison for fake news detection.
Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            random_state=42
        ),
    }


def prepare_feature_matrix(tfidf_matrix, linguistic_df, feature_cols):
    """Combine TF-IDF and linguistic features into one matrix."""
    scaler = StandardScaler()
    ling_feats = scaler.fit_transform(linguistic_df[feature_cols].values)
    ling_sparse = csr_matrix(ling_feats)
    X_combined = hstack([tfidf_matrix, ling_sparse])
    return X_combined, scaler


def evaluate_model(model, X_test, y_test):
    """Return comprehensive metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """Train all models and return results dict."""
    results = {}
    trained_models = {}
    models = get_models()

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    return results, trained_models


def cross_validate_models(X, y, cv=5):
    """Run stratified k-fold CV for all models."""
    models = get_models()
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
        cv_results[name] = {
            "mean_f1": scores.mean(),
            "std_f1": scores.std(),
            "scores": scores.tolist()
        }
    return cv_results


def get_best_model(results):
    """Return the name of the best model by F1 score."""
    return max(results, key=lambda k: results[k]["f1"])


def save_model(model, vectorizer, scaler, path_prefix):
    """Save model artifacts."""
    joblib.dump(model, f"{path_prefix}_model.pkl")
    joblib.dump(vectorizer, f"{path_prefix}_vectorizer.pkl")
    joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
    print(f"  Model saved to {path_prefix}_*.pkl")


def predict_article(title, text, model, vectorizer, scaler, ling_feature_cols):
    """Predict a single article."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessor import clean_text, tokenize_and_lemmatize, extract_linguistic_features

    combined = f"{title} {text}"
    cleaned = tokenize_and_lemmatize(clean_text(combined))
    tfidf_vec = vectorizer.transform([cleaned])

    ling_raw = extract_linguistic_features(combined)
    ling_arr = np.array([[ling_raw[col] for col in ling_feature_cols]])
    ling_scaled = csr_matrix(scaler.transform(ling_arr))

    X = hstack([tfidf_vec, ling_scaled])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return {
        "prediction": "FAKE" if pred == 1 else "REAL",
        "confidence": float(max(prob)),
        "fake_probability": float(prob[1]),
        "real_probability": float(prob[0]),
    }
