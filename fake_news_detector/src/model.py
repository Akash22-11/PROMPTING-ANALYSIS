"""
Model Training & Evaluation
Trains multiple classifiers on TF-IDF + linguistic features,
evaluates them and persists the best model.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes       import MultinomialNB
from sklearn.svm               import LinearSVC
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score,
)
from sklearn.preprocessing    import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import preprocess, extract_linguistic_features, FEATURE_NAMES


# ─── Feature Engineering ──────────────────────────────────────────────────

def build_features(texts_raw: list[str], tfidf: TfidfVectorizer = None,
                scaler: MinMaxScaler = None, fit: bool = True):
    """
    Returns:
    X_combined  – sparse matrix: TF-IDF + scaled linguistic features
    tfidf       – fitted (or passed) vectoriser
    scaler      – fitted (or passed) scaler
    """
    texts_clean = [preprocess(t) for t in texts_raw]

    if fit:
        tfidf  = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                                sublinear_tf=True, min_df=2)
        X_tfidf = tfidf.fit_transform(texts_clean)
    else:
        X_tfidf = tfidf.transform(texts_clean)

    X_ling = extract_linguistic_features(texts_raw)

    if fit:
        scaler  = MinMaxScaler()
        X_ling  = scaler.fit_transform(X_ling)
    else:
        X_ling  = scaler.transform(X_ling)

    X_combined = hstack([X_tfidf, csr_matrix(X_ling)])
    return X_combined, tfidf, scaler


# ─── Classifiers ──────────────────────────────────────────────────────────

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Linear SVM"         : LinearSVC(max_iter=1000, C=1.0, random_state=42),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Naive Bayes"        : MultinomialNB(),          # TF-IDF values are non-negative
}


# ─── Training Pipeline ────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, model_dir: str = "models") -> dict:
    os.makedirs(model_dir, exist_ok=True)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("⚙  Building features …")
    X_train, tfidf, scaler = build_features(X_train_raw, fit=True)
    X_test,  _,     _      = build_features(X_test_raw, tfidf=tfidf,
                                            scaler=scaler, fit=False)

    results = {}
    best_name, best_f1, best_model = None, -1, None

    print(f"\n{'Model':<25}  {'Accuracy':>8}  {'F1':>8}  {'ROC-AUC':>8}")
    print("─" * 58)

    for name, clf in CLASSIFIERS.items():
        # Naive Bayes needs a fully dense non-negative matrix — use only TF-IDF part
        if isinstance(clf, MultinomialNB):
            tfidf_cols = X_train.shape[1] - len(FEATURE_NAMES)
            X_tr = X_train[:, :tfidf_cols]
            X_te = X_test[:, :tfidf_cols]
        else:
            X_tr, X_te = X_train, X_test

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        # ROC-AUC: use decision_function / predict_proba where available
        try:
            scores = clf.decision_function(X_te)
        except AttributeError:
            scores = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, scores)

        results[name] = {
            "accuracy": acc, "f1": f1, "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred,
                                target_names=["Real", "Fake"], output_dict=True),
        }

        print(f"{name:<25}  {acc:>8.4f}  {f1:>8.4f}  {auc:>8.4f}")

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, clf

    print(f"\n✅ Best model: {best_name}  (F1 = {best_f1:.4f})")

    # ── Persist best model + transformers ─────────────────────────────────
    joblib.dump(best_model, f"{model_dir}/best_model.pkl")
    joblib.dump(tfidf,      f"{model_dir}/tfidf.pkl")
    joblib.dump(scaler,     f"{model_dir}/scaler.pkl")

    metadata = {
        "best_model_name": best_name,
        "best_f1": best_f1,
        "feature_dim": X_train.shape[1],
        "train_size": len(X_train_raw),
        "test_size":  len(X_test_raw),
    }
    joblib.dump(metadata, f"{model_dir}/metadata.pkl")

    return {
        "results"         : results,
        "best_model_name" : best_name,
        "tfidf"           : tfidf,
        "scaler"          : scaler,
        "best_model"      : best_model,
        "X_test_raw"      : X_test_raw,
        "y_test"          : y_test,
    }


# ─── Inference helper ─────────────────────────────────────────────────────

def predict_article(text: str, model_dir: str = "models") -> dict:
    """
    Loads the saved model and returns a prediction dict:
    {"label": "FAKE"|"REAL", "confidence": float, "risk_factors": [...]}
    """
    model  = joblib.load(f"{model_dir}/best_model.pkl")
    tfidf  = joblib.load(f"{model_dir}/tfidf.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")

    X, _, _ = build_features([text], tfidf=tfidf, scaler=scaler, fit=False)

    is_nb = isinstance(model, MultinomialNB)
    if is_nb:
        tfidf_cols = X.shape[1] - len(FEATURE_NAMES)
        X_inp = X[:, :tfidf_cols]
    else:
        X_inp = X

    pred = model.predict(X_inp)[0]

    try:
        prob = model.predict_proba(X_inp)[0]
        conf = float(max(prob))
    except AttributeError:
        score = float(model.decision_function(X_inp)[0])
        conf  = 1 / (1 + np.exp(-abs(score)))

    # Highlight linguistic red flags
    from preprocessor import (count_exclamation_marks, count_caps_words,
                            sensational_score, credibility_score)
    risk_factors = []
    if count_exclamation_marks(text) >= 2:
        risk_factors.append("Multiple exclamation marks detected")
    if count_caps_words(text) >= 3:
        risk_factors.append("Excessive ALL-CAPS words")
    if sensational_score(text) >= 2:
        risk_factors.append("Sensational / alarmist language")
    if credibility_score(text) == 0 and pred == 1:
        risk_factors.append("No credible sourcing markers")

    return {
        "label"       : "FAKE" if pred == 1 else "REAL",
        "confidence"  : round(conf * 100, 1),
        "risk_factors": risk_factors,
    }
