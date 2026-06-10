"""    
main.py — Full pipeline for the Fake News Detection project.
Orchestrates data generation, preprocessing, training, evaluation, and reporting.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from data_generator import generate_dataset
from preprocessor import preprocess_dataframe, build_tfidf_features
from models import (
    train_and_evaluate_all, prepare_feature_matrix,
    cross_validate_models, get_best_model, save_model
)
from visualizer import (
    plot_class_distribution, plot_linguistic_features, plot_model_comparison,
    plot_confusion_matrices, plot_roc_curves, plot_wordclouds,
    plot_feature_importance, plot_confidence_distribution
)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LINGUISTIC_FEATURES = [
    "caps_ratio", "exclamation_ratio", "sensational_score",
    "credibility_score", "avg_word_length", "text_length", "question_ratio"
]

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    section("1. Generating Dataset")
    df = generate_dataset(n_real=600, n_fake=600)
    df.to_csv(f"{DATA_DIR}/news_dataset.csv", index=False)
    print(f"  Total: {len(df)} articles | Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")

    section("2. Preprocessing")
    df = preprocess_dataframe(df)
    print(f"  Processed {len(df)} articles with {len(df.columns)} features")

    section("3. Splitting Data")
    X_df = df
    y = df["label"].values
    train_df, test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    section("4. Building TF-IDF Features")
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        train_df["cleaned"], test_df["cleaned"],
        max_features=5000, ngram_range=(1, 2)
    )
    print(f"  TF-IDF shape: {X_train_tfidf.shape[1]} features")

    section("5. Combining Features")
    X_train, scaler = prepare_feature_matrix(X_train_tfidf, train_df, LINGUISTIC_FEATURES)
    X_test, _      = prepare_feature_matrix(X_test_tfidf,  test_df,  LINGUISTIC_FEATURES)
    # use the same scaler for test set
    from scipy.sparse import hstack, csr_matrix
    from sklearn.preprocessing import StandardScaler
    scaler2 = StandardScaler()
    scaler2.fit(train_df[LINGUISTIC_FEATURES].values)
    X_train = hstack([X_train_tfidf, csr_matrix(scaler2.transform(train_df[LINGUISTIC_FEATURES].values))])
    X_test  = hstack([X_test_tfidf,  csr_matrix(scaler2.transform(test_df[LINGUISTIC_FEATURES].values))])
    print(f"  Combined feature matrix: {X_train.shape}")

    section("6. Training & Evaluating Models")
    results, trained_models = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    section("7. Generating Visualizations")
    print("  Plotting...")

    plot_class_distribution(df, f"{REPORTS_DIR}/01_class_distribution.png")
    plot_linguistic_features(df, f"{REPORTS_DIR}/02_linguistic_features.png")
    plot_wordclouds(df, f"{REPORTS_DIR}/03_wordclouds.png")
    plot_model_comparison(results, f"{REPORTS_DIR}/04_model_comparison.png")
    plot_confusion_matrices(results, f"{REPORTS_DIR}/05_confusion_matrices.png")
    plot_roc_curves(results, f"{REPORTS_DIR}/06_roc_curves.png", y_test)

    best_name = get_best_model(results)
    best_model = trained_models[best_name]

    feature_names = vectorizer.get_feature_names_out().tolist() + LINGUISTIC_FEATURES
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_names,
                                f"{REPORTS_DIR}/07_feature_importance.png")

    plot_confidence_distribution(results, best_name, y_test,
                                f"{REPORTS_DIR}/08_confidence_distribution.png")

    section("8. Saving Best Model")
    save_model(best_model, vectorizer, scaler2, f"{MODELS_DIR}/{best_name.replace(' ', '_').lower()}")
    print(f"  Best model: {best_name}")

    section("9. Summary Report")
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        marker = "(Best)" if name == best_name else ""
        print(f"  {name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}{marker}")

    print(f"\n  Best Model: {best_name}")
    print(f"  F1 Score:   {results[best_name]['f1']:.4f}")
    print(f"  ROC AUC:    {results[best_name]['roc_auc']:.4f}")

    print(f"\n  Classification Report ({best_name}):")
    print(results[best_name]["classification_report"])

    print("\n  All outputs saved to:")
    print(f"    Reports: {REPORTS_DIR}/")
    print(f"    Models:  {MODELS_DIR}/")

    return results, trained_models, vectorizer, scaler2


if __name__ == "__main__":
    main()
