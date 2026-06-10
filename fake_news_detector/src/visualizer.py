"""
Visualization module — generates all plots for the fake news detection report.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")

# --- Design palette ---
PALETTE = {
    "real":   "#2ECC71",   # Green
    "fake":   "#E74C3C",   # Red
    "accent": "#3498DB",   # Blue
    "dark":   "#2C3E50",
    "light":  "#ECF0F1",
    "mid":    "#7F8C8D",
    "bg":     "#FAFBFC",
}
MODELS_COLORS = ["#3498DB", "#E67E22", "#9B59B6"]

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.color":       "#CCCCCC",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})


def plot_class_distribution(df, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["label_name"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                color=[PALETTE["real"], PALETTE["fake"]],
                edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"{val}\n({val/len(df)*100:.1f}%)", ha='center', va='bottom',
                fontweight='bold', color=PALETTE["dark"])
    ax.set_title("Dataset Class Distribution", fontweight='bold', pad=15)
    ax.set_xlabel("Article Class")
    ax.set_ylabel("Number of Articles")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.tick_params(bottom=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_linguistic_features(df, save_path):
    features = ["caps_ratio", "exclamation_ratio", "sensational_score",
                "credibility_score", "avg_word_length", "text_length"]
    labels = ["CAPS Ratio", "Exclamation\nRatio", "Sensational\nScore",
            "Credibility\nScore", "Avg Word\nLength", "Text Length\n(words)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    for i, (feat, label) in enumerate(zip(features, labels)):
        ax = axes[i]
        ax.hist(real_df[feat], bins=25, alpha=0.65, color=PALETTE["real"],
                label="Real", edgecolor="white", linewidth=0.5)
        ax.hist(fake_df[feat], bins=25, alpha=0.65, color=PALETTE["fake"],
                label="Fake", edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel("Count")
        if i == 0:
            ax.legend(frameon=True, fancybox=True)

    fig.suptitle("Linguistic Feature Distributions: Real vs Fake News",
                fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, color) in enumerate(zip(model_names, MODELS_COLORS)):
        vals = [results[name][m] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color,
                    alpha=0.85, edgecolor="white", linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8,
                    color=PALETTE["dark"])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["REAL", "FAKE"],
                    yticklabels=["REAL", "FAKE"],
                    linewidths=0.5, cbar=False,
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"{name}\n(Confusion Matrix)", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(results, save_path, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, color) in zip(results.keys(), MODELS_COLORS):
        y_prob = results[name]["y_prob"]
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = results[name]["roc_auc"]
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Fake News Detection", fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_wordclouds(df, save_path):
    def make_wc(texts, color):
        text = " ".join(texts)
        return WordCloud(
            width=600, height=350, background_color='white',
            colormap='Greens' if color == "real" else 'Reds',
            max_words=80, collocations=False,
            prefer_horizontal=0.7
        ).generate(text)

    real_texts = df[df["label"] == 0]["cleaned"].tolist()
    fake_texts = df[df["label"] == 1]["cleaned"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, texts, color, title in zip(
        axes, [real_texts, fake_texts], ["real", "fake"],
        ["Real News — Common Terms", "Fake News — Common Terms"]
    ):
        wc = make_wc(texts, color)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold',
                    color=PALETTE[color], pad=12)

    plt.suptitle("Word Cloud Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE["fake"] if i < top_n // 3 else PALETTE["accent"] for i in range(top_n)]
    ax.barh(range(top_n), top_importances[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Most Important Features\n(Random Forest)", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confidence_distribution(results, best_model_name, y_test, save_path):
    probs = results[best_model_name]["y_prob"]
    if probs is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    real_probs = probs[y_test == 0]
    fake_probs = probs[y_test == 1]

    ax.hist(real_probs, bins=30, alpha=0.65, color=PALETTE["real"],
            label=f"Real articles (n={len(real_probs)})", edgecolor='white')
    ax.hist(fake_probs, bins=30, alpha=0.65, color=PALETTE["fake"],
            label=f"Fake articles (n={len(fake_probs)})", edgecolor='white')
    ax.axvline(0.5, color=PALETTE["dark"], linestyle='--', lw=2, label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted Probability of being FAKE")
    ax.set_ylabel("Number of Articles")
    ax.set_title(f"Prediction Confidence Distribution\n{best_model_name}", fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
