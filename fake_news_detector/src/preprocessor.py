"""
Text preprocessing pipeline for fake news detection.
Handles cleaning, tokenization, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Linguistic red flags common in fake news
SENSATIONAL_WORDS = [
    "shocking", "bombshell", "exposed", "breaking", "urgent", "must", "share",
    "truth", "wake", "deep", "state", "agenda", "cover", "silenced", "secret",
    "refuses", "mainstream", "elites", "patriots", "deleted", "hidden"
]

CREDIBILITY_WORDS = [
    "according", "study", "research", "published", "confirmed", "data",
    "evidence", "analysis", "report", "findings", "peer", "reviewed",
    "official", "university", "journal", "researchers", "scientists"
]

FEATURE_NAMES = [
    "caps_ratio",
    "exclamation_ratio",
    "sensational_score",
    "credibility_score",
    "avg_word_length",
    "text_length",
    "question_ratio"
] 

FEATURE_NAMES = [
    "caps_ratio",
    "exclamation_ratio",
    "sensational_score",
    "credibility_score",
    "avg_word_length",
    "text_length",
    "question_ratio"
]
def preprocess(df):
    return preprocess_dataframe(df)

def clean_text(text):
    """Clean and normalize text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)              # Remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', ' ', text)              # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()           # Collapse whitespace
    return text


def tokenize_and_lemmatize(text):
    """Tokenize, remove stopwords, and lemmatize."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def extract_linguistic_features(texts):
    """Extract linguistic features for a list of texts."""
    features = []

    for text in texts:
        raw = str(text)
        words = raw.split()

        clean = clean_text(raw)
        clean_words = clean.split()

        caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        caps_ratio = caps_count / (len(words) + 1e-9)

        excl_count = raw.count('!')
        excl_ratio = excl_count / (len(words) + 1e-9)

        sensational_score = sum(
            1 for w in clean_words if w in SENSATIONAL_WORDS
        ) / (len(clean_words) + 1e-9)

        credibility_score = sum(
            1 for w in clean_words if w in CREDIBILITY_WORDS
        ) / (len(clean_words) + 1e-9)

        avg_word_len = (
            np.mean([len(w) for w in clean_words])
            if clean_words else 0
        )

        text_length = len(clean_words)

        question_ratio = raw.count('?') / (len(words) + 1e-9)

        features.append([
            caps_ratio,
            excl_ratio,
            sensational_score,
            credibility_score,
            avg_word_len,
            text_length,
            question_ratio
        ])

    return np.array(features)

def preprocess_dataframe(df):
    """Full preprocessing pipeline on a DataFrame."""
    df = df.copy()

    # Combine title + text for richer signal
    df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Clean and lemmatize
    df["cleaned"] = df["combined"].apply(clean_text).apply(tokenize_and_lemmatize)

    # Linguistic features
    feature_matrix = extract_linguistic_features(df["combined"].tolist())

    feat_df = pd.DataFrame(
        feature_matrix,
        columns=FEATURE_NAMES,
        index=df.index
    )

    df = pd.concat([df, feat_df], axis=1)

    return df


def build_tfidf_features(train_texts, test_texts=None, max_features=5000, ngram_range=(1, 2)):
    """Fit TF-IDF on train, transform both sets."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode'
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts) if test_texts is not None else None
    return X_train, X_test, vectorizer


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_generator import generate_dataset

    df = generate_dataset(100, 100)
    processed = preprocess_dataframe(df)
    print("Sample processed row:")
    print(processed[["cleaned", "caps_ratio", "sensational_score", "credibility_score"]].head(3))
    print(f"\nFeature columns: {list(processed.columns)}")
