"""Train a TF-IDF + Logistic Regression sentiment classifier."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent


def train(data_path: Path, model_path: Path, metrics_path: Path, seed: int = 42) -> dict:
    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["positive", "negative"])]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=seed, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "report": classification_report(y_test, preds, output_dict=True),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--data", default=str(ROOT / "data" / "reviews.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" / "sentiment_model.joblib"))
    parser.add_argument("--metrics", default=str(ROOT / "models" / "metrics.json"))
    args = parser.parse_args()

    m = train(Path(args.data), Path(args.model), Path(args.metrics))
    print(f"Accuracy: {m['accuracy']}  |  Macro F1: {m['f1_macro']}")
    print(f"Model saved to {args.model}")
