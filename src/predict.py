"""Load the trained model and run sentiment predictions."""
import argparse
from pathlib import Path
from typing import List, Optional

import joblib

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "sentiment_model.joblib"


def load_model(model_path: Path = DEFAULT_MODEL):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python src/train.py")
    return joblib.load(model_path)


def predict(texts, model: Optional[object] = None) -> List[dict]:
    """Return sentiment label and confidence for each input text."""
    model = model or load_model()
    if isinstance(texts, str):
        texts = [texts]
    labels = model.predict(texts)
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    return [
        {
            "text": text,
            "sentiment": label,
            "confidence": round(float(p[classes.index(label)]), 4),
        }
        for text, label, p in zip(texts, labels, probs)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for text")
    parser.add_argument("text", nargs="+", help="One or more sentences to classify")
    args = parser.parse_args()
    for r in predict(args.text):
        print(f"[{r['sentiment'].upper():8}] ({r['confidence']:.2%})  {r['text']}")
