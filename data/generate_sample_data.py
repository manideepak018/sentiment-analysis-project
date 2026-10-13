"""
Generates a small synthetic product-review dataset for demonstration.
Replace data/reviews.csv with a real dataset (IMDB, Amazon reviews, Sentiment140, ...)
that has 'text' and 'label' columns for production use.
"""
import random
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 800

POSITIVE = [
    "I absolutely love this product, it works perfectly.",
    "Great quality and fast delivery, highly recommended.",
    "The service was excellent and the staff were very friendly.",
    "Best purchase I have made this year, five stars.",
    "Really happy with it, exceeded my expectations.",
    "Super easy to use and the battery lasts all day.",
    "Wonderful experience from start to finish.",
    "The design is beautiful and it feels very sturdy.",
    "Amazing value for the money, I would buy again.",
    "Customer support was helpful and resolved my issue quickly.",
    "The food was delicious and the portions were generous.",
    "Impressed by the performance, it is fast and smooth.",
]

NEGATIVE = [
    "Terrible product, it broke after two days.",
    "Very disappointed, the quality is poor and cheap.",
    "The service was awful and the staff were rude.",
    "Worst purchase ever, do not waste your money.",
    "It stopped working within a week, completely useless.",
    "The app keeps crashing and the interface is confusing.",
    "Shipping took forever and the package arrived damaged.",
    "The battery drains in just a few hours, very annoying.",
    "Overpriced for what you get, I regret buying it.",
    "Customer support ignored my emails for weeks.",
    "The food was cold and tasteless, never ordering again.",
    "Flimsy build and the instructions were impossible to follow.",
]


def build_dataset(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n // 2):
        pos = rng.sample(POSITIVE, k=rng.randint(1, 2))
        neg = rng.sample(NEGATIVE, k=rng.randint(1, 2))
        rows.append({"text": " ".join(pos), "label": "positive"})
        rows.append({"text": " ".join(neg), "label": "negative"})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "reviews.csv"
    df = build_dataset()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["label"].value_counts())
