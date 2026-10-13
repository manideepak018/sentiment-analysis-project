import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

import joblib  # noqa: E402
from generate_sample_data import build_dataset  # noqa: E402
from predict import predict  # noqa: E402
from preprocess import clean_text  # noqa: E402
from train import train  # noqa: E402


def test_clean_text_lowercases_and_strips_urls():
    assert clean_text("Visit HTTPS://example.com NOW!") == "visit now"


def test_clean_text_removes_html_and_numbers():
    assert clean_text("<b>Great</b> 100% product") == "great product"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_model_end_to_end(tmp_path):
    data = tmp_path / "reviews.csv"
    build_dataset(n=300).to_csv(data, index=False)
    model_path = tmp_path / "model.joblib"
    metrics = train(data, model_path, tmp_path / "metrics.json")
    assert metrics["accuracy"] > 0.8

    model = joblib.load(model_path)
    results = predict(["I love this, it is excellent!", "Terrible, broke immediately."], model=model)
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"
