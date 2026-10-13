# Sentiment Analysis Project

An end-to-end sentiment analysis project: data preparation, text preprocessing, model training,
evaluation, a CLI predictor, a Streamlit web app, unit tests, and Docker deployment.

## Project structure

```
sentiment-analysis-project/
├── app.py                       # Streamlit web app (single text, batch CSV, model info)
├── Dockerfile                   # Container for deployment
├── requirements.txt
├── data/
│   └── generate_sample_data.py  # Creates demo data (replace with real data)
├── models/                      # Trained model + metrics (created by train.py)
├── src/
│   ├── preprocess.py            # Text cleaning
│   ├── train.py                 # Training and evaluation
│   └── predict.py               # Inference (CLI and importable)
└── tests/
    └── test_pipeline.py         # Pytest suite
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare data

Generate the demo dataset:
```bash
python data/generate_sample_data.py
```

For real work, place a CSV at `data/reviews.csv` with two columns:

| column | description                 |
|--------|-----------------------------|
| text   | review, tweet, or comment   |
| label  | `positive` or `negative`    |

Public datasets to try: IMDB movie reviews, Amazon product reviews, Sentiment140 (tweets).

## 3. Train

```bash
python src/train.py
```

This writes `models/sentiment_model.joblib` and `models/metrics.json`.

## 4. Predict

Command line:
```bash
python src/predict.py "The movie was fantastic!" "Worst service I have ever had."
```

Python:
```python
import sys; sys.path.insert(0, "src")
from predict import predict
predict(["I love it"])
# [{'text': 'I love it', 'sentiment': 'positive', 'confidence': 0.87}]
```

## 5. Run the web app

```bash
streamlit run app.py
```

Open http://localhost:8501.

## 6. Test

```bash
pytest -q
```

## 7. Deploy

**Streamlit Community Cloud (free):** push this folder to GitHub, open https://share.streamlit.io,
choose **New app**, and set the main file to `app.py`. Remove `models/*.joblib` from `.gitignore`
and commit the trained model, or add a step that runs `python src/train.py` first.

**Docker:**
```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

**Render / Railway / Hugging Face Spaces:** use the Dockerfile, or set the start command to
`streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.

## Method

1. **Cleaning:** lowercase, strip HTML and URLs, remove punctuation and digits.
2. **Features:** TF-IDF on unigrams and bigrams (sublinear term frequency, `min_df=2`).
3. **Model:** Logistic Regression with `class_weight="balanced"`.
4. **Evaluation:** stratified 80/20 split, reporting accuracy and macro F1.

## Extensions for the internship task

- Compare Naive Bayes, Linear SVM, and a transformer (e.g. `distilbert-base-uncased-finetuned-sst-2-english`).
- Add a neutral class and report per-class precision and recall.
- Add a confusion-matrix plot to the About tab.
- Log experiments with MLflow or a CSV file.
