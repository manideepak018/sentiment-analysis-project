"""Streamlit web app for sentiment analysis. Run with: streamlit run app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_model, predict  # noqa: E402

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def get_model():
    return load_model(ROOT / "models" / "sentiment_model.joblib")


@st.cache_data
def get_metrics():
    path = ROOT / "models" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


st.title("💬 Sentiment Analysis")
st.write("Enter text to classify it as **positive** or **negative**.")

tab_single, tab_batch, tab_about = st.tabs(["Single text", "Batch (CSV)", "About model"])

with tab_single:
    text = st.text_area("Your text", placeholder="e.g. The delivery was fast and the product is great!")
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            result = predict([text], model=get_model())[0]
            color = "green" if result["sentiment"] == "positive" else "red"
            st.markdown(f"### Sentiment: :{color}[{result['sentiment'].upper()}]")
            st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.1%}")

with tab_batch:
    st.write("Upload a CSV with a column named **text**.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("The CSV must contain a 'text' column.")
        else:
            with st.spinner("Scoring..."):
                results = predict(df["text"].fillna("").astype(str).tolist(), model=get_model())
            out = df.copy()
            out["sentiment"] = [r["sentiment"] for r in results]
            out["confidence"] = [r["confidence"] for r in results]
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False),
                               "sentiment_results.csv", "text/csv")

with tab_about:
    metrics = get_metrics()
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Macro F1", f"{metrics['f1_macro']:.2%}")
        st.caption(f"Trained on {metrics['n_train']} samples, tested on {metrics['n_test']}.")
    else:
        st.info("No metrics found. Run `python src/train.py` first.")
    st.markdown("**Pipeline:** text cleaning → TF-IDF (1–2 grams) → Logistic Regression (balanced classes).")
