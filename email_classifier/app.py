import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import re
import unicodedata
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# =========================
# LOAD MODELS
# =========================
model = pickle.load(open("models/best_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# =========================
# TEXT PREPROCESSING
# =========================
def preprocess_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(words)

    def get_pos(tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('R'):
            return 'r'
        return 'n'

    words = [lemmatizer.lemmatize(w, get_pos(t)) for w, t in pos_tags]
    return " ".join(words)

# =========================
# SINGLE PREDICTION
# =========================
def predict_email(text):
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])

    pred = model.predict(vec)[0]
    label = label_encoder.inverse_transform([pred])[0]

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(vec)))
    else:
        confidence = None

    return label, confidence

# =========================
# BATCH PREDICTION (CSV)
# =========================
def batch_predict(df):
    results = []
    
    for text in df["subject"]:
        clean = preprocess_text(text)
        vec = vectorizer.transform([clean])

        pred = model.predict(vec)[0]
        label = label_encoder.inverse_transform([pred])[0]

        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(vec)))
        else:
            confidence = None

        results.append((label, confidence))

    df["prediction"] = [r[0] for r in results]
    df["confidence"] = [r[1] for r in results]

    return df

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Email Classifier", layout="centered")

st.title("📧 Email Classification System")
st.write("Classify emails into Spam, Important, Social, Promotions")

# -------------------------
# SINGLE PREDICTION
# -------------------------
st.header("🔹 Single Prediction")

text = st.text_area("Enter email subject")

if st.button("Predict Single"):
    if text.strip() == "":
        st.error("Please enter text")
    else:
        label, conf = predict_email(text)

        st.success(f"Category: {label}")
        st.info(f"Confidence: {conf:.2f}" if conf else "Confidence not available")


# -------------------------
# BATCH PREDICTION
# -------------------------
st.header("🔹 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file with 'subject' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "subject" not in df.columns:
        st.error("CSV must contain a 'subject' column")
    else:
        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            result_df = batch_predict(df)

            st.success("Prediction completed!")

            st.dataframe(result_df.head())

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )