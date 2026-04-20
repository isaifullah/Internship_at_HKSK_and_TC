import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    label = "Real News ✅" if pred == 1 else "Fake News 🛑"
    confidence = max(prob)
    
    return label, confidence

st.title("📰 Fake News Detector")

menu = ["Single Prediction", "Batch Prediction"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Single Prediction":
    text = st.text_area("Enter News Text")
    
    if st.button("Predict"):
        label, conf = predict(text)
        st.success(label)
        st.info(f"Confidence: {conf:.2f}")

elif choice == "Batch Prediction":
    file = st.file_uploader("Upload CSV", type=["csv"])
    
    if file:
        import pandas as pd
        df = pd.read_csv(file)
        df['prediction'] = df['text'].apply(lambda x: predict(x)[0])
        st.write(df)