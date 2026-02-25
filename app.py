import streamlit as st
from src.inference import predict_sentiment
import json

st.set_page_config(page_title="Flipkart Sentiment Analyzer", layout="centered")

st.title("Flipkart Review Sentiment Analyzer")
st.write("Real-time sentiment prediction using an MLOps-trained model")

review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if review.strip():
        result = predict_sentiment(review)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter a review.")

with open("artifacts/model_metadata.json") as f:
    metadata = json.load(f)

st.sidebar.header("Model Info")
st.sidebar.write(f"Model: {metadata['model_name']}")
st.sidebar.write(f"F1 Score: {metadata['f1_score']:.4f}")