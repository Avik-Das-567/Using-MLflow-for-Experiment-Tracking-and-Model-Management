import joblib
from src.preprocessing import clean_text

model = joblib.load("artifacts/sentiment_model.pkl")
vectorizer = joblib.load("artifacts/vectorizer.pkl")

def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"