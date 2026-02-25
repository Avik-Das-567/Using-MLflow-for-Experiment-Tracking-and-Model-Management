import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.feature_engineering import tfidf_features
from src.train import train_models
from src.evaluate import evaluate_models
from src.config import TEXT_COLUMN, RATING_COLUMN, TEST_SIZE, RANDOM_STATE

# Set MLflow experiment
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

df = load_data()
df["cleaned"] = df[TEXT_COLUMN].apply(clean_text)
df["sentiment"] = df[RATING_COLUMN].apply(lambda x: 1 if x >= 4 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"],
    df["sentiment"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

X_train_vec, X_test_vec, vectorizer = tfidf_features(X_train, X_test)

models = train_models(X_train_vec, y_train)
scores = evaluate_models(models, X_test_vec, y_test)

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_TFIDF"):

        # Log parameters
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)

        # Log metric
        mlflow.log_metric("f1_score", scores[model_name])

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="FlipkartSentimentModel"
        )

        # Log vectorizer as artifact
        os.makedirs("temp", exist_ok=True)
        import joblib
        joblib.dump(vectorizer, "temp/vectorizer.pkl")
        mlflow.log_artifact("temp/vectorizer.pkl")

        # Add tags
        mlflow.set_tag("project", "Flipkart Sentiment Analysis")
        mlflow.set_tag("dataset", "Flipkart Reviews")
        mlflow.set_tag("model_type", model_name)

        print(f"Logged {model_name} with F1-score: {scores[model_name]:.4f}")