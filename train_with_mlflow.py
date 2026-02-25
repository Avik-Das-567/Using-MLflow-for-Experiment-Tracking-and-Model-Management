from src.data_loader import load_data
from src.preprocessing import clean_text
from src.feature_engineering import tfidf_features
from src.train import train_models as base_train_model
from src.evaluate import evaluate_models
from src.model_registry import register_best_model

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "Flipkart_Sentiment_Analysis"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_with_mlflow(max_iter, C):
    with mlflow.start_run(run_name=f"LogReg_iter={max_iter}_C={C}"):

        # Load data
        df = load_data()

        # Preprocess
        df["clean_text"] = df["Review text"].apply(clean_text)

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["clean_text"],
            df["Ratings"].apply(lambda x: 1 if x >= 4 else 0),
            test_size=0.2,
            random_state=42
        )

        # TF-IDF features
        X_train, X_test, vectorizer = tfidf_features(
            X_train_text, X_test_text
        )

        # Log parameters
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)

        # Train model
        model = LogisticRegression(max_iter=max_iter, C=C)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)

        # Log metric
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="FlipkartSentimentModel"
        )

        print(f"Run completed | F1-score: {f1:.4f}")

if __name__ == "__main__":
    train_with_mlflow(max_iter=200, C=1.0)
    train_with_mlflow(max_iter=300, C=0.5)