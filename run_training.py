from src.data_loader import load_data
from src.preprocessing import clean_text
from src.feature_engineering import tfidf_features
from src.train import train_models
from src.evaluate import evaluate_models
from src.model_registry import register_best_model
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE
from src.config import TEXT_COLUMN, RATING_COLUMN

df = load_data()

df["cleaned"] = df[TEXT_COLUMN].apply(clean_text)
df["sentiment"] = df[RATING_COLUMN].apply(lambda x: 1 if x >= 4 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["sentiment"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

X_train_vec, X_test_vec, vectorizer = tfidf_features(X_train, X_test)

models = train_models(X_train_vec, y_train)
scores = evaluate_models(models, X_test_vec, y_test)

best_model, best_score = register_best_model(models, scores, vectorizer)

print(f"Best Model: {best_model}")
print(f"Best F1 Score: {best_score}")