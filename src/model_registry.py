import joblib
import json
import os
from src.config import MODEL_DIR

def register_best_model(models, scores, vectorizer):
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, f"{MODEL_DIR}/sentiment_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

    metadata = {
        "model_name": best_model_name,
        "f1_score": scores[best_model_name]
    }

    with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return best_model_name, scores[best_model_name]