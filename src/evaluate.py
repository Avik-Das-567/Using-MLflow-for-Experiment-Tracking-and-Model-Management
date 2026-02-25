from sklearn.metrics import f1_score

def evaluate_models(models, X_test, y_test):
    scores = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        scores[name] = f1_score(y_test, preds)
    return scores