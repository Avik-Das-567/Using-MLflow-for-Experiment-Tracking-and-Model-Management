from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LinearSVM": LinearSVC()
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained