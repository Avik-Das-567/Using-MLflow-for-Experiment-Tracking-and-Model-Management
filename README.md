# Using MLflow for Experiment Tracking and Model Management

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-3.9.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tracking-Local%20Server-lightgrey?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>🔗 Live App: <a href="https://sentiment-analysis-flipkart-reviews.streamlit.app">sentiment-analysis-flipkart-reviews.streamlit.app</a></b>
  &nbsp;|&nbsp;
  <b>📦 Base Project: <a href="https://github.com/Avik-Das-567/Sentiment-Analysis-Flipkart-Product-Reviews">Sentiment-Analysis-Flipkart-Product-Reviews</a></b>
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Relationship to Base Project](#2-relationship-to-base-project)
3. [Repository Structure](#3-repository-structure)
4. [MLflow Architecture & Tracking Setup](#4-mlflow-architecture--tracking-setup)
5. [Training Script 1 — Multi-Model Experiment Tracking](#5-training-script-1--multi-model-experiment-tracking)
6. [Training Script 2 — Hyperparameter Tuning with Run Comparison](#6-training-script-2--hyperparameter-tuning-with-run-comparison)
7. [What Gets Logged — Parameters, Metrics, Artifacts & Tags](#7-what-gets-logged--parameters-metrics-artifacts--tags)
8. [MLflow Model Registry & Version Management](#8-mlflow-model-registry--version-management)
9. [MLflow UI — Experiments & Visualizations](#9-mlflow-ui--experiments--visualizations)
10. [Experiment Results & Run Comparison](#10-experiment-results--run-comparison)
11. [Streamlit Web Application](#11-streamlit-web-application)
12. [Configuration Reference](#12-configuration-reference)
13. [Local Setup & Execution](#13-local-setup--execution)
14. [Future Scope](#14-future-scope)

---

## 1. Project Overview

This project extends an existing end-to-end Sentiment Analysis system by integrating **MLflow 3.9.0** for full ML lifecycle management. The core objective is to demonstrate production-grade experiment tracking, reproducibility, model versioning, and governance — applied to the real-world problem of classifying Flipkart product reviews as Positive or Negative.

Rather than simply training models, this project treats every training run as a **tracked, reproducible, auditable experiment**, where all decisions — hyperparameter choices, feature engineering strategies, evaluation outcomes — are recorded and queryable through a local MLflow tracking server.

**What this project specifically demonstrates:**

- Setting up a named MLflow experiment and launching multiple tracked runs within it
- Logging parameters, metrics, models, and file artifacts per run using MLflow Tracking APIs
- Assigning descriptive custom run names to make the MLflow UI immediately readable
- Registering all model candidates under a single centralized MLflow Model Registry entry
- Applying structured tags to individual model versions for governance and lifecycle management
- Comparing run metrics and hyperparameters visually through MLflow's built-in UI charts
- Serving predictions through a Streamlit app backed by the best registered model

**MLflow version used:** `3.9.0` (confirmed from MLflow UI, running at `http://127.0.0.1:5000`)

---

## 2. Relationship to Base Project

This repository is a direct extension of the [Sentiment Analysis of Real-time Flipkart Product Reviews](https://github.com/Avik-Das-567/Sentiment-Analysis-Flipkart-Product-Reviews) project. The entire `src/` package — `config.py`, `data_loader.py`, `preprocessing.py`, `feature_engineering.py`, `train.py`, `evaluate.py`, `model_registry.py`, and `inference.py` — is carried over unchanged.

The key additions in this repository are:

| New File | Purpose |
|---|---|
| `run_training_mlflow.py` | Tracks both Logistic Regression and LinearSVC as separate MLflow runs under a shared experiment |
| `train_with_mlflow.py` | Performs hyperparameter tuning over Logistic Regression configurations, with each configuration logged as a distinct MLflow run |
| `mlflow.db` | SQLite file used as the MLflow backend store for persisting experiment and run metadata locally |
| `mlflow` (in `requirements.txt`) | The only new dependency added over the base project |

`run_training.py` is retained as the non-MLflow baseline training script, providing a clean reference point for comparing instrumented vs. uninstrumented training flows.

---

## 3. Repository Structure

```
MLflow_Flipkart_Sentiment_Project/
│
├── data/
│   └── data.csv                      # 8,518 Flipkart product reviews
│
├── artifacts/                        # Joblib-serialized artifacts for Streamlit inference
│   ├── sentiment_model.pkl           # Best model from baseline training (LinearSVC)
│   ├── vectorizer.pkl                # Fitted TF-IDF vectorizer
│   └── model_metadata.json           # Model name and F1 score metadata
│
├── src/                              # Core ML pipeline package (shared with base project)
│   ├── config.py                     # Centralized path and hyperparameter constants
│   ├── data_loader.py                # CSV ingestion with null filtering
│   ├── preprocessing.py              # Text cleaning and lemmatization
│   ├── feature_engineering.py        # TF-IDF vectorization (fit/transform)
│   ├── train.py                      # Model definitions — LogisticRegression & LinearSVC
│   ├── evaluate.py                   # F1-Score evaluation loop
│   ├── model_registry.py             # Joblib artifact serialization and metadata export
│   └── inference.py                  # Prediction pipeline for Streamlit app
│
├── app.py                            # Streamlit web application
├── run_training.py                   # Baseline training pipeline (no MLflow)
├── run_training_mlflow.py            # MLflow-tracked multi-model experiment
├── train_with_mlflow.py              # MLflow-tracked hyperparameter tuning runs
├── mlflow.db                         # SQLite backend store for MLflow tracking server
└── requirements.txt                  # Python dependency manifest
```

---

## 4. MLflow Architecture & Tracking Setup

### 4.1 Tracking Server

MLflow is run as a **local tracking server** backed by a SQLite database. All experiment metadata — run IDs, parameters, metrics, tags, and artifact paths — is persisted in `mlflow.db`, a file committed to the repository root.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The UI is accessible at `http://127.0.0.1:5000`. No remote server, cloud storage, or authentication is configured — this is a fully self-contained local setup ideal for reproducible demonstration and development-time experimentation.

### 4.2 Experiment Namespace

Both MLflow training scripts share a single experiment namespace:

```python
mlflow.set_experiment("Flipkart_Sentiment_Analysis")
```

MLflow creates this experiment on first call if it does not already exist, assigning it a unique experiment ID. All runs from both scripts are grouped under this experiment, enabling cross-run comparison within the same UI view without any additional configuration.

As confirmed in the MLflow Experiments page, two experiments exist in the tracking server:

| Experiment Name | Created | Purpose |
|---|---|---|
| `Flipkart_Sentiment_Analysis` | 02/04/2026, 11:49 AM | All tracked ML runs for this project |
| `Default` | 02/04/2026, 11:27 AM | MLflow system default (auto-created on first use) |

### 4.3 Run Lifecycle

Each call to `mlflow.start_run(run_name=...)` opens a new run context manager. All `log_param()`, `log_metric()`, `log_artifact()`, `log_model()`, and `set_tag()` calls within the `with` block are scoped to that specific run. The run is automatically closed with status `FINISHED` when the block exits cleanly, or `FAILED` if an exception propagates out.

```
mlflow.start_run()
    ├── mlflow.log_param()       → Stored in run metadata (backend store)
    ├── mlflow.log_metric()      → Stored as a time-series metric in backend store
    ├── mlflow.log_artifact()    → File copied to artifact store (default: ./mlruns/)
    ├── mlflow.sklearn.log_model() → Model serialized + registered in Model Registry
    └── mlflow.set_tag()         → Key-value annotation on the run
```

---

## 5. Training Script 1 — Multi-Model Experiment Tracking

**File:** `run_training_mlflow.py`

This script runs the complete baseline ML pipeline and wraps each model's logging in its own MLflow run. Its purpose is to **compare Logistic Regression and LinearSVC side-by-side** as distinct tracked experiments within the same experiment group.

```python
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

df = load_data()
df["cleaned"] = df[TEXT_COLUMN].apply(clean_text)
df["sentiment"] = df[RATING_COLUMN].apply(lambda x: 1 if x >= 4 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["sentiment"],
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_train_vec, X_test_vec, vectorizer = tfidf_features(X_train, X_test)
models = train_models(X_train_vec, y_train)
scores = evaluate_models(models, X_test_vec, y_test)

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_TFIDF"):
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_metric("f1_score", scores[model_name])
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="FlipkartSentimentModel"
        )
        joblib.dump(vectorizer, "temp/vectorizer.pkl")
        mlflow.log_artifact("temp/vectorizer.pkl")
        mlflow.set_tag("project", "Flipkart Sentiment Analysis")
        mlflow.set_tag("dataset", "Flipkart Reviews")
        mlflow.set_tag("model_type", model_name)
        print(f"Logged {model_name} with F1-score: {scores[model_name]:.4f}")
```

### Design Decisions

**Training outside the run loop:** Data loading, preprocessing, splitting, vectorization, and model training all happen _before_ any `mlflow.start_run()` call. This ensures both models are trained on identical data splits with an identical TF-IDF vocabulary, making their F1 scores a strictly controlled comparison. The MLflow run contexts are used purely for logging, not for controlling execution.

**Custom run names (`run_name=f"{model_name}_TFIDF"`):** Produces human-readable identifiers (`LogisticRegression_TFIDF`, `LinearSVM_TFIDF`) in the MLflow Experiments UI, eliminating the need to decode auto-generated run UUIDs when browsing or comparing runs.

**Vectorizer logged as a file artifact:** The fitted `TfidfVectorizer` is serialized to `temp/vectorizer.pkl` via `joblib` and logged with `mlflow.log_artifact()`. This ensures the exact vocabulary mapping used in each run is permanently stored in MLflow's artifact store — critical for reproducibility, since loading a model without its corresponding vectorizer would produce incorrect sparse vector representations at inference time.

**Shared registered model name:** Both models are registered to `registered_model_name="FlipkartSentimentModel"`. Each `log_model()` call auto-increments the version under this single registry entry, enabling side-by-side version comparison within the Model Registry without creating separate registry entries per model type.

---

## 6. Training Script 2 — Hyperparameter Tuning with Run Comparison

**File:** `train_with_mlflow.py`

This script focuses exclusively on **Logistic Regression** and demonstrates how MLflow is used to systematically track the effect of varying hyperparameters across multiple independent runs. Each invocation of `train_with_mlflow()` is a fully self-contained, isolated, and logged experiment run.

```python
EXPERIMENT_NAME = "Flipkart_Sentiment_Analysis"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_with_mlflow(max_iter, C):
    with mlflow.start_run(run_name=f"LogReg_iter={max_iter}_C={C}"):

        df = load_data()
        df["clean_text"] = df["Review text"].apply(clean_text)

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["clean_text"],
            df["Ratings"].apply(lambda x: 1 if x >= 4 else 0),
            test_size=0.2, random_state=42
        )

        X_train, X_test, vectorizer = tfidf_features(X_train_text, X_test_text)

        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)

        model = LogisticRegression(max_iter=max_iter, C=C)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="FlipkartSentimentModel"
        )

if __name__ == "__main__":
    train_with_mlflow(max_iter=200, C=1.0)
    train_with_mlflow(max_iter=300, C=0.5)
```

### Hyperparameter Configurations Executed

| Run Name | `max_iter` | `C` (Inverse Regularization Strength) | F1-Score |
|---|---|---|---|
| `LogReg_iter=200_C=1.0` | 200 | 1.0 — weaker regularization, larger decision boundary tolerance | 0.92 |
| `LogReg_iter=300_C=0.5` | 300 | 0.5 — stronger regularization, penalizes large coefficients more | 0.92 |

### Structural Difference from Script 1

Unlike `run_training_mlflow.py` where data loading and training happen once outside the run loop, here each `train_with_mlflow()` call reloads, preprocesses, splits, vectorizes, and trains entirely within its own run context. This makes every run independently reproducible — any single call can be re-executed in isolation with only its logged `max_iter` and `C` values, without relying on shared external state.

**Run naming convention** (`f"LogReg_iter={max_iter}_C={C}"`): Encodes hyperparameter values directly into the run name, making the MLflow UI self-documenting. No lookup into parameter tables is needed to understand what each run tested.

**`C` parameter — what it controls:** In scikit-learn's `LogisticRegression`, `C` is the inverse of regularization strength. A smaller `C` applies stronger L2 regularization, shrinking coefficient magnitudes and reducing overfitting risk. A larger `C` relaxes regularization, giving the model more freedom to fit training data. Both values tested here (1.0 and 0.5) produced identical F1 scores, suggesting the model is not sensitive to regularization strength in this range on this dataset.

---

## 7. What Gets Logged — Parameters, Metrics, Artifacts & Tags

### 7.1 Parameters (`mlflow.log_param`)

Parameters are scalar values that describe the configuration of a run. They are stored immutably once logged and are displayed in the MLflow UI's parameter comparison columns.

| Parameter Key | Logged In | Value(s) |
|---|---|---|
| `vectorizer` | `run_training_mlflow.py` | `"TF-IDF"` |
| `model_name` | `run_training_mlflow.py` | `"LogisticRegression"`, `"LinearSVM"` |
| `test_size` | `run_training_mlflow.py` | `0.2` |
| `max_iter` | `train_with_mlflow.py` | `200`, `300` |
| `C` | `train_with_mlflow.py` | `1.0`, `0.5` |

### 7.2 Metrics (`mlflow.log_metric`)

Metrics are numeric values used to evaluate model performance. MLflow stores metrics as a time series (step-indexed), supporting metric history plots.

| Metric Key | Logged In | Values Recorded |
|---|---|---|
| `f1_score` | Both scripts | 0.9218 (LinearSVC), 0.92 (LogReg both configs) |

### 7.3 Model Artifacts (`mlflow.sklearn.log_model`)

Every run logs its trained scikit-learn model using MLflow's sklearn flavor:

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="FlipkartSentimentModel"
)
```

This serializes the model in MLflow's standard format (wrapping `joblib` internally), stores it under the run's artifact directory at path `model/`, and simultaneously registers a new version in the **Model Registry** under the name `FlipkartSentimentModel`.

### 7.4 File Artifacts (`mlflow.log_artifact`)

In `run_training_mlflow.py`, the fitted TF-IDF vectorizer is additionally saved as a raw file artifact:

```python
joblib.dump(vectorizer, "temp/vectorizer.pkl")
mlflow.log_artifact("temp/vectorizer.pkl")
```

This makes the vectorizer retrievable from any run's artifact store, ensuring the complete inference pipeline (model + vectorizer) can be reconstructed from MLflow alone.

### 7.5 Run-Level Tags (`mlflow.set_tag`)

Tags in `run_training_mlflow.py` annotate each run with project-level metadata:

| Tag Key | Value |
|---|---|
| `project` | `"Flipkart Sentiment Analysis"` |
| `dataset` | `"Flipkart Reviews"` |
| `model_type` | `"LogisticRegression"` or `"LinearSVM"` |

These are filterable in the MLflow UI and useful for querying runs programmatically via the MLflow client API.

---

## 8. MLflow Model Registry & Version Management

### 8.1 Registered Model — `FlipkartSentimentModel`

Every `mlflow.sklearn.log_model(..., registered_model_name="FlipkartSentimentModel")` call creates a new version in the centralized model registry. Across both scripts and all runs, three versions were registered:

| Version | Source Run | Algorithm | Status |
|---|---|---|---|
| Version 1 | `LogisticRegression_TFIDF` | Logistic Regression | — |
| Version 2 | `LinearSVM_TFIDF` | Linear SVC | — |
| Version 3 | `LogReg_iter=300_C=0.5` | Logistic Regression | **Production** ✅ |

As confirmed in the Registered Models page, `FlipkartSentimentModel` is at **Version 3** (latest), last modified on 02/04/2026 at 11:57 AM.

### 8.2 Version 3 — Tags & Governance

Version 3, sourced from the `LogReg_iter=300_C=0.5` run, carries the following model-version-level tags applied via the MLflow UI:

| Tag Name | Tag Value | Purpose |
|---|---|---|
| `algorithm` | `logistic_regression` | Documents the model class used in this version |
| `features` | `tfidf` | Records the text representation technique |
| `metric` | `f1_score` | Identifies the evaluation criterion used for selection |
| `stage` | `production` | Marks this version as the approved production model |
| `use_case` | `sentiment_analysis` | Scopes the model to its intended application domain |

These tags serve as **model governance metadata** — in a team setting, they communicate to other practitioners which model is live, what it uses, and how it was evaluated, without requiring access to training code or experiment logs.

### 8.3 Why Version 3 Was Tagged as Production

Both Logistic Regression runs (`iter=200,C=1.0` and `iter=300,C=0.5`) achieved identical F1 scores of 0.92. Version 3 (`iter=300, C=0.5`) was selected and tagged as production because stronger regularization (`C=0.5`) reduces the risk of overfitting on the training vocabulary, generally yielding better generalization on unseen review text — a pragmatic tiebreaker when metrics are equal.

---

## 9. MLflow UI — Experiments & Visualizations

### 9.1 Experiments Page

The MLflow Experiments page (accessible at `http://127.0.0.1:5000/#/experiments`) lists all registered experiments with their creation timestamps, last-modified times, and optional tags.

![MLflow Experiments](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/MLFlow_Experiments.png)

Both `Flipkart_Sentiment_Analysis` and the system `Default` experiment are visible. Clicking into `Flipkart_Sentiment_Analysis` reveals all individual runs, their parameters, metrics, and status.

### 9.2 F1-Score Metric Plot

MLflow's built-in chart view renders a **horizontal bar chart** comparing `f1_score` across all runs selected in the UI.

![F1 Score Metric Plot](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/f1_score_plot.png)

The chart shows both Logistic Regression hyperparameter runs side by side:

- `LogReg_iter=300_C=0.5` → **0.92**
- `LogReg_iter=200_C=1.0` → **0.92**

The visual confirms that neither hyperparameter configuration yields a measurable advantage on this dataset, validating that the production tag on Version 3 was applied on principled grounds rather than raw metric superiority.

### 9.3 Registered Models Page

The Models section of the MLflow UI (`http://127.0.0.1:5000/#/models`) provides a centralized registry view.

![Registered Models](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/Registered_Models.png)

`FlipkartSentimentModel` is listed at **Version 3**, with tags (`stage: production`, `algorithm: logistic_regression`, `features: tfidf`) visible directly in the registry overview — demonstrating that model governance metadata is surfaced without having to open individual versions.

### 9.4 Model Version Detail — Version 3

The version detail page for `FlipkartSentimentModel v3` (`http://127.0.0.1:5000/#/models/FlipkartSentimentModel/versions/3`) shows:

![Model Version 3](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/Model_Version_3.png)

- **Registered At:** 02/04/2026, 11:57:46 AM
- **Source Run:** `LogReg_iter=300_C=0.5` (hyperlink back to the originating experiment run)
- **Tags:** `algorithm: logistic_regression`, `features: tfidf`, `metric: f1_score`, `stage: production`, `use_case: sentiment_analysis`
- **Stage (deprecated field):** None — the newer MLflow registry UI uses tags for lifecycle management instead of the legacy `Staging/Production/Archived` stage enum

---

## 10. Experiment Results & Run Comparison

A consolidated summary of all runs logged under the `Flipkart_Sentiment_Analysis` experiment:

| Run Name | Script | Model | `vectorizer` | `max_iter` | `C` | F1-Score | Registry Version |
|---|---|---|---|---|---|---|---|
| `LogisticRegression_TFIDF` | `run_training_mlflow.py` | Logistic Regression | TF-IDF | 1000 | — | ~0.92 | Version 1 |
| `LinearSVM_TFIDF` | `run_training_mlflow.py` | LinearSVC | TF-IDF | — | — | **0.9218** | Version 2 |
| `LogReg_iter=200_C=1.0` | `train_with_mlflow.py` | Logistic Regression | TF-IDF | 200 | 1.0 | 0.92 | — |
| `LogReg_iter=300_C=0.5` | `train_with_mlflow.py` | Logistic Regression | TF-IDF | 300 | 0.5 | 0.92 | **Version 3 (Production)** |

**Key observations:**
- LinearSVC achieves the highest absolute F1 score (0.9218) but was not selected for the production tag in the registry — the registry's production version is a Logistic Regression model, reflecting a deliberate choice to prioritize the model explored through the hyperparameter tuning workflow.
- Both Logistic Regression hyperparameter configurations converge to the same F1, indicating the model has plateaued on this dataset with TF-IDF features and that further gains would likely require richer representations (e.g., BERT embeddings).

---

## 11. Streamlit Web Application

**File:** `app.py`

The Streamlit app is identical to the base project and remains independent of the MLflow tracking infrastructure. It loads pre-serialized artifacts from the `artifacts/` directory using `joblib`, not from the MLflow artifact store, keeping the deployment runtime free of any MLflow dependency.

```python
st.set_page_config(page_title="Flipkart Sentiment Analyzer", layout="centered")

review = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if review.strip():
        result = predict_sentiment(review)   # From src/inference.py
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter a review.")

with open("artifacts/model_metadata.json") as f:
    metadata = json.load(f)

st.sidebar.write(f"Model: {metadata['model_name']}")    # LinearSVM
st.sidebar.write(f"F1 Score: {metadata['f1_score']:.4f}")  # 0.9218
```

The sidebar dynamically reads from `model_metadata.json` rather than hardcoding values, so any future retraining that updates the artifact will automatically be reflected in the UI. The inference chain (`clean_text → vectorizer.transform → model.predict`) is defined in `src/inference.py` and is structurally identical across both the base and this repository.

**Live App:** [https://sentiment-analysis-flipkart-reviews.streamlit.app](https://sentiment-analysis-flipkart-reviews.streamlit.app)

---

## 12. Configuration Reference

**File:** `src/config.py`

```python
DATA_PATH     = "data/data.csv"    # Input dataset path
MODEL_DIR     = "artifacts"        # Output directory for joblib artifacts
LOG_DIR       = "logs"             # Reserved for future logging integration

TEXT_COLUMN   = "Review text"      # Column used as model input feature
RATING_COLUMN = "Ratings"          # Column used to derive sentiment labels

RANDOM_STATE  = 42                 # Global seed for reproducibility
TEST_SIZE     = 0.2                # 80/20 train-test split ratio
```

All MLflow training scripts import from `config.py` for `TEST_SIZE` and `RANDOM_STATE`, ensuring that the split used in MLflow runs is identical to the baseline — making cross-script metric comparisons valid.

---

## 13. Local Setup & Execution

### Prerequisites

- Python 3.8 or higher
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management.git
cd Using-MLflow-for-Experiment-Tracking-and-Model-Management
```

### Step 2 — Create & Activate a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
mlflow
pandas
numpy
scikit-learn
nltk
streamlit
joblib
matplotlib
seaborn
tqdm
```

### Step 4 — Launch the MLflow Tracking Server

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` in your browser. Keep this terminal running in the background while executing training scripts so all runs are captured in real time.

### Step 5 — Run Multi-Model Experiment Tracking

```bash
python run_training_mlflow.py
```

This logs two runs (`LogisticRegression_TFIDF` and `LinearSVM_TFIDF`) to the `Flipkart_Sentiment_Analysis` experiment and registers both models as Version 1 and Version 2 under `FlipkartSentimentModel`.

**Expected output:**
```
Logged LogisticRegression with F1-score: 0.9200
Logged LinearSVM with F1-score: 0.9218
```

### Step 6 — Run Hyperparameter Tuning Experiment

```bash
python train_with_mlflow.py
```

This sequentially executes two Logistic Regression runs with different hyperparameters and logs each to the same experiment.

**Expected output:**
```
Run completed | F1-score: 0.9200
Run completed | F1-score: 0.9200
```

### Step 7 — Explore Results in MLflow UI

Navigate to `http://127.0.0.1:5000` and:
- Open the `Flipkart_Sentiment_Analysis` experiment to view all runs
- Select multiple runs and click **Compare** to generate metric and parameter comparison charts
- Navigate to **Models → FlipkartSentimentModel** to inspect registered versions and tags

### Step 8 — Run the Streamlit App

```bash
streamlit run app.py
```

The app loads artifacts from `artifacts/` and serves predictions at `http://localhost:8501`.

---

## 14. Future Scope

| Enhancement | Description |
|---|---|
| **Remote MLflow Tracking Server** | Replace the local SQLite backend with a remote server (e.g., hosted on AWS EC2 with S3 artifact store) to support team-wide experiment sharing |
| **Prefect Workflow Orchestration** | Wrap the training pipeline in Prefect flows and tasks to enable scheduled auto-retraining, dependency management, and a visual Prefect Dashboard |
| **Hyperparameter Search with Optuna** | Replace manual grid search with Optuna's Bayesian optimization, logging each trial as an MLflow run for full search history tracking |
| **MLflow Model Serving** | Deploy the production-tagged model version directly via `mlflow models serve` as a REST endpoint, replacing the current joblib-based inference module |
| **Automated Model Promotion** | Write scripts using the MLflow client API (`MlflowClient.set_registered_model_alias`) to automatically promote the best-performing run to production based on metric thresholds |
| **BERT/Transformer Embeddings** | Integrate transformer-based feature extraction as an alternative to TF-IDF, tracked as a separate experiment branch within the same `Flipkart_Sentiment_Analysis` namespace |
| **Data Versioning with DVC** | Combine MLflow experiment tracking with DVC data versioning so each MLflow run is linked to a specific, reproducible snapshot of `data.csv` |

---

## Author

**Avik Das**
GitHub: [@Avik-Das-567](https://github.com/Avik-Das-567)

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this project with attribution.
