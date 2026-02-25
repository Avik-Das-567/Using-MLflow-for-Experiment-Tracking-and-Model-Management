# Using MLflow for Experiment Tracking and Model Management

### App Link: https://sentiment-analysis-flipkart-reviews.streamlit.app

## Project Overview
This project demonstrates the use of **MLflow** for experiment tracking, model management, and reproducibility in a machine learning workflow.  
The use case is **Sentiment Analysis of Flipkart Product Reviews**, where multiple models are trained, evaluated, compared, and registered using MLflow.

The project covers the end-to-end ML lifecycle:
- Data preprocessing
- Feature engineering
- Model training & evaluation
- Experiment tracking
- Model versioning and registry
- Visualization of metrics and hyperparameters

---

## Project Structure

```
MLflow_Flipkart_Sentiment_Project/
│
├── artifacts/    # MLflow logged artifacts
│   ├── vectorizer.pkl
│   ├── sentiment_model.pkl
│   └── model_metadata.json
├── data/
│   └── data.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── model_registry.py
│
├── app.py 
├── train_with_mlflow.py    # MLflow experiment tracking script
├── run_training.py
├── run_training_mlflow.py
├── requirements.txt
└── mlflow.db    # MLflow backend store
```

---

## Objective
The main objective of this project is to:
- Learn and apply MLflow for tracking ML experiments
- Log parameters, metrics, models, and artifacts
- Compare multiple model runs
- Register and manage models with versioning and tags
- Ensure reproducibility of ML experiments

---

## Dataset
- **Source:** Flipkart product reviews  
- **Size:** 8,518 reviews  
- **Target:** Binary sentiment classification  
  - Positive (Rating ≥ 4)
  - Negative (Rating < 4)

---

## Tech Stack

- mlflow
- pandas
- numpy
- scikit-learn
- nltk
- streamlit
- joblib
- matplotlib
- seaborn
- tqdm

---

## MLflow Features Demonstrated

### Experiment Tracking:
- Multiple runs tracked under a single experiment
- Custom run names for better readability

### Parameter Logging:
- Model hyperparameters (e.g., `max_iter`, `C`)
- Feature extraction method

### Metric Logging:
- F1-score used as the primary evaluation metric
- Metric comparison across runs

### Model Logging:
- Models logged using `mlflow.sklearn.log_model`
- Each run creates a new model version

### Model Registry & Versioning:
- Centralized model registry
- Automatic version creation
- Production-ready model tagging

### Visualization:
- Metric comparison plots
- Hyperparameter comparison through MLflow UI

---

## How to Run the Project

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Start MLflow UI
```
mlflow ui
```
Open in browser:
```
http://127.0.0.1:5000
```
### 3. Run Training with MLflow
```
python train_with_mlflow.py
```
### 4. Run Streamlit App
```
streamlit run app.py
```

---

## Results

- Achieved **F1-score ≈ 0.92**

- Multiple model versions registered

- Model performance evaluated on held-out test data

- Best-performing model tagged as **production** in MLflow Model Registry

---

## MLflow Dashboard Screenshots

- ### MLflow Experiments :

![MLflow experiments](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/MLFlow_Experiments.png)

- ### Metric Plots (F1 Score) :

![F1 Score](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/f1_score_plot.png)

- ### Registered Model - FlipkartSentimentModel :

![Registered Model](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/Registered_Models.png)

- ### Registered Model Version & Tags :

![Registered Model Version](https://github.com/Avik-Das-567/Using-MLflow-for-Experiment-Tracking-and-Model-Management/blob/main/MLflow%20Dashboard%20Screenshots/Model_Version_3.png)

---
