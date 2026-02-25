import pandas as pd
from src.config import DATA_PATH, TEXT_COLUMN, RATING_COLUMN

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN])
    return df