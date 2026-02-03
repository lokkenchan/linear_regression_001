# 3rd Party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# linear_regression_001
from linear_regression_001.data.loader import load_processed_data
from linear_regression_001.features.build_features import split_features_target, build_features, FEATURE_LIST
from linear_regression_001.utils.paths import MODELS

MODEL_PATH = MODELS / "linear_regression.pkl"

def evaluate(y_true, y_pred, name = "Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{name} Performance")
    print("-" * 25)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")


def evaluate_model():
    df = load_processed_data("processed_insurance.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    evaluate(y_test,preds)



if __name__ == "__main__":
    evaluate_model()