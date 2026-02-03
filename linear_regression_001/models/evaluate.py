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
from linear_regression_001.data.loader import load_raw_data, load_clean_data, load_processed_data
from linear_regression_001.features.build_features import split_features_target, build_features, FEATURE_LIST
from linear_regression_001.utils.paths import INTERIM, MODELS, PREDICTIONS

MODEL_PATH = MODELS / "linear_regression.pkl"

def evaluate_model(model_path, test_data_path, baseline_path=None):
    model = joblib.load(model_path)
    df_test = load_clean_data(test_data_path)
    X_test, y_test = split_features_target(df_test)
    X_test_features = build_features(X_test, FEATURE_LIST)

    # Generate predictions
    y_pred = model.predict(X_test_features)

    # Initialize results dictionary
    results = {}

    # Calculate Comprehensive Metrics
    results['metrics'] = calculate_metrics(y_test,y_pred)

    return results

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true,y_pred) * 100,
        'adjusted_r2': calculate_adjusted_r2(y_true, y_pred, n_features = len(FEATURE_LIST))
    }

def calculate_adjusted_r2(y_true, y_pred, n_features):
    """Calculate adjusted R2 for model comparison.
        n is the total sample size,
        n_features is the number of independent features,
        r2 is sample r2
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - ( ((1 - r2) * (n - 1)) / (n - n_features - 1))


if __name__ == "__main__":
    evaluate_model(
        model_path = MODELS / "linear_regression.pkl",
        test_data_path = INTERIM / "test.csv",
        baseline_path = MODELS / "baseline.pkl"
    )