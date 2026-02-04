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

def evaluate_model(model_path, test_data_path, baseline_path=None):
    # Load model and test data to build features
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

    # Baseline
    if baseline_path:
        results['baseline_comparison'] = compare_to_baseline(
            baseline_path, X_test_features, y_test, y_pred
            )

    # Residual analysis
    results['residual_analysis'] = analyze_residuals(y_test,y_pred)

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
    return 1 - ( ((1 - r2) * (n - 1)) / (n - n_features - 1) )

def compare_to_baseline(baseline_path,X_test,y_test,y_pred):
    baseline = joblib.load(baseline_path)
    baseline_pred = baseline.predict(X_test)

    model_rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    baseline_rmse = np.sqrt(mean_squared_error(y_test,baseline_pred))

    improvement = ((model_rmse - baseline_rmse)/ baseline_rmse) * 100

    return {
        'baseline_rmse': baseline_rmse,
        'model_rmse': model_rmse,
        'improvement_percent': improvement,
        'beats_baseline': model_rmse < baseline_rmse
    }

def analyze_residuals(y_true, y_pred):
    """Analyze residuals for model diagnostics"""

    residuals = y_true - y_pred

    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'residual_skewness': pd.Series(residuals).skew(),
        'residual_kurtosis': pd.Series(residuals).kurtosis(),
        'max_overestimation': np.min(residuals),
        'max_underestimation': np.max(residuals),
        'residuals_within_1std': np.mean(np.abs(residuals)<np.std(residuals))*100
    }



if __name__ == "__main__":
    evaluate_model(
        model_path = MODELS / "linear_regression.pkl",
        test_data_path = INTERIM / "test.csv",
        baseline_path = MODELS / "baseline.pkl"
    )