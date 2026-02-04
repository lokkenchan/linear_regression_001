# 3rd Party
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

# linear_regression_001
from linear_regression_001.utils import MODELS
from linear_regression_001.data import load_raw_data
# Note: build_preprocessor defines transforms with ColumnTransformer, Scaler, and Encoder.
from linear_regression_001.features import split_features_target,build_features,build_preprocessor, FEATURE_LIST

def train_models():
    """
    Inputs raw data with cleaning, followed by feature_building, and preprocessing (encoding/scaling)
    and finally fits
    """
    # Load raw data
    df = load_raw_data("insurance.csv")
    # Separate X and y
    X, y = split_features_target(df)
    # Build out features from feature engineering
    X_features_built = build_features(X, FEATURE_LIST)
    df = pd.concat([X_features_built,y],axis=1)
    # Setup preprocessor from build_features.py
    preprocessor = build_preprocessor()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_features_built, y, test_size=0.2, random_state=42)

    # Setup pipeline
    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train linear regression model
    model.fit(X_train, y_train)

    # Train baseline dummy model
    baseline = Pipeline([
        ("preprocessing",preprocessor),
        ("model", DummyRegressor(strategy="mean"))
    ])
    baseline.fit(X_train,y_train)

    # Save
    joblib.dump(model, MODELS / "linear_regression.pkl")
    joblib.dump(baseline, MODELS / "baseline.pkl")

    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_models()