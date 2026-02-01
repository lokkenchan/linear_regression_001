import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

# linear_regression_001
from linear_regression_001.utils import MODELS
from linear_regression_001.data import load_processed_data
from linear_regression_001.features import split_features_target

def train_models():
    df = load_processed_data("processed_insurance.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dectect column types
    categorical_cols = X_train.select_dtypes(exclude="number").columns
    numeric_cols = X_train.select_dtypes(include="number").columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train linear regression model
    model.fit(X_train, y_train)

    # Train baseline dummy model
    baseline = DummyRegressor(strategy = "mean")
    baseline.fit(X_train,y_train)

    # Save
    joblib.dump(model, MODELS / "linear_regression.pkl")
    joblib.dump(baseline, MODELS / "baseline.pkl")

    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_models()