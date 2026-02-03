import joblib
import pandas as pd

# linear_regression_001
from linear_regression_001.utils.paths import MODELS, INFERENCE, PREDICTIONS
from linear_regression_001.features.build_features import build_features, FEATURE_LIST

# This MUST be the final model when in production
MODEL_PATH = MODELS / "linear_regression.pkl"


def predict_features(X_data_file):
    """
    Give the features to be predicted from
    :param X_data_file: The name of the X data within the inference folder with only raw features

    Returns:
        1) predictions as pred to be used when imported,
        2) predictions as a dataframe called predictions.csv in the PREDICTIONS data directory,
        3) prints out a confirmation of completion.
    """
    model = joblib.load(MODEL_PATH)
    # The X_data_file should not have any features from feature engineering
    X_new = pd.read_csv(INFERENCE / X_data_file)
    # Add the built features with the defined FEATURE_LIST
    X_built_features = build_features(X_new,FEATURE_LIST)
    # Model has the preprocesser to do transforms like encoding and scaling
    preds = model.predict(X_built_features)

    # Save a copy into predictions
    results = pd.DataFrame({'predictions':preds})
    results.to_csv(PREDICTIONS / "predictions.csv", index=False)

    print("Inference completed. Predictions saved.")

    return preds

if __name__ == "__main__":
    # predict_features automatically looks within INFERENCE
    # Run by python3 -m linear_regression_001.models.predict from root dir
    predictions = predict_features("predict_data.csv")
    print(predictions[:5])



