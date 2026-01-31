import joblib
import pandas as pd

# linear_regression_001
from linear_regression_001.utils.paths import MODELS, INFERENCE

MODEL_PATH = MODELS / "linear_regression.pkl"


def predict(X_data_path):
    """
    Give the data path from the inference folder to return predictions
    :param X_data_path: The name of the X data within the inference folder
    """
    model = joblib.load(MODEL_PATH)
    X_new = pd.read_csv(X_data_path)
    preds = model.predict(X_new)
    return preds

if __name__ == "__main__":
    predictions = predict(INFERENCE / "new_data.csv")
    print(predictions[:5])



