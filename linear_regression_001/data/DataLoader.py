from linear_regression_001.utils.paths import RAW
import pandas as pd

def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW / 'insurance.csv')