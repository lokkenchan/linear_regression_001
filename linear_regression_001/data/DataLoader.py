from linear_regression_001.utils.paths import RAW
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW / 'insurance.csv')