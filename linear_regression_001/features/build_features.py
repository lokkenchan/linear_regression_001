import pandas as pd
import numpy as np

TARGET = "charges"
def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# --- Individual feature definitions ---
    # Deterministic transformations only
def feat_log_bmi(df):
    return np.log1p(df["bmi"])

def feat_log_charges(df):
    return np.log1p(df["charges"])

def feat_age_sq(df):
    return df["age"]**2

    # raw features
def feat_age(df):
    return df["age"]

def feat_bmi(df):
    return df["bmi"]

def feat_children(df):
    return df["children"]




# --- Feature registry (menu of available features) ---

FEATURE_REGISTRY = {
    # Raw
    "age": feat_age,
    "bmi": feat_bmi,
    "children": feat_children,

    # Engineered
    "log_charges": feat_log_charges,
    "log_bmi": feat_log_bmi,
    "age_sq": feat_age_sq

}


# --- Main builder ---

def build_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Build features, but allow A/B test.
    We can keep the same data, model, and CV split, but have new feature(s) as difference.


    :param df: Original dataframe to build features.
    :type df: pd.DataFrame
    :param feature_list: Provide list of feature names to look up in FEATURE_REGISTRY
    :type list:
    :return: Description
    :rtype: DataFrame
    """
    df = df.copy()
    out = pd.DataFrame(index=df.index)

    for feat_name in feature_list:
        out[feat_name] = FEATURE_REGISTRY[feat_name](df)

    return out