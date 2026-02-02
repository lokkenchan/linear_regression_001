import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

"""
This is used to define feature creation, encoding, scaling, and column selection.
There are TWO types of features:
1) Feature engineering such as ratios, bins, flags defined in build_features.py
2) Feature transform such as scaling or encoding defined in build_features.py via preprocessor
Both 1) and 2) are not saved as data.
"""

TARGET = "charges"
def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


# --- Individual feature definitions ---
# Deterministic transformations only

def feat_bmi_smoker(df):
    df["bmi_smoker"] = df["bmi"]*(df["smoker"]=="yes").astype(int)
    return df["bmi_smoker"]


# Raw features
def feat_age(df):
    return df["age"]

def feat_bmi(df):
    return df["bmi"]

def feat_children(df):
    return df["children"]

def feat_region(df):
    return df["region"]

def feat_smoker(df):
    return df["smoker"]

def feat_sex(df):
    return df["sex"]



# --- Feature registry (menu of available features) ---

FEATURE_REGISTRY = {
    # Raw
    "age": feat_age,
    "bmi": feat_bmi,
    "children": feat_children,
    "region": feat_region,
    "smoker": feat_smoker,
    "sex": feat_sex,

    # Engineered
    "bmi_smoker":feat_bmi_smoker
}

# FEATURE_LIST - Final feature_list
FEATURE_LIST = ["age","bmi","children","region","smoker","sex","bmi_smoker"]

# --- Main builder ---

def build_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Build features from DataFrame returning only the features.
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

def build_preprocessor():
    """
    Create encoder for numeric_cols and categorical_cols for later use.
    Remainder is specified as passthrough to not automatically drop a column.
    For train it would be used as preprocessor.fit_transform(X_train).
    For predict it would be used as preprocessor.transform(X_new).
    """
    numeric_cols = ["age","bmi","children","bmi_smoker"]
    categorical_cols = ["region","smoker","sex"]
    preprocessor = ColumnTransformer([
        ("num",StandardScaler(),numeric_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=False),categorical_cols)
    ], remainder="passthrough")

    return preprocessor