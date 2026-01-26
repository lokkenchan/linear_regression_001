import pandas as pd
import numpy as np

TARGET = "charges"
def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

