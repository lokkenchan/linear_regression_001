import pandas as pd

def missing_summary(df):
    return(
        df.isna()
          .mean()
          .sort_values(ascending=False)
    )


def preprocess(df: pd.DataFrame)-> pd.DataFrame:
    pass