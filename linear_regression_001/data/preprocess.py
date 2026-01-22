import pandas as pd

def missing_summary(df):
    return(
        df.isna()
          .mean()
          .sort_values(ascending=False)
    )

def drop_high_missing_rows(df,threshold=0.5)->pd.DataFrame:
    '''
    Remove rows that have data missing beyond the threshold. 50% by default.

    :param df: DataFrame preprocessed
    :param threshold: default of 0.5 (50%)
    :return: DataFrame keeps only the rows that have more than the threshold present in data
    :rtype: DataFrame with cleaned of rows with missing data beyond threshold
    '''
    before = len(df)
    df = df[df.isna().mean(axis=1) < threshold]
    after = len(df)
    print(f"Dropped {before - after} rows (> {threshold:.0%} missing)")

    return df

def preprocess(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()

    df = drop_high_missing_rows(df, threshold = 0.5)

    return df