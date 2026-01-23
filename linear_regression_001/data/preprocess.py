import pandas as pd

def missing_summary(df):
    """
    Lists out columns in descending order of mean amount of missing values

    :param df: Description
    """
    return(
        df.isna()
          .mean()
          .sort_values(ascending=False)
    )

def missingness_indicator(df):
    """
    Add a binary column for missingness
    :param df: Description
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f'{col}_missing']=df[col].isna().astype(int)
    return df

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

def clean(df: pd.DataFrame)-> pd.DataFrame:
    """
    Logic check for numeric values being > or >= 0,
    Make categorical values lowercased and trimmed,
    Removing duplicates and resetting the index.

    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: DataFrame
    """
    df = df.copy()
    # numeric
    df = df[df["age"]>0]
    df = df[df["bmi"]>0]
    df = df[df["children"]>=0]
    df = df[df["charges"]>0]
    # categorical - have all values lowercased and remove leading and trailing whitespace
    df["sex"] = df["sex"].str.lower().str.strip()
    df["smoker"] = df["smoker"].str.lower().str.strip()
    df["region"] = df["region"].str.lower().str.strip()

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df

def preprocess(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()

    df = clean(df)
    df = missingness_indicator(df)
    df = drop_high_missing_rows(df, threshold = 0.5)

    return df