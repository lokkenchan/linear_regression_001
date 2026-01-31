from linear_regression_001.utils.paths import RAW, INTERIM, PROCESSED, FEATURES
import pandas as pd

# Defined schema to enforce
SCHEMA = {
    "age":"int64",
    "sex":"string",
    "bmi":"float64",
    "children":"int64",
    "smoker":"string",
    "region":"string",
    "charges":"float64"
}

def normalize_column_names(df):
    """
    Makes column names lowercase, replacing non-word characters with _, and removing leading or trailing whitespace or _.

    Parameters:
        df: A dataframe to normalize columns
    Returns:
        df: A normalized dataframe
    """
    df = df.copy()
    df.columns = (
        df.columns
            .str.strip() # removes leading and trailing whitespace
            .str.lower() # make lowercase
            .str.replace(r"[^\w]+","_",regex=True) # Anything that is not a word a-z A-Z 0-9 or _ is replaced with _
            .str.replace(r"_+","_",regex=True) # replace all instances of 1+ _ to be just _ to prevent multiple underscore
            .str.strip("_") # removes any _ from the leading and trailing ends

    )
    print('*normalized*')
    return df




def load_raw_data(file: str) -> pd.DataFrame:
    """
    Load data from /data/raw given the file name
    :param file: the file name
    :type file: str
    :return: Returns the DataFrame with the file name from /data/raw
    :rtype: DataFrame
    """
    df = pd.read_csv(RAW / file)
    df = normalize_column_names(df)

    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"Missing Columns: {missing}")
    print('*missing checked*')

    # Enforce dtypes
    df = df.astype(SCHEMA)
    print('*schema enforced*')

    return df

def load_clean_data(file: str) -> pd.DataFrame:
    """
    Load data from /data/interim given the file name
    :param file: the file name
    :type file: str
    :return: Returns the DataFrame with the file name from /data/interim
    :rtype: DataFrame
    """
    df = pd.read_csv(INTERIM / file)

    return df

def load_features_data(file: str) -> pd.DataFrame:
    """
    Docstring for load_features_data

    :param file: Description
    :type file: str
    :return: Description
    :rtype: DataFrame
    """
    df = pd.read_csv(FEATURES / file)
    return df

def load_processed_data(file: str) -> pd.DataFrame:
    """
    Docstring for load_processed_data

    :param file: Description
    :type file: str
    :return: Description
    :rtype: DataFrame
    """
    df = pd.read_csv(PROCESSED / file)
    return df