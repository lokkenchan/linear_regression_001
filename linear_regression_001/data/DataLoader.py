from linear_regression_001.utils.paths import RAW
import pandas as pd

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
    print('normalized')
    return df



def load_raw_data(file: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / file)
    df = normalize_column_names(df)

    return df