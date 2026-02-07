import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS

def calculate_vif(df, features=None):
    """
    Calculate Variance Inflation Factor for features.
    VIF = 1 is not correlated
    VIF 1-5 is moderately correlated
    VIF >5-10 highly correlated suggesting multicollinearity.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing features
    features: list
        List of feature column names

    Returns
    -------
    pd.DataFrame: VIF for each feature

    """
    # By default if there are no features specified, select numeric features
    if features is None:
        numeric_df = df.select_dtypes(include='number')
        features = numeric_df.columns.to_list()
    # Otherwise, of the features only select the numeric features.
    else:
        numeric_df = df[features].select_dtypes(include='number')
        features = numeric_df.columns.to_list()

    # In case there are no numeric features.
    if len(features) == 0:
        raise ValueError("No numeric features found for VIF calculation.")

    # Inspect for nulls anywhere in the dataframe and removing them.
    # .any() within a column .any().any() for all of dataframe.
    if numeric_df.isnull().any().any():
        raise ValueError("Warning: Missing values detected. Dropping rows with NaN.")
        numeric_df = numeric_df.dropna()

    # Checking for infinite values and removing them.
    if not np.isfinite(numeric_df.values).all():
        raise ValueError("Infinite values detected in features.")
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [
        # Normally gives VIF of an exogenous variable of
        # a ndarray or dataframe with an index
        # list comprehension allows for the column
        variance_inflation_factor(df[features].values, i)
        for i in range(len(features))
    ]
    return vif_data.sort_values(by='VIF', ascending=False)

def plot_feature_correlations(df, features, figsize=(15,10)):
    """
    Plot of annotated and colored correlation matrix

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with features
    features: list
        List of feature column names
    figsize: tuple
        Figure size
    """
    corr_matrix = df[features].corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_linearity_checks():
    pass
