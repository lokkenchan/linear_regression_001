import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
# statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
# linear_regression_001
from linear_regression_001.features.build_features import build_preprocessor
"""
L.I.N.N.E. Acronym:
===================
- Linearity (linear relationship exists between target and features),

- Independence of Errors (residuals not correlated),

- Normality of Errors (residuals are approx normal),
- No perfect multicollinearity (VIF and correlation matrix),

- Equal Variance of Errors (spread of residuals is consistent across predictions),

Extra
- Outliers and Influence. Cook's Distance > 1 is a significant problem. >0.5 is worth investigating.
"""

def plot_linearity_checks(df, features, target, figsize=(15,10)):
    """
    Plot scatter plots to check linearity assumption

    Parameters
    ----------
    df pd.DataFrame
        Dataframe with features and target
    features list
        List of the column names
    target: str
        Target column name
    figsize: tuple
        Figure size
    """
    n_features = len(features)
    n_cols = 3
    # Ceiling allows to round up for an extra row for stray features
    n_rows = math.ceil(n_features/n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    """axes is a varying type
    it can be 1x1 as a single obj, 1D array, or 2D array.
    Later, axes[idx] is used, if axes as a single obj
    cannot be indexed, so to be consistent w/ the 1D or 2D forms
    it is wrapped as a list.
    ravel() allows a 2D to be flatten into a 1D, so it is consistently 1D
    with iteration.
    """
    axes = axes.ravel() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        axes[idx].scatter(df[feature], df[target], alpha=0.5)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel(target)
        axes[idx].set_title(f'{feature} vs {target}')
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

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

def plot_feature_correlations(df, features, figsize=(10,5)):
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


def independence_of_errors(resids, axis=0):
    """
    Ho: DW=2, errors are not autocorrelated
    Ha: DW!=2, errors are autocorrelated
    Rule of thumb 1.5 to 2.5 are accepted as not autocorrelated
    """

    result = durbin_watson(resids,axis)
    if result < 1.5 or result > 2.5:
        result = f"{result:.2f} => Errors are autocorrelated!"
    else:
        result = f"{result:.2f} => Errors are not autocorrelated."

    return {
        'durbin_watson': result
    }

def normality_of_errors(resid):
    """Plot out the probplot of residuals"""
    plt.figure()
    stats.probplot(resid, dist='norm', plot=plt)
    plt.show()

def equal_var_of_errors(y_true,y_pred):
    """Plot scatter of residuals across predicted values"""
    plt.figure()
    sns.scatterplot(x=y_pred, y=y_true-y_pred)
    plt.title("Residual vs Prediction of Charges")
    plt.ylabel("Residual")
    plt.xlabel("Prediction")
    plt.show()


def cooks_distance(OLS_model):
    """
    Return the dataframe of cook's distance given the model.

    :param OLS_model: Fitted OLS model with all X features and constant and y.
    """
    print(f"Are there any points worth investigating (>0.5)? {(OLS_model.get_influence().summary_frame()['cooks_d'].sort_values(ascending = False)>=0.5).any()}")
    influence_df = OLS_model.get_influence().summary_frame()['cooks_d'].sort_values(ascending = False)
    print(influence_df.to_string())
    return influence_df



