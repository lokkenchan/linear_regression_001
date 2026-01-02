# Consider the column names and data types
import pandas as pd

class SchemaValidationError(Exception):
    pass

def validate_schema(df: pd.DataFrame, schema: dict) -> None:
    """
    Validate cleaned, model-ready data against schema.yaml.
    Raises SchemaValidationError if validation fails.

    :param df: Description
    :type df: pd.DataFrame
    :param schema: Description
    :type schema: dict
    """

    errors = []

    # -------------------
    # 1. Column presence
    # -------------------

    feature_cols = []

    for group in ("numerical","categorical"):
        feature_cols.extend(schema.get("features", {}).get(group,{}).keys())

    # -------------------
    # 2. Data type checks
    # -------------------


    # -------------------
    # 3. Missing values
    # -------------------


    # -------------------
    # 4. Numerical constraints
    # -------------------


    # -------------------
    # 5. Categorical constraints
    # -------------------


    # -------------------
    # Final Decision
    # -------------------