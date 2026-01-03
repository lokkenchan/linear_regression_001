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
        # find features/numerical and features/categorical
        # get and {} for empty if not found, get keys
        feature_cols.extend(schema.get("features", {}).get(group,{}).keys())

    target_col = schema["dataset"]["target"]
    # feature_cols is a list, so lists can only be concatenated with other lists
    expected_cols = set(feature_cols + [target_col])

    actual_cols = set(df.columns)

    if not schema.get("validation",{}).get("allow_extra_columns",False):
        extra_cols = actual_cols - expected_cols
        if extra_cols:
            errors.append(f"Unexpected columns: {extra_cols}")

    missing_cols = expected_cols - actual_cols
    if missing_cols:
        errors.append(f"Missing rquired columns: {missing_cols}")

    # -------------------
    # 2. Data type checks
    # -------------------

    def check_dtype(col,expected):
        actual = df[col].dtype
        if expected == "int" and not pd.api.types.is_integer_dtype(actual):
            return False
        if expected == "float" and not pd.api.types.is_float_dtype(actual):
            return False
        if expected == "str" and not pd.api.types.is_object_dtype(actual):
            return False
        return True

    for group, features in schema.get("features",{}).items():
        for col, rules in features.items():
            if col in df and not check_dtype(col,rules["dtype"]):
                errors.append(
                    f"{col} expected {rules['dtype']} but got {df[col].dtype}"
                )
    target_rules = schema["target"][target_col]
    if target_col in df and not check_dtype(target_col, target_rules["dtype"]):
        errors.append(
            f"Target {target_col} expected {target_rules['dtype']}"
            f"but got {df[target_col].dtype}"
        )

    # -------------------
    # 3. Missing values
    # -------------------
    max_null_rate = schema.get("validation",{}).get("max_null_rate",0)

    for col in expected_cols:
        null_rate = df[col].isnull().mean()
        if null_rate > max_null_rate:
            errors.append(
                f"{col} null rate {null_rate:.2%} exceeds {max_null_rate:.2%}"
            )


    # -------------------
    # 4. Numerical constraints
    # -------------------
    for col, rules in schema.get("features",{}).get("numerical",{}).items():
        if col not in df:
            continue
        if "min" in rules and (df[col]<rules["min"]).any():
            errors.append(f"{col} has values below {rules['min']}")
        if "max" in rules and (df[col]>rules["max"]).any():
            errors.append(f"{col} has values above {rules['max']}")

    # -------------------
    # 5. Categorical constraints
    # -------------------
    for col,rules in schema.get("features",{}).get("categorical",{}).items():
        if col not in df:
            continue
        allowed = set(rules.get("allowed",[]))
        invalid = set(df[col].dropna().unique()) - allowed
        if invalid:
            errors.append(f"{col} contains invalid values: {invalid}")

    # -------------------
    # Final Decision
    # -------------------
    if errors:
        raise SchemaValidationError(
            "Schema validation failed:\n" + "\n".join(errors)
        )