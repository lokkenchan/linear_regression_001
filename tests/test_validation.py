import pandas as pd
import yaml
import pytest
from src.data.validation import validate_schema

def test_schema_valid_data():
    """
    Tests a df with a valid schema to the yaml schema.

    Params:
        None

    Returns:
        None
    """
    df = pd.DataFrame({
        "age":[25],
        "sex":["female"],
        "bmi":[22.5],
        "children":[1],
        "smoker":["yes"],
        "region":["southwest"],
        "charges":[5000]
    })

    with open("config/schema.yaml",encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    validate_schema(df, schema) # should not raise

def test_schema_missing_column():
    """
    Tests a df with a missing column to the yaml schema.

    Params:
        None

    Returns:
        None
    """
    df = pd.DataFrame({
        "age":[25],
        "sex":["female"],
        "bmi":[22.5],
        "children":[1],
        "smoker":["yes"],
        "charges":[5000]
    })

    with open("config/schema.yaml",encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    with pytest.raises(Exception):
        validate_schema(df,schema)