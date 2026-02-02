from pathlib import Path

# __file__ gets you this file path
# Path(__file__) returns pathlib.Path object to perform operations
# resolve creates the absolute path
# parents[0] is utils package (b/c parent of file), parents[1] is linear_regression_001
# parents[2] is linear_regression_001 as a root linear_regression_001/linear_regression_001 (the first)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
# Uneditted data
RAW = DATA_DIR / "raw"
# External supplementary data to be combined with raw data
EXTERNAL = DATA_DIR / "external"
# Cleaned data
INTERIM = DATA_DIR / "interim"
# Data that includes feature engineered columns - excluding transformations that require a pipeline to prevent data-leakage
PROCESSED = DATA_DIR / "processed"
# Data that includes just features to predict
INFERENCE = DATA_DIR / "inference"
# Output of Inference data
PREDICTIONS = DATA_DIR/ "predictions"
# Where models are saved
MODELS = PROJECT_ROOT / "linear_regression_001/models/saved_models/"