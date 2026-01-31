from pathlib import Path

# __file__ gets you this file path
# Path(__file__) returns pathlib.Path object to perform operations
# resolve creates the absolute path
# parents[0] is utils package (b/c parent of file), parents[1] is linear_regression_001
# parents[2] is linear_regression_001 as a root linear_regression_001/linear_regression_001 (the first)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
EXTERNAL = DATA_DIR / "external"
PROCESSED = DATA_DIR / "processed"
FEATURES = DATA_DIR / "features"
INFERENCE = DATA_DIR / "inference"
MODELS = PROJECT_ROOT / "linear_regression_001/models/saved_models/"