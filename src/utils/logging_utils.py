import logging
import logging.config # setting up loggers, handlers, and formatters
import yaml
from pathlib import Path

def setup_logging(config_path="src/config/logging.yaml"):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())

    # Ensure log directory exists
    log_file = config["handlers"]["file"]["filename"]
    Path(log_file).parent.mkdir(parents=True,exist_ok=True)

    logging.config.dictConfig(config)