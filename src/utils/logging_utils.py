from pathlib import Path
import logging
import logging.config # setting up loggers, handlers, and formatters
import yaml

def setup_logging(config_path="src/config/logging.yaml"):
    """
    Docstring for setup_logging

    :param config_path: Description
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f.read())

    # Ensure log directory exists
    log_file = config["handlers"]["file"]["filename"]
    Path(log_file).parent.mkdir(parents=True,exist_ok=True)

    logging.config.dictConfig(config)