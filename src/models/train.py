"""Module providing a logging function"""
import logging
from ..utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def train():
    """
    Docstring for train
    """
    logger.info("Training started")
    logger.debug("Loading dataset")
    logger.warning("Class imbalance detected")
    logger.info("Training completed")


if __name__=="__main__":
    train()

    import mlflow

    # Setting
    # mlflow.set_tracking_uri("https://dagshub.com/lokkenchan/linear_regression_001.mlflow")
    # mlflow.set_experiment("my-first-experiment")

    # # Test Connection
    # print(f"ML Tracking URI: {mlflow.get_tracking_uri()}")
    # print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")
    # # Test Logging
    # with mlflow.start_run():
    #     mlflow.log_param("test_param","test_value")
    #     print("Successfully connected to MLFLOW!")

    import dagshub
    dagshub.init(repo_owner='lokkenchan', repo_name='linear_regression_001', mlflow=True)

    with mlflow.start_run():
        mlflow.log_param('parameter name', 'value')
        mlflow.log_metric('metric name', 1)