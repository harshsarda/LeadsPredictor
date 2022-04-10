import yaml
import logging.config
import os


def get_logger():
    """get logger using the logging config file"""
    config_path = os.path.join(os.path.dirname(__file__), "../config/logging.yaml")
    with open(config_path, "r") as logging_config:
        config = yaml.safe_load(logging_config)
    logging.config.dictConfig(config)
    logger = logging.getLogger("lead_predictor_logger")
    return logger
