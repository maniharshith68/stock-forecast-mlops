import logging
import logging.config
import yaml
import os
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    config_path = Path(__file__).parents[2] / "config" / "logging.yaml"
    logs_dir = Path(__file__).parents[2] / "logs"
    logs_dir.mkdir(exist_ok=True)

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    return logging.getLogger(name)
