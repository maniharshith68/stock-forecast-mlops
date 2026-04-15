import yaml
from pathlib import Path
from typing import Any

_config = None

def load_config() -> dict:
    global _config
    if _config is None:
        config_path = Path(__file__).parents[2] / "config" / "config.yaml"
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    return _config

def get(key_path: str, default: Any = None) -> Any:
    """Dot-notation access: get('aws.region')"""
    config = load_config()
    keys = key_path.split(".")
    val = config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val
