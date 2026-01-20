# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import yaml
from pathlib import Path
from typing import Dict, Union

# Local Imports
from workflow_16s import constants

# ==================================== FUNCTIONS ===================================== #

def resolve_relative_paths(config: Dict, config_dir: Path) -> Dict:
    """Converts any relative paths in the configuration to absolute paths based on 
    the directory of the config file."""
    for key, value in config.items():
        if isinstance(value, str):
            # Check if the value is a relative path
            if value.startswith("./") or value.startswith("../"):
                # Convert relative path to absolute path
                config[key] = (config_dir / value).resolve()
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            config[key] = resolve_relative_paths(value, config_dir)
    return config


def get_config(
    config_path: Union[str, Path] = constants.DEFAULT_CONFIG_PATH
) -> Dict:
    # Load the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Resolve any relative paths in the config
    config_dir = Path(config_path).resolve().parent
    config = resolve_relative_paths(config, config_dir)
    
    return config
