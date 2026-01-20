import yaml
from pathlib import Path
from typing import Dict, Union

from workflow_16s.constants import DEFAULT_CONFIG

def resolve_relative_paths(config: Dict, config_dir: Path) -> Dict:
    for key, value in config.items():
        if isinstance(value, str):
            if value.startswith("./") or value.startswith("../"):
                config[key] = (config_dir / value).resolve()
        elif isinstance(value, dict):
            config[key] = resolve_relative_paths(value, config_dir)
    return config



# Note: For downstream ingestion, the number of parallel workers/cores is set by:
#   execution.threads (preferred), cpu.limit, or execution.cpu_limit (fallback order)
#   If not set, defaults to 8. Max is capped at system CPU count (64).
def get_config(
    config_path: Union[str, Path] = DEFAULT_CONFIG
) -> Dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        print(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in config file: {e}")
    if config is None:
        raise ValueError("Config file is empty or could not be parsed.")
    config_dir = config_path.resolve().parent
    config = resolve_relative_paths(config, config_dir)
    return config