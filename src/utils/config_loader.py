import yaml
import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    # This assumes this file is in src/utils/
    # We go up two levels: src/utils/ -> src/ -> root/
    return Path(__file__).parent.parent.parent

def load_config(config_name="config.yaml"):
    """
    Loads the YAML configuration file.
    """
    root_path = get_project_root()
    config_path = root_path / "configs" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Automatically convert relative paths in config to absolute paths
    # This ensures code runs correctly regardless of where it is executed
    if "paths" in config:
        for key, relative_path in config["paths"].items():
            config["paths"][key] = str(root_path / relative_path)

    return config

# Test block to verify it works when you run this file directly
if __name__ == "__main__":
    conf = load_config()
    print("Project Root:", get_project_root())
    print("Raw Data Path:", conf["paths"]["raw_data"])