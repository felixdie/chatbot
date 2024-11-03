from pathlib import Path
import os
from helper_functions.config_helpers import read_yaml

# Set paths of repo
REPOSITORY_PATH = Path(os.path.abspath(__file__)).parent.parent.parent.absolute()
PROJECT_PATH = os.path.join(REPOSITORY_PATH, "chatbot")
CONFIG_PATH = os.path.join(PROJECT_PATH, "config")

# Load config
config = dict()
config["backend"] = read_yaml(os.path.join(CONFIG_PATH, "backend.yaml"))
# config["logger"] = read_yaml(os.path.join(CONFIG_PATH, "logger.yaml"))
