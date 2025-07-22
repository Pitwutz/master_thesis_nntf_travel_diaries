import os
from pathlib import Path


def get_project_root() -> Path:
    # Use environment variable if set, otherwise auto-detect
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # Fallback: assume config.py is in the project root or one subdir below
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CODE_DIR = PROJECT_ROOT / "code"
SRC_DIR = PROJECT_ROOT / "src"
