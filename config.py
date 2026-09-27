"""Update these folder paths based on your own setup."""
from pathlib import Path

DATABASE_FOLDER = Path(".")
FEATURE_FOLDER = Path("features")

PROTOCOL_FILE = DATABASE_FOLDER / "metadata.csv"
RECORDINGS_FOLDER = DATABASE_FOLDER / "data"
