import hashlib
from pathlib import Path

# The files that determine what a feature actually contains. If any changes,
# any previously cached .pkl files no longer reflect the current extraction
# logic and must be recomputed. local/extract_feature.py is the actual active
# entry point (by default a thin forward to voicedtracks/); both are hashed so
# this stays correct whether it's left as-is or replaced with a custom approach.
_REPO_ROOT = Path(__file__).parent.parent
_EXTRACTION_SOURCE_FILES = [
    _REPO_ROOT / "local" / "extract_feature.py",
    _REPO_ROOT / "voicedtracks" / "extract_feature.py",
    _REPO_ROOT / "voicedtracks" / "voiced_tracks.py",
]

_VERSION_FILENAME = ".extraction_version"


def extraction_code_hash() -> str:
    h = hashlib.sha256()
    for path in _EXTRACTION_SOURCE_FILES:
        h.update(path.read_bytes())
    return h.hexdigest()


def is_cache_valid(feature_folder) -> bool:
    version_file = feature_folder / _VERSION_FILENAME
    if not version_file.exists():
        return False
    return version_file.read_text().strip() == extraction_code_hash()


def mark_cache_valid(feature_folder):
    (feature_folder / _VERSION_FILENAME).write_text(extraction_code_hash())
