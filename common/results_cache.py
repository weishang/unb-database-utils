import hashlib
from pathlib import Path

from .feature_cache import extraction_code_hash

# The files that determine how results.pkl's scores are computed from the
# (already-cached) features -- similarity scoring itself, how per-recording
# scores get combined, how recordings are selected/excluded per capture set,
# and the id/metadata lookup logic. If any of these change, or the underlying
# feature extraction code changes, or the protocol file itself changes,
# previously generated scores no longer reflect the current pipeline.
_REPO_ROOT = Path(__file__).parent.parent
_SCORING_SOURCE_FILES = [
    _REPO_ROOT / "local" / "calculate_similarity_score.py",
    _REPO_ROOT / "voicedtracks" / "calculate_similarity_score.py",
    Path(__file__).parent / "compute_scores.py",
    Path(__file__).parent / "get_recording_ids.py",
    _REPO_ROOT / "generate_scores.py",
]

RESULTS_FILE = _REPO_ROOT / "results.pkl"
_VERSION_FILENAME = ".results_version"


def results_code_hash(protocol_file) -> str:
    h = hashlib.sha256()
    for path in _SCORING_SOURCE_FILES:
        h.update(path.read_bytes())
    h.update(Path(protocol_file).read_bytes())
    h.update(extraction_code_hash().encode())
    return h.hexdigest()


def is_results_cache_valid(protocol_file) -> bool:
    version_file = _REPO_ROOT / _VERSION_FILENAME
    if not (RESULTS_FILE.exists() and version_file.exists()):
        return False
    return version_file.read_text().strip() == results_code_hash(protocol_file)


def mark_results_cache_valid(protocol_file):
    (_REPO_ROOT / _VERSION_FILENAME).write_text(results_code_hash(protocol_file))
