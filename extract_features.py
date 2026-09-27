import gzip
import pickle

import pandas as pd

from common.feature_cache import is_cache_valid, mark_cache_valid
from common.get_filename import get_filename
from local.extract_feature import extract_feature
from config import FEATURE_FOLDER, PROTOCOL_FILE, RECORDINGS_FOLDER


def main():
    recordings = pd.read_csv(PROTOCOL_FILE)

    FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)

    cache_valid = is_cache_valid(FEATURE_FOLDER)
    if not cache_valid:
        print("Feature extraction code has changed (or this is the first run) -- recomputing all features.")

    skipped = 0
    for rec_id in recordings["rec_id"]:
        filename = get_filename(rec_id)
        feature_file = FEATURE_FOLDER / f"{filename}.pkl.gz"
        wav_file = RECORDINGS_FOLDER / f"{filename}.wav"

        if cache_valid and feature_file.exists():
            skipped += 1
            continue

        print(f"Extract feature set for recording {wav_file}")
        feature = extract_feature(str(wav_file))

        with gzip.open(feature_file, "wb") as f:
            pickle.dump(feature, f)

    if skipped:
        print(f"Skipped {skipped} recording(s) already cached with the current extraction code.")

    mark_cache_valid(FEATURE_FOLDER)


if __name__ == "__main__":
    main()
