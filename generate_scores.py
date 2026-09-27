import itertools
import pickle

import pandas as pd

from common.compute_scores import compute_scores
from common.get_recording_ids import get_recording_ids
from common.results_cache import RESULTS_FILE, is_results_cache_valid, mark_results_cache_valid
from config import FEATURE_FOLDER, PROTOCOL_FILE


def main():
    if is_results_cache_valid(PROTOCOL_FILE):
        print(f"{RESULTS_FILE} already up to date (scoring code and protocol unchanged) -- nothing to do.")
        return

    recordings = pd.read_csv(PROTOCOL_FILE)

    unique_clients = recordings["client_id"].unique()
    unique_phrases = recordings["phrase_id"].unique()

    # PR-dvr recordings (played back via the low-quality built-in DVR speaker) are
    # excluded from evaluation, per the tech report: their audio quality is poor
    # enough that they'd be unlikely to pass a speaker verification system anyway.
    dvr_rec_ids = set(recordings.loc[recordings["pb_device"] == "D", "rec_id"])

    cap_sets = []

    for client, phrase in itertools.product(unique_clients, unique_phrases):
        stored_recordings = get_recording_ids(recordings, client, phrase, "stored")
        playback_recordings = get_recording_ids(recordings, client, phrase, "playback")
        playback_recordings = [r for r in playback_recordings if r not in dvr_rec_ids]
        authentic_recordings = get_recording_ids(recordings, client, phrase, "authentic")

        cap_sets.append({
            "client": client,
            "phrase": phrase,
            "stored_recordings": stored_recordings,
            "playback_recordings": playback_recordings,
            "authentic_recordings": authentic_recordings,
            "playback_scores": compute_scores(FEATURE_FOLDER, playback_recordings, stored_recordings),
            "authentic_scores": compute_scores(FEATURE_FOLDER, authentic_recordings, stored_recordings),
        })

    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(cap_sets, f)
    mark_results_cache_valid(PROTOCOL_FILE)


if __name__ == "__main__":
    main()
