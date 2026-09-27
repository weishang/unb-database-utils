import pickle

import numpy as np

from common.get_eer import get_eer
from common.results_cache import RESULTS_FILE


def main():
    with open(RESULTS_FILE, "rb") as f:
        cap_sets = pickle.load(f)

    all_pb_scores = []
    all_au_scores = []

    for cap_set in cap_sets:
        pb_scores = cap_set["playback_scores"]
        au_scores = cap_set["authentic_scores"]

        if len(pb_scores) == 0 or len(au_scores) == 0:
            print(f"Client {cap_set['client']} Phrase {cap_set['phrase']}: skipped (no scores)")
            continue

        eer, thrd = get_eer(pb_scores, au_scores)

        all_pb_scores.append(pb_scores)
        all_au_scores.append(au_scores)

        print(f"Client {cap_set['client']} Phrase {cap_set['phrase']}: eer = {eer:4.2f} with thrd = {thrd:4.2f}")

    eer, thrd = get_eer(np.concatenate(all_pb_scores), np.concatenate(all_au_scores))
    print(f"Overall: eer = {eer:4.2f} with thrd = {thrd:4.2f}")


if __name__ == "__main__":
    main()
