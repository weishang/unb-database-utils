import gzip
import os
import pickle

import numpy as np

from .get_filename import get_filename
from local.calculate_similarity_score import calculate_similarity_score


def _load_feature(feature_folder, rec_id):
    path = os.path.join(feature_folder, get_filename(rec_id) + ".pkl.gz")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def compute_scores(feature_folder, incoming_recordings, stored_recordings):
    scores = np.zeros(len(incoming_recordings))

    for i, incoming_id in enumerate(incoming_recordings):
        incoming_feature = _load_feature(feature_folder, incoming_id)

        scores_per_recording = np.zeros(len(stored_recordings))
        for j, stored_id in enumerate(stored_recordings):
            stored_feature = _load_feature(feature_folder, stored_id)
            # replace calculate_similarity_score with your own similarity measure
            scores_per_recording[j] = calculate_similarity_score(incoming_feature, stored_feature)

        scores[i] = scores_per_recording.max() if len(stored_recordings) else np.nan

    return scores
