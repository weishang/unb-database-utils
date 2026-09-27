import numpy as np


def get_eer(playback_scores, authentic_scores):
    """Threshold-sweep EER, vectorized equivalent of the original MATLAB loop."""
    playback_scores = np.sort(np.asarray(playback_scores, dtype=float).ravel())
    authentic_asc = np.sort(np.asarray(authentic_scores, dtype=float).ravel())

    num_pb = len(playback_scores)
    num_au = len(authentic_asc)

    all_scores = np.sort(np.concatenate([playback_scores, authentic_asc]))
    thresholds = (all_scores[:-1] + all_scores[1:]) / 2

    miss_rates = np.searchsorted(playback_scores, thresholds, side="left") / num_pb
    false_accept_rates = (num_au - np.searchsorted(authentic_asc, thresholds, side="right")) / num_au

    diffs = np.abs(miss_rates - false_accept_rates)
    i = int(np.argmin(diffs))
    min_err = diffs[i]
    thrd = thresholds[i]

    # Special case: better off putting one whole set of scores below/above the threshold.
    if min_err > min(num_pb, num_au):
        if num_pb > num_au:
            thrd = playback_scores.min() * 0.999
        else:
            thrd = authentic_asc.max() * 0.999

    md = np.sum(playback_scores < thrd)
    fa = np.sum(authentic_asc >= thrd)

    err_rate = (md + fa) / (num_pb + num_au)
    return err_rate, thrd
