import numpy as np


def calculate_similarity_score(feature1, feature2) -> float:
    """VT similarity score s_AB, per Shang & Stevenson (2020), "Detection of
    speech playback attacks using robust harmonic trajectories," eq. (1)-(2):
    cross-correlate the two binary VoicedTracks matrices over time-shift only
    (frequency bins need no alignment -- both come from the same fixed
    512-point FFT), take the best shift, and normalize by the matrices'
    Frobenius norms so the score lies in [0, 1] (1 only for an exact,
    time-shifted copy of the other).
    """
    norm1 = np.sqrt(feature1.sum())
    norm2 = np.sqrt(feature2.sum())
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # np.correlate on bool arrays does logical AND+OR instead of real integer
    # arithmetic, silently losing the overlap count -- cast to a real integer
    # dtype first (features are stored as bool on disk for compactness).
    feature1 = feature1.astype(np.int32)
    feature2 = feature2.astype(np.int32)

    n_freq = feature1.shape[0]
    r = np.zeros(feature1.shape[1] + feature2.shape[1] - 1)
    for i in range(n_freq):
        r += np.correlate(feature1[i], feature2[i], mode="full")

    return float(r.max() / (norm1 * norm2))
