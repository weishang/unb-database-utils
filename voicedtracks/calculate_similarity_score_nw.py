"""Modified Needleman-Wunsch similarity score, per comparison_by_segments.m
(Shang's later/experimental MATLAB code, alongside build_voicedtracks.m).

Unlike calculate_similarity_score.py's single global cross-correlation shift,
this aligns the two VT matrices frame-by-frame with a Needleman-Wunsch-style
DP: a frame pair (i, j) is only allowed a free "gap" (skip a frame in one
sequence without penalty) when at least one of the two frames is silent (no
VT peaks at all); whenever both frames carry real content, the alignment
path is forced to advance diagonally. This lets the alignment absorb
variable-length pauses (e.g. from VoIP channel jitter) without a global
shift, while still requiring strict frame-to-frame correspondence through
voiced segments.
"""
import numpy as np


def calculate_similarity_score_nw(feature1, feature2) -> float:
    # feature1, feature2: (N_BINS, n_frames) boolean arrays (this project's
    # storage convention). The reference works in (frames, bins) orientation.
    pm1 = feature1.T.astype(np.int32)
    pm2 = feature2.T.astype(np.int32)

    n0, m0 = pm1.shape[0], pm2.shape[0]

    nnz1 = int(pm1.sum())
    nnz2 = int(pm2.sum())
    if nnz1 == 0 or nnz2 == 0:
        return 0.0

    # Pad with one trailing all-zero frame each, per the reference (forces a
    # free-gap boundary condition at the very end of both sequences).
    n, m = n0 + 1, m0 + 1
    pm1p = np.zeros((n, pm1.shape[1]), dtype=np.int32)
    pm1p[:n0] = pm1
    pm2p = np.zeros((m, pm2.shape[1]), dtype=np.int32)
    pm2p[:m0] = pm2

    ppf1 = pm1p.sum(axis=1)
    ppf2 = pm2p.sum(axis=1)
    gap1 = ppf1 == 0
    gap2 = ppf2 == 0

    # Frame-pair overlap matrix: o[i, j] = number of frequency bins where
    # both frame i (pm1) and frame j (pm2) have a VT point. A single matrix
    # multiply of the binary matrices does this for every pair at once.
    o = pm1p @ pm2p.T

    c = np.zeros((n, m))
    c[0, :] = o[0, :]
    c[:, 0] = o[:, 0]

    for i in range(1, n):
        gi = gap1[i]
        c_prev = c[i - 1]
        c_cur = c[i]
        o_cur = o[i]
        for j in range(1, m):
            if gi or gap2[j]:
                sd = c_prev[j - 1] + o_cur[j]
                sr = c_cur[j - 1]
                sc = c_prev[j]
                c_cur[j] = sd if sd >= sr and sd >= sc else (sc if sc >= sr else sr)
            else:
                c_cur[j] = c_prev[j - 1] + o_cur[j]

    return float(c[n - 1, m - 1] / np.sqrt(nnz1 * nnz2))
