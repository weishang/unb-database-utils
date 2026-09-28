"""VoicedTracks feature extraction, per Shang & Stevenson, "Feature extraction
process for the VoicedTracks feature set" (UNB, v0.1.0).

Three stages: frame-level peak selection, speech frame detection, and voiced
track construction. See tech_report.pdf for the full derivation; parameter
values and equation numbers below refer to that document.

A few points the report describes at a conceptual level, resolved here the
most natural/standard way:
- FFT bin b's frequency interval is taken as [(b-0.5), (b+0.5)] * delta_f.
- Within one frame-pair's linking pass, a frame k+1 peak can be claimed by at
  most one frame k peak (first come, in descending-magnitude order, wins).
- Non-speech frames contribute no candidate peaks to stage 3, so a voiced
  track simply cannot bridge a non-speech gap; it is finalized on one side and
  a new track begins (if anything links) on the other.
"""
import numpy as np

FRAME_SIZE = 384       # 48 ms at 8000 Hz
HOP_SIZE = 80           # 10 ms at 8000 Hz
NFFT = 512
N_BINS = NFFT // 2      # 256; FFT bin indices 0..255 (Nyquist bin excluded, per report)
SIGNIFICANCE_RATIO = 0.02   # condition (i): 2% of (34 dB below) the frame's highest peak
MIN_PEAK_SEPARATION = 4     # condition (ii): bins
LOCAL_DOMINANCE_HALF_WIDTH = 2   # condition (iii): dominant within a centered 5-bin window
LSM_TOP_N = 5
SPEECH_LOW = 0.2    # a valid speech/non-speech valley's LSM-CDF must exceed this
SPEECH_HIGH = 0.7   # ...and stay under this
MIN_SPEECH_RUN = 8
MIN_TRACK_LENGTH = 5
TRACK_FREQ_MIN = 300
TRACK_FREQ_MAX = 3000
LINK_RATIO_INIT = 0.05      # initial rmin/rmax = 1 -/+ this, per frame-pair


def _frame_signal(data):
    n_frames = 1 + (len(data) - FRAME_SIZE) // HOP_SIZE
    if n_frames < 1:
        return np.empty((0, FRAME_SIZE))
    starts = np.arange(n_frames) * HOP_SIZE
    return np.stack([data[s:s + FRAME_SIZE] for s in starts])


def _frame_magnitudes(frames):
    window = np.hamming(FRAME_SIZE)
    window = window / window.sum()  # matches the reference's HW normalization
    windowed = frames * window
    # No DC/mean removal: find_stft.m (the reference's later, superseding
    # feature-extraction pipeline) windows and FFTs directly, with no mean
    # subtraction step at all.
    spectra = np.fft.rfft(windowed, n=NFFT, axis=1)
    return np.abs(spectra)[:, :N_BINS]  # drop the Nyquist bin, per report


def _select_frame_peaks(magnitude):
    """Stage 1: return selected peaks for one frame as a list of (bin, mag),
    sorted by descending magnitude."""
    # detect_peaks.m (the reference's later pipeline) circshifts the frame by
    # 2 bins and additionally skips shifted-index <= 5 before applying the
    # same (2, 253) range check used by the older get_voicedTracks.m -- net
    # effect, converted back to this module's true-bin/0-based convention:
    # candidates are restricted to bins [3, 249] instead of [2, 251].
    interior = np.arange(3, N_BINS - 6)
    is_local_max = (magnitude[interior] > magnitude[interior - 1]) & (magnitude[interior] > magnitude[interior + 1])
    candidates = interior[is_local_max]
    if len(candidates) == 0:
        return []

    # Stable descending sort (MATLAB's sort(...,'descend') preserves the
    # original -- here, ascending-bin -- order among tied magnitudes;
    # argsort()[::-1] would instead reverse it, and the default 'quicksort'
    # kind isn't stable to begin with).
    order = np.argsort(-magnitude[candidates], kind="stable")
    candidates = candidates[order]

    threshold = SIGNIFICANCE_RATIO * magnitude[candidates[0]]

    selected = []
    for b in candidates:
        m = magnitude[b]
        if m <= threshold:
            break  # descending order: nothing further can pass condition (i)

        if any(abs(int(b) - sb) < MIN_PEAK_SEPARATION for sb, _ in selected):
            continue  # condition (ii)

        lo = max(0, b - LOCAL_DOMINANCE_HALF_WIDTH)
        hi = min(N_BINS, b + LOCAL_DOMINANCE_HALF_WIDTH + 1)
        if m < magnitude[lo:hi].max():
            continue  # condition (iii)

        selected.append((int(b), float(m)))

    # The reference stores each peak's bin as (true local-max position) + 1
    # -- a leftover of its two-stage index derivation (find(...)+1 to get the
    # true position, then +1 again on the whole array right before storing).
    # Selection above (threshold/separation/dominance) uses the true
    # position; only the stored/returned bin carries the extra +1, same as
    # get_voicedTracks.m lines 125 and 160.
    return [(b + 1, m) for b, m in selected]


def _compute_lsm(peaks):
    if not peaks:
        return None
    top = sorted((m for _, m in peaks), reverse=True)[:LSM_TOP_N]
    return float(np.log(sum(top)))


def _speech_non_speech_threshold(lsm_values):
    n = len(lsm_values)
    n_bins = int(_matlab_round(np.sqrt(n)))
    hist, edges = np.histogram(lsm_values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Tie-break with a small increasing ramp before the sign-of-diff-of-diff
    # valley scan, so that flat/plateau runs in the (integer) histogram
    # counts resolve to a single valley at the start of the lowest plateau,
    # matching the reference's `N + (0:(num_bins-1))/num_bins` trick.
    ramped = hist + np.arange(n_bins) / n_bins
    d = np.diff(np.sign(np.diff(ramped)))
    valley_idx = np.where(d == 2)[0] + 1  # interior bins only

    if len(valley_idx) == 0:
        return None  # no valley found: all frames are speech

    cdf = np.cumsum(hist) / n

    in_range = [i for i in valley_idx if SPEECH_LOW < cdf[i] < SPEECH_HIGH]
    if in_range:
        return centers[in_range[0]]

    below = [i for i in valley_idx if cdf[i] < SPEECH_LOW]
    if below:
        return centers[below[-1]]

    return None  # no candidate has cdf < SPEECH_HIGH: all frames are speech


def _detect_speech_frames(frame_peaks):
    n_frames = len(frame_peaks)
    lsm = [_compute_lsm(peaks) for peaks in frame_peaks]
    lsm_values = np.array([v for v in lsm if v is not None])

    speech = np.zeros(n_frames, dtype=bool)
    if len(lsm_values) > 0:
        threshold = _speech_non_speech_threshold(lsm_values)
        for i, v in enumerate(lsm):
            if v is None:
                continue
            speech[i] = True if threshold is None else v > threshold

    # Refinement: keep only runs of >= MIN_SPEECH_RUN consecutive speech frames.
    refined = np.zeros(n_frames, dtype=bool)
    run_start = None
    for i in range(n_frames + 1):
        is_speech = speech[i] if i < n_frames else False
        if is_speech and run_start is None:
            run_start = i
        elif not is_speech and run_start is not None:
            if i - run_start >= MIN_SPEECH_RUN:
                refined[run_start:i] = True
            run_start = None

    return refined


def _matlab_round(x):
    """MATLAB's round() is half-away-from-zero; Python's builtin round() and
    np.round() are both half-to-even, which disagrees at exact .5 ties."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _bin_edges_hz(b, delta_f):
    return (b - 0.5) * delta_f, (b + 0.5) * delta_f


def _bin_from_freq(f, delta_f):
    # No clamping to [0, N_BINS-1]: the reference computes this as a plain
    # numeric bound for a `>=`/`<=` comparison against real peak bins, never
    # as an array index, so it's never clipped either.
    return int(_matlab_round(f / delta_f))


def _build_tracks(frame_peaks, speech_mask, delta_f):
    n_frames = len(frame_peaks)
    # peaks-by-bin lookup per frame, restricted to speech frames
    peaks_by_frame = []
    for i in range(n_frames):
        if speech_mask[i]:
            peaks_by_frame.append({b: m for b, m in frame_peaks[i]})
        else:
            peaks_by_frame.append({})

    open_tracks = []   # list of dicts: {"points": [(frame_idx, bin, mag), ...], "last_bin": b}
    finished_tracks = []

    # frame 0: every peak starts an open track
    for b, m in sorted(peaks_by_frame[0].items(), key=lambda bm: -bm[1]):
        open_tracks.append({"points": [(0, b, m)], "last_bin": b})

    for k in range(n_frames - 1):
        next_peaks = peaks_by_frame[k + 1]
        claimed = set()

        rmin, rmax = 1 - LINK_RATIO_INIT, 1 + LINK_RATIO_INIT
        # process open tracks (i.e. frame k's peaks) in descending magnitude order
        still_open = []
        tracks_this_frame = sorted(open_tracks, key=lambda t: -t["points"][-1][2])

        stopped = False
        for track in tracks_this_frame:
            if stopped:
                finished_tracks.append(track)
                continue

            bk = track["last_bin"]
            fmin_k, fmax_k = _bin_edges_hz(bk, delta_f)

            blower = _bin_from_freq(fmin_k * rmin, delta_f)
            bupper = _bin_from_freq(fmax_k * rmax, delta_f)
            if blower > bupper:
                blower, bupper = bupper, blower

            candidates = [b for b in next_peaks if blower <= b <= bupper and b not in claimed]
            if not candidates:
                finished_tracks.append(track)
                stopped = True  # no further (lower-magnitude) peaks in frame k are linked
                continue

            center = (blower + bupper) / 2
            b_next = min(candidates, key=lambda b: abs(b - center))
            claimed.add(b_next)
            m_next = next_peaks[b_next]
            track["points"].append((k + 1, b_next, m_next))
            track["last_bin"] = b_next
            still_open.append(track)

            fmin_next, fmax_next = _bin_edges_hz(b_next, delta_f)
            r_lo = fmin_next / fmax_k
            r_hi = fmax_next / fmin_k
            rmin = max(rmin, r_lo)
            rmax = min(rmax, r_hi)

        open_tracks = still_open

        # unclaimed peaks in frame k+1 start new tracks
        for b, m in sorted(next_peaks.items(), key=lambda bm: -bm[1]):
            if b not in claimed:
                open_tracks.append({"points": [(k + 1, b, m)], "last_bin": b})

    return finished_tracks, open_tracks


def extract_voiced_tracks(data, samplerate):
    """Returns a dict with:
    - frame_times: (Nt,) frame center times in seconds
    - freqs: (N_BINS,) frequency of each FFT bin in Hz
    - matrix: (N_BINS, Nt) binary VoicedTracks feature matrix
    - speech_mask: (Nt,) bool, per-frame speech/non-speech label
    - magnitude: (Nt, N_BINS) per-frame FFT magnitude (for overlay plotting)
    """
    delta_f = samplerate / NFFT
    frames = _frame_signal(data)
    n_frames = len(frames)

    if n_frames == 0:
        return {
            "frame_times": np.array([]), "freqs": np.arange(N_BINS) * delta_f,
            "matrix": np.zeros((N_BINS, 0), dtype=bool), "speech_mask": np.array([], dtype=bool),
            "magnitude": np.zeros((0, N_BINS)),
        }

    magnitude = _frame_magnitudes(frames)
    frame_peaks = [_select_frame_peaks(magnitude[i]) for i in range(n_frames)]
    speech_mask = _detect_speech_frames(frame_peaks)
    tracks, open_at_eof = _build_tracks(frame_peaks, speech_mask, delta_f)

    matrix = np.zeros((N_BINS, n_frames), dtype=bool)
    for track in tracks:
        if len(track["points"]) < MIN_TRACK_LENGTH:
            continue
        # build_voicedtracks.m (the reference's later pipeline) computes this
        # as (mean_bin - 1) * FS/FFTSIZE, not mean_bin * FS/FFTSIZE -- the "-1"
        # offsets the +1 bin-storage quirk specifically for this frequency
        # check, without touching the stored/output bin positions themselves.
        avg_freq = (np.mean([b for _, b, _ in track["points"]]) - 1) * delta_f
        if not (TRACK_FREQ_MIN <= avg_freq <= TRACK_FREQ_MAX):
            continue
        for frame_idx, b, _ in track["points"]:
            matrix[b, frame_idx] = 1

    # The reference's frame loop (both track construction and matrix output)
    # stops one frame short of the end (`for frm_num = 1:num_frms-1`), so a
    # track still open at EOF never reaches the length/frequency filter (only
    # applied at termination) and its point in the true last frame is never
    # written to the output matrix. Replicate both quirks: write these
    # tracks' earlier points unconditionally, dropping only the last-frame one.
    for track in open_at_eof:
        for frame_idx, b, _ in track["points"]:
            if frame_idx == n_frames - 1:
                continue
            matrix[b, frame_idx] = 1

    frame_times = (np.arange(n_frames) * HOP_SIZE + FRAME_SIZE / 2) / samplerate
    freqs = np.arange(N_BINS) * delta_f

    return {
        "frame_times": frame_times, "freqs": freqs, "matrix": matrix,
        "speech_mask": speech_mask, "magnitude": magnitude,
    }
