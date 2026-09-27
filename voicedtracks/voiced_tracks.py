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
MIN_SPEECH_RUN = 10
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
    spectra = np.fft.rfft(frames * window, n=NFFT, axis=1)
    return np.abs(spectra)[:, :N_BINS]  # drop the Nyquist bin, per report


def _select_frame_peaks(magnitude):
    """Stage 1: return selected peaks for one frame as a list of (bin, mag),
    sorted by descending magnitude."""
    interior = np.arange(1, N_BINS - 1)
    is_local_max = (magnitude[interior] > magnitude[interior - 1]) & (magnitude[interior] > magnitude[interior + 1])
    candidates = interior[is_local_max]
    if len(candidates) == 0:
        return []

    order = np.argsort(magnitude[candidates])[::-1]
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

    return selected


def _compute_lsm(peaks):
    if not peaks:
        return None
    top = sorted((m for _, m in peaks), reverse=True)[:LSM_TOP_N]
    return float(np.log(sum(top)))


def _speech_non_speech_threshold(lsm_values):
    n = len(lsm_values)
    n_bins = int(np.ceil(np.sqrt(n)))
    hist, edges = np.histogram(lsm_values, bins=n_bins)

    lsm_sorted = np.sort(lsm_values)

    def cdf(value):
        return np.searchsorted(lsm_sorted, value, side="right") / n

    valleys = []
    for i in range(1, n_bins - 1):
        if hist[i] < hist[i - 1] and hist[i] <= hist[i + 1]:
            valleys.append(edges[i])

    candidates = [(v, cdf(v)) for v in valleys]
    in_range = [v for v, c in candidates if 0.2 <= c <= 0.7]
    if in_range:
        return min(in_range)

    below = [(v, c) for v, c in candidates if c < 0.2]
    if below:
        return max(below, key=lambda vc: vc[1])[0]

    return None  # no candidate has cdf < 0.7: all frames are speech


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


def _bin_edges_hz(b, delta_f):
    return (b - 0.5) * delta_f, (b + 0.5) * delta_f


def _bin_from_freq(f, delta_f):
    return int(np.clip(round(f / delta_f), 0, N_BINS - 1))


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

    finished_tracks.extend(open_tracks)
    return finished_tracks


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
    tracks = _build_tracks(frame_peaks, speech_mask, delta_f)

    matrix = np.zeros((N_BINS, n_frames), dtype=bool)
    for track in tracks:
        if len(track["points"]) < MIN_TRACK_LENGTH:
            continue
        avg_freq = np.mean([b for _, b, _ in track["points"]]) * delta_f
        if not (TRACK_FREQ_MIN <= avg_freq <= TRACK_FREQ_MAX):
            continue
        for frame_idx, b, _ in track["points"]:
            matrix[b, frame_idx] = 1

    frame_times = (np.arange(n_frames) * HOP_SIZE + FRAME_SIZE / 2) / samplerate
    freqs = np.arange(N_BINS) * delta_f

    return {
        "frame_times": frame_times, "freqs": freqs, "matrix": matrix,
        "speech_mask": speech_mask, "magnitude": magnitude,
    }
