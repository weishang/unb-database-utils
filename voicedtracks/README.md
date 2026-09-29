# voicedtracks

The shipped reference implementation of Shang & Stevenson's playback attack
detector: the VoicedTracks (VT) feature set, plus two interchangeable
similarity measures. Meant to be read as a working example of the
`local/extract_feature.py` / `local/calculate_similarity_score.py` contract
described in the repo root [README](../README.md), not edited in place — if
you want a different approach, that's what `local/` is for.

## Files

- **`voiced_tracks.py`** — the three-stage VT feature extraction process:
  frame-level spectral peak selection, speech/non-speech frame detection, and
  frame-to-frame harmonic track linking. Produces a binary time-frequency
  matrix per recording. See `tech_report.pdf` for the full derivation.
- **`extract_feature.py`** — thin wrapper matching the `local/` contract:
  reads a wav file and calls `voiced_tracks.extract_voiced_tracks`.
- **`calculate_similarity_score.py`** — the normalized cross-correlation
  similarity measure from `2020_detection of speech playback attacks using
  robust harmonic.pdf` (eq. 1-2), the paper's own primary method. The two VT
  matrices are cross-correlated over time-shift only (frequency bins need no
  alignment); the best-shift score is normalized by both matrices' Frobenius
  norms, giving a value in [0, 1].
- **`calculate_similarity_score_nw.py`** — an alternative similarity measure:
  a modified Needleman-Wunsch frame-by-frame alignment (from an unpublished,
  later revision of the original MATLAB code, `comparison_by_segments.m`).
  Unlike the cross-correlation measure's single global time-shift, this
  allows free realignment at silent (non-speech) frames while forcing strict
  frame-to-frame correspondence through voiced content — useful for
  channels (e.g. VoIP) that stretch or shrink pauses unpredictably. Not
  wired in as the default; swap it in by pointing
  `local/calculate_similarity_score.py` at
  `calculate_similarity_score_nw.calculate_similarity_score_nw` instead.
- **`check_recording.py`** — inspect a single recording (see below).
- **`compare_recordings.py`** — visually compare two recordings' feature sets
  (see below).

## Results

Run against the full UNB database (4,680 recordings), with the `pb_device ==
'D'` (low-quality built-in DVR speaker) recordings excluded from evaluation
per the report's own methodology:

| Similarity measure | overall EER (client/passphrase-independent threshold) |
|---|---|
| Cross-correlation (`calculate_similarity_score.py`) | **2.22%** |
| Modified Needleman-Wunsch (`calculate_similarity_score_nw.py`) | **1.84%** |

The paper itself reports 2.26% for the cross-correlation measure. Including
the excluded DVR recordings pushes the cross-correlation EER up to ~10% — a
good illustration of why that exclusion matters, not just a cosmetic
evaluation choice.

**Storage**: features are stored in `FEATURE_FOLDER` as gzip-compressed
pickles (`.pkl.gz`) of the `bool` VT matrix. Since a typical matrix is >99%
zeros, this brings the whole feature set for all 4,680 recordings down to
~18MB (vs. ~2.6GB stored as dense `float64`, `np.zeros`'s default dtype).
Note `np.correlate` treats `bool` arrays as logical AND/OR rather than
integer arithmetic, silently losing the overlap count -- both similarity
functions cast to `int32` before correlating for exactly this reason, while
keeping the compact `bool` dtype for storage.

## Inspecting recordings

`python voicedtracks/check_recording.py <rec_id | filename> [--play] [--save out.png]`
— plot the waveform and spectrogram of a recording (e.g. `python
voicedtracks/check_recording.py 42` or `python voicedtracks/check_recording.py
00042.wav`), optionally playing it through the speakers at the same time with
`--play`. The VoicedTracks feature set is overlaid directly on the
spectrogram as black dots, matching `tech_report.pdf`'s Figure 3.1.

By default this opens the plot in its own background window and returns your
terminal immediately — run it again for as many recordings as you like, each
gets its own window. Pass `--save out.png` to save to a file instead of
opening a window.

`python voicedtracks/compare_recordings.py <rec_id1> <rec_id2> [--save out.png]`
— overlay two recordings' VoicedTracks feature sets at the frame-shift
alignment that yields the highest cross-correlation similarity score,
matching Figs. 3-4 of the 2020 paper: red circles for the first recording,
black dots for the second (shifted onto the first's frame axis). Same
background-window/`--save` behavior as `check_recording.py`.
