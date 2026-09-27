# unb-database-utils

A collection of utility functions for playback attack detection performance evaluation using the UNB Database, i.e., Audio recordings of genuine and replayed speech at both ends of a telecommunication channel, which can be downloaded .

Two equivalent implementations are provided: the original MATLAB scripts, and a Python port (no MATLAB/Octave license needed).

## Python

1. Clone this repo
2. Download the UNB database from [here](https://data.mendeley.com/datasets/5t56sjbgf6/1) and unzip it directly into the repo root — this creates `metadata.csv` and a `data/` folder of recordings next to the scripts. `config.py` already points at this layout by default, so no editing needed unless you want the database somewhere else.
3. `pip install -r requirements.txt`
4. Replace `extract_feature` in `local/extract_feature.py` and `calculate_similarity_score` in `local/calculate_similarity_score.py` with your own implementations (or leave them as-is to use the reference implementation described below, which they forward to by default).
5. Run `python extract_features.py` to generate the feature files for all recordings. Cached under `FEATURE_FOLDER`: rerunning skips any recording whose feature file already exists, unless `local/extract_feature.py` or the reference implementation it forwards to has changed since, in which case everything is recomputed.
6. Run `python generate_scores.py` to generate the playback and authentic scores.
7. Run `python show_results.py` to generate the equal error rates.

### Project layout: common / voicedtracks / local

- **`common/`** — generic, algorithm-agnostic utilities shared by every script: recording lookup, EER computation, the feature/results caches, etc. Nothing here is specific to any one feature-extraction approach.
- **`voicedtracks/`** — the shipped reference implementation of Shang & Stevenson's playback attack detector, described below. Meant to be read as a working example, not edited.
- **`local/`** — the swap-in-your-own-approach slot. `extract_features.py` and `compute_scores.py` always import `extract_feature`/`calculate_similarity_score` from here; by default these two files just forward to `voicedtracks/`. To use a different feature set or similarity measure, replace the body of `local/extract_feature.py` and/or `local/calculate_similarity_score.py` with your own code — nothing else needs to change.

### Reference implementation: VoicedTracks + cross-correlation similarity

- **Feature extraction** (`voicedtracks/voiced_tracks.py`): the three-stage VoicedTracks process from `tech_report.pdf` — frame-level spectral peak selection, speech/non-speech frame detection, and frame-to-frame harmonic track linking — producing a binary time-frequency matrix per recording.
- **Similarity scoring** (`voicedtracks/calculate_similarity_score.py`): the normalized cross-correlation from `2020_detection of speech playback attacks using robust harmonic.pdf` (eq. 1-2) — the two VT matrices are cross-correlated over time-shift only (frequency bins need no alignment), the best-shift score is normalized by both matrices' Frobenius norms, giving a value in [0, 1].
- **Storage**: features are stored in `FEATURE_FOLDER` as gzip-compressed pickles (`.pkl.gz`) of the `bool` VT matrix. Since a typical matrix is >99% zeros, this brings the whole feature set for all 4,680 recordings down to ~18MB (vs. ~2.6GB stored as dense `float64`, `np.zeros`'s default dtype). Note `np.correlate` treats `bool` arrays as logical AND/OR rather than integer arithmetic, silently losing the overlap count -- `calculate_similarity_score` casts to `int32` before correlating for exactly this reason, while keeping the compact `bool` dtype for storage.
- **Evaluation** (`generate_scores.py`): playback recordings made via the low-quality built-in DVR speaker (`pb_device == 'D'`, the report's "PR-dvr" set) are excluded from evaluation, per the report's own methodology — this audio is poor enough that it's unlikely to pass a speaker verification system regardless, and including it skews the EER substantially (see below).

Run against the full UNB database (4,680 recordings), this reference implementation gets an overall EER of **3%** (client/passphrase-independent threshold), close to the report's own **2.26%**. Including the excluded DVR recordings pushes the overall EER up to **10%** — a good illustration of why that exclusion matters, not just a cosmetic evaluation choice.

### Inspecting recordings

`python check_recording.py <rec_id | filename> [--play] [--save out.png]` — plot the waveform and spectrogram of a recording (e.g. `python check_recording.py 42` or `python check_recording.py 00042.wav`), optionally playing it through the speakers at the same time with `--play`.

By default this opens the plot in its own background window and returns your terminal immediately — run it again for as many recordings as you like, each gets its own window. Pass `--save out.png` to save to a file instead of opening a window.

The spectrogram includes an overlaid pitch (F0) contour, estimated per-frame via normalized autocorrelation (`estimate_pitch_track` in `check_recording.py`).

**Known limitation**: the contour is unreliable for low-pitched speakers (e.g. client A, whose true F0 is ~70-90 Hz) — it currently searches a fixed, generic 60-500 Hz range with one voicing threshold for everyone, which is wide enough to let octave errors (locking onto a harmonic instead of the fundamental) and noise-floor artifacts compete with the real pitch. This is worse for recordings transmitted over a telecommunication channel, which band-limits the spectrum (traditional phone bandpass is roughly 300-3400 Hz) and can filter out the fundamental itself, leaving only its harmonics for the autocorrelation to lock onto. A likely fix, to revisit later: make the search range (and maybe the voicing threshold) configurable per speaker/client instead of one-size-fits-all. A more robust alternative, also to revisit: derive pitch from the VoicedTracks harmonic spacing below instead (the average distance between adjacent harmonic trajectories), which only needs *some* harmonics present, not the fundamental itself.

A third panel plots the **VoicedTracks feature set** (`voicedtracks/voiced_tracks.py`), implementing the three-stage extraction process described in `tech_report.pdf` (Shang & Stevenson): frame-level spectral peak selection, speech/non-speech frame detection via an LSM histogram, and frame-to-frame harmonic track linking. It shows only the surviving robust-harmonic points, matching the report's Figure 3.1.

## MATLAB

1. Clone this repo
2. Download the UNB database from [here](https://data.mendeley.com/datasets/5t56sjbgf6/1).
3. Update `database_folder` and `feature_folder` in `load_configs.m`. 
4. Replace `extract_feature.m` and `calculate_similarity_score.m` in `./common` folder. 
5. Run `./extract_features.m` to generate the feature files for all recordings. 
6. Run `./generate_scores.m` to generate the playback and authentic scores.
7. Run `./show_results.m` to generate the equal error rates.
