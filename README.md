# unb-database-utils

A collection of utility functions for playback attack detection performance evaluation using the UNB Database, i.e., Audio recordings of genuine and replayed speech at both ends of a telecommunication channel, which can be downloaded .

Two equivalent implementations are provided: the original MATLAB scripts, and a Python port (no MATLAB/Octave license needed).

## Python

1. Clone this repo
2. Download the UNB database from [here](https://data.mendeley.com/datasets/5t56sjbgf6/1) and unzip it directly into the repo root — this creates `metadata.csv` and a `data/` folder of recordings next to the scripts. `config.py` already points at this layout by default, so no editing needed unless you want the database somewhere else.
3. `pip install -r requirements.txt`
4. Replace `extract_feature` in `local/extract_feature.py` and `calculate_similarity_score` in `local/calculate_similarity_score.py` with your own implementations (or leave them as-is to use the shipped [VoicedTracks reference implementation](voicedtracks/README.md), which they forward to by default).
5. Run `python extract_features.py` to generate the feature files for all recordings. Cached under `FEATURE_FOLDER`: rerunning skips any recording whose feature file already exists, unless `local/extract_feature.py` or the reference implementation it forwards to has changed since, in which case everything is recomputed.
6. Run `python generate_scores.py` to generate the playback and authentic scores.
7. Run `python show_results.py` to generate the equal error rates.

### Project layout: common / voicedtracks / local

- **`common/`** — generic, algorithm-agnostic utilities shared by every script: recording lookup, EER computation, the feature/results caches, etc. Nothing here is specific to any one feature-extraction approach.
- **`voicedtracks/`** — the shipped reference implementation of Shang & Stevenson's playback attack detector: VoicedTracks feature extraction, two interchangeable similarity measures, and recording-inspection tools. See [voicedtracks/README.md](voicedtracks/README.md) for details, results, and usage. Meant to be read as a working example, not edited.
- **`local/`** — the swap-in-your-own-approach slot. `extract_features.py` and `compute_scores.py` always import `extract_feature`/`calculate_similarity_score` from here; by default these two files just forward to `voicedtracks/`. To use a different feature set or similarity measure, replace the body of `local/extract_feature.py` and/or `local/calculate_similarity_score.py` with your own code — nothing else needs to change.

## MATLAB

1. Clone this repo
2. Download the UNB database from [here](https://data.mendeley.com/datasets/5t56sjbgf6/1).
3. Update `database_folder` and `feature_folder` in `load_configs.m`. 
4. Replace `extract_feature.m` and `calculate_similarity_score.m` in `./common` folder. 
5. Run `./extract_features.m` to generate the feature files for all recordings. 
6. Run `./generate_scores.m` to generate the playback and authentic scores.
7. Run `./show_results.m` to generate the equal error rates.
