# unb-database-utils

A collection of utility functions for playback attack detection performance evaluation using the UNB Database, i.e., Audio recordings of genuine and replayed speech at both ends of a telecommunication channel, which can be downloaded .

## Steps 

1. Clone this repo
2. Download the UNB database from [here](https://data.mendeley.com/datasets/5t56sjbgf6/1).
3. Update `database_folder` and `feature_folder` in `load_configs.m`. 
4. Replace `extract_feature.m` and `calculate_similarity_score.m` in `./common` folder. 
5. Run `./extract_features.m` to generate the feature files for all recordings. 
6. Run `./generate_scores.m` to generate the playback and authentic scores.
7. Run `./show_results.m` to generate the equal error rates.
