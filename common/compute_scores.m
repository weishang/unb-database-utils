function scores_per_recording = compute_scores(feature_folder, incoming_recordings, stored_recordings)

scores = zeros(1,length(incoming_recordings));

for i = 1:length(incoming_recordings)
    load(fullfile(feature_folder, get_filename(incoming_recordings(i))), 'feature');
    incoming_feature = feature; 
    scores_per_recording = zeros(1, length(stored_recordings));
    for j = 1:length(stored_recordings)
        load(fullfile(feature_folder, get_filename(stored_recordings(j))), 'feature');
        % replace the following with your own similarity measure function
        scores_per_recording(j) = compare_features(incoming_feature, feature);
    end
    scores(i) = max(scores_per_recording);
end