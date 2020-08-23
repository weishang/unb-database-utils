close all; clear; clc;

addpath("./common");

load_configs; 

recordings = readtable(protocol_file);

num_rec = size(recordings);

if ~exist(feature_folder, 'dir')
    mkdir(feature_folder);
end

rec_id = recordings(:,{'rec_id'});
for i = 1:num_rec
    
    filename = get_filename(recordings(i,{'rec_id'}).rec_id);
    feature_file = fullfile(feature_folder, '/', filename);
    wav_file = fullfile(recordings_folder, [filename, '.wav']);
    
    fprintf("Extract feature set for recording %s\n", wav_file);
    % call your feature extraction function, e.g., 
    feature = extract_feature(wav_file);
    save(feature_file, 'feature');
end
