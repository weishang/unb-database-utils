clear; clc; close all;

load_configs; 

recordings = readtable(protocol_file);

clients = cell2mat(recordings.client_id);
unique_clients = unique(clients);

phrases = cell2mat(recordings.phrase_id); 
unique_phrases = unique(phrases);


num_clients = length(unique_clients);
num_phrases = length(unique_phrases);


cap_sets = cell(1, num_clients * num_phrases);

set_num = 0;

for cur_client_ind = 1:num_clients
    for cur_phrase_ind = 1:num_phrases
        
        cur_client = unique_clients(cur_client_ind);
        cur_phrase = unique_phrases(cur_phrase_ind);
        
        cap_set = struct; 
        
        cap_set.client = cur_client; 
        cap_set.phrase = cur_phrase; 
        
        cap_set.stored_recordings = get_recording_ids(recordings, cur_client,...
            cur_phrase, 'stored', '', '');
        cap_set.playback_recordings = get_recording_ids(recordings, cur_client,...
            cur_phrase, 'playback', '', '');
        cap_set.authentic_recordings = get_recording_ids(recordings, cur_client,...
            cur_phrase, 'authentic', '', '');
        
        cap_set.playback_scores = compute_scores(feature_folder, cap_set.playback_recordings, cap_set.stored_recordings); 
        cap_set.authentic_scores = compute_scores(feature_folder, cap_set.authentic_recordings, cap_set.stored_recordings); 
        
        
        set_num = set_num + 1; 
        cap_sets{set_num} = cap_set;
        
    end
    
end


save('results', 'cap_sets');
