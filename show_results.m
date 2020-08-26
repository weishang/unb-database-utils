close all;  clc;

addpath('./common');

load results;

num_cap_sets = length(cap_sets);

all_pb_scores = [];
all_au_scores = []; 

for cur_set_ind = 1:num_cap_sets
    
    cur_cap_set = cap_sets{1};
    
    cur_pb_scores = cur_cap_set.playback_scores; 
    cur_au_scores = cur_cap_set.authentic_scores; 
    
    [eer, thrd] = get_eer(cur_pb_scores, cur_au_scores);
    
    all_pb_scores = [all_pb_scores; cur_pb_scores;]; 
    all_au_scores = [all_au_scores; cur_au_scores;]; 
    
    fprintf('Client %s Phrase %s: eer = %4.2f with thrd = %4.2f\n', cur_cap_set.client, cur_cap_set.phrase, eer, thrd);
end 

[eer, thrd] = get_eer(all_pb_scores, all_au_scores);

fprintf('Overall: eer = %4.2f with thrd = %4.2f\n', eer, thrd);
