close all;  clc;

addpath('../libs/DETware');

features = {'scores-check', 'scores-512'};
figure(1);
title ('PAD Performance Comparison');
hold on;

% setting up tick marks
Pmiss_min = 0.002;
Pmiss_max = 0.50;
Pfa_min = 0.002;
Pfa_max = 0.50;
Set_DET_limits(Pmiss_min,Pmiss_max,Pfa_min,Pfa_max);


% setting up DCF detection cost function
C_miss = 1;
C_fa = 1;
P_target = 0.5;
Set_DCF(C_miss,C_fa,P_target);

linetypes = ["-k", ":b"];


for i = 1:length(features)
    load([features{i}, '.mat']);
    
    % det curves for each CaP set
    
    figure(1); 
    cur_cap_set = 1; 
    for cur_client = 1:4
        for cur_phr = 1:3

            subplot(4, 3, cur_cap_set); hold on; 
            plot([-100 100],[-100 100], 'k:');
            temp_pb_scores = pb_scores(cur_client, cur_phr, :);           
            sub_pb_scores = reshape(temp_pb_scores, 9, 30);
            sub_au_scores = au_scores(cur_client, cur_phr, :);
            
            sub_pb_scores = sub_pb_scores(:);
            sub_au_scores = sub_au_scores(:);
            
            [P_miss,P_fa] = Compute_DET (sub_pb_scores, sub_au_scores);
            Plot_DET (P_miss, P_fa, '-k');
            [x,y] = get_ths(sub_pb_scores, sub_au_scores, 'EER');
            [md, mdr, fa, far, nERR, rERR] = get_err(sub_pb_scores, sub_au_scores, x);
            format long;
            x;
            format short;
            fprintf('featrue: %s, mdr: %4.2f%%, far: %4.2f%%, EER: %4.2f%% \n', features{i}, mdr*100, far*100, rERR*100);
            cur_cap_set = cur_cap_set + 1;
        end
    end
    
    figure(2); hold on; 
    pb_scores = reshape(pb_scores, 4,3, 9, 30); 
    pb_scores = pb_scores(:, :, 4:9, :);
    [P_miss,P_fa] = Compute_DET (pb_scores(:), au_scores(:));
    Plot_DET (P_miss, P_fa, linetypes(i));
    [x,y] = get_ths(pb_scores(:), au_scores(:), 'EER');
    [md, mdr, fa, far, nERR, rERR] = get_err(pb_scores(:), au_scores(:), x);
    

    fprintf('featrue: %s, mdr: %4.2f%%, far: %4.2f%%, EER: %4.2f%% \n', features{i}, mdr*100, far*100, rERR*100);
end

% legend({'VT', 'peakmap', 'landmarks',  'B01', 'B02'}, 'location', 'northeast', 'autoupdate', 'off');
legend(features, 'location', 'northeast', 'autoupdate', 'off');


ylabel('Missed Detection Rate (in %)');
xlabel('False Alarm Rate (in %)');
