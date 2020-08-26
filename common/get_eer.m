function [err_rate, thrd] = get_eer(playback_scores, authentic_scores)

if size(playback_scores, 2) == 1
    playback_scores = playback_scores';
end

if size(authentic_scores, 2) == 1
    authentic_scores = authentic_scores';
end


playback_scores = sort(playback_scores);
authentic_scores = sort(authentic_scores, 'descend');

all_scores = [playback_scores, authentic_scores];

% Which is then sorted.
all_scores = sort(all_scores);   % sort again.
% the length of the loop is set
len = length(all_scores)-1;

num_pb_scores = length(playback_scores); 
num_au_scores = length(authentic_scores);

pRs = zeros(1,len); 
aRs = pRs; 
ths = pRs;

for i = 1:len
    thrd = (all_scores(i) + all_scores(i+1)) / 2;
    ths(i) = thrd;
    
    pEs = length(find( playback_scores < thrd));
    eEs = length(find( authentic_scores > thrd));
    pRs(i) = pEs / num_pb_scores;
    aRs(i) = eEs / num_au_scores;
    
end
r = abs(pRs - aRs);
[min_err , I ] = min(r);
thrd = ths(I);
% Special case: where it is better off make one set of the scores
% below/above the threshold.
if ( min_err > min( [num_pb_scores,num_au_scores]))
    if num_pb_scores > num_au_scores % more higher scores than lower scores,
        thrd = min(playback_scores)*.999;
    else
        thrd = max(authentic_scores)*.999;
    end
end

md = length(find( playback_scores < thrd));
fa = length(find( authentic_scores >= thrd));

nERR = md + fa;
rERR = nERR / (length(playback_scores) + length(authentic_scores));

err_rate = rERR;
