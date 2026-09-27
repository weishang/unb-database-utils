# This is the swap-in-your-own-approach slot: compute_scores.py always
# imports calculate_similarity_score from here. By default it forwards to the
# shipped VoicedTracks reference implementation (see ../voicedtracks/);
# replace the body of this file with your own similarity measure to use a
# different approach instead, without touching voicedtracks/ or any pipeline
# script.
from voicedtracks.calculate_similarity_score import calculate_similarity_score

__all__ = ["calculate_similarity_score"]
