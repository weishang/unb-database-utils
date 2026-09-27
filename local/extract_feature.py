# This is the swap-in-your-own-approach slot: extract_features.py always
# imports extract_feature from here. By default it forwards to the shipped
# VoicedTracks reference implementation (see ../voicedtracks/); replace the
# body of this file with your own feature extraction to use a different
# approach instead, without touching voicedtracks/ or any pipeline script.
from voicedtracks.extract_feature import extract_feature

__all__ = ["extract_feature"]
