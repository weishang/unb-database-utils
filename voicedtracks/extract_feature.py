import numpy as np
import soundfile as sf

from .voiced_tracks import extract_voiced_tracks


def extract_feature(filepath: str) -> np.ndarray:
    """The VT (VoicedTracks) feature set: a binary Nf x Nt matrix marking the
    surviving robust-harmonic track points of the recording."""
    signal, samplerate = sf.read(filepath)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    result = extract_voiced_tracks(signal, samplerate)
    return result["matrix"]
