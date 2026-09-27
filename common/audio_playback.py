import sys

import numpy as np
import sounddevice as sd


def resample(data, orig_sr, target_sr):
    """Linearly resample audio to target_sr (good enough for playback, not analysis)."""
    if orig_sr == target_sr:
        return data

    duration = len(data) / orig_sr
    orig_times = np.linspace(0, duration, len(data), endpoint=False)
    target_times = np.linspace(0, duration, int(round(duration * target_sr)), endpoint=False)

    if data.ndim == 1:
        return np.interp(target_times, orig_times, data)
    return np.column_stack([np.interp(target_times, orig_times, data[:, ch]) for ch in range(data.shape[1])])


def default_output_device():
    """The OS's actual current default output device.

    On Windows, sounddevice's overall default can be pinned to a stale MME
    device captured at process start. WASAPI queries the live Windows default,
    so prefer it there; other platforms just use sounddevice's own default.
    """
    if sys.platform != "win32":
        return None

    for api in sd.query_hostapis():
        if api["name"] == "Windows WASAPI" and api["default_output_device"] >= 0:
            return api["default_output_device"]

    return None


def play(data, samplerate, wait=True):
    """Play audio through the OS default output, resampling if the device requires it."""
    device = default_output_device()
    play_rate = samplerate

    if device is not None:
        play_rate = int(sd.query_devices(device)["default_samplerate"])
        data = resample(data, samplerate, play_rate)

    sd.play(data, play_rate, device=device)
    if wait:
        sd.wait()
