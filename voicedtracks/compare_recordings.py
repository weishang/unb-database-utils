"""Visually compare two recordings' VoicedTracks feature sets at the
frame-shift alignment that yields the highest similarity score -- matching
Figs. 3-4 of "Detection of speech playback attacks using robust harmonic
trajectories" (red circles vs. black dots, one feature set shifted onto the
other's frame axis).
"""
import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

# Lets this script (and its re-spawned worker processes, which invoke it via
# __file__) resolve the repo-root-level common/ and config.py regardless of
# how it's launched, since it lives one directory down in voicedtracks/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from check_recording import background_interpreter, disable_power_throttling, force_to_front
from common.recording_lookup import describe_recording, resolve_wav_path
from voiced_tracks import extract_voiced_tracks

_PLOT_WORKER_FLAG = "--plot-worker"


def find_best_alignment(feature1, feature2):
    """Same normalized cross-correlation as
    voicedtracks/calculate_similarity_score.py, but also returns the
    frame-shift that achieves the best score: shifting feature2's frame
    indices by `shift` (i.e. frame + shift) overlays it onto feature1's
    frame axis at the alignment the score was computed from."""
    norm1 = np.sqrt(feature1.sum())
    norm2 = np.sqrt(feature2.sum())
    if norm1 == 0 or norm2 == 0:
        return 0.0, 0

    f1 = feature1.astype(np.int32)
    f2 = feature2.astype(np.int32)

    n_freq = f1.shape[0]
    len2 = f2.shape[1]
    r = np.zeros(f1.shape[1] + f2.shape[1] - 1)
    for i in range(n_freq):
        r += np.correlate(f1[i], f2[i], mode="full")

    best_idx = int(r.argmax())
    shift = best_idx - (len2 - 1)
    score = float(r[best_idx] / (norm1 * norm2))
    return score, shift


def load_vt(wav_path):
    data, samplerate = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return extract_voiced_tracks(data, samplerate)


def build_figure(label1, vt1, label2, vt2, score, shift):
    import matplotlib.pyplot as plt

    bins1, frames1 = np.nonzero(vt1["matrix"])
    bins2, frames2 = np.nonzero(vt2["matrix"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(frames1, vt1["freqs"][bins1],
               s=25, facecolors="none", edgecolors="red", marker="o", label=label1)
    ax.scatter(frames2 + shift, vt2["freqs"][bins2],
               s=8, color="black", marker=".", label=label2)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"score = {score:.4f}   (shift = {shift:+d} frames)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def compute(recording1, recording2):
    wav1, wav2 = resolve_wav_path(recording1), resolve_wav_path(recording2)
    for r, wav in [(recording1, wav1), (recording2, wav2)]:
        if not wav.exists():
            raise SystemExit(f"No such file: {wav}")

    label1, label2 = describe_recording(recording1), describe_recording(recording2)
    print(label1)
    print(label2)

    vt1, vt2 = load_vt(wav1), load_vt(wav2)
    score, shift = find_best_alignment(vt1["matrix"], vt2["matrix"])
    print(f"score = {score:.4f}, best shift = {shift:+d} frames (of recording 2 onto recording 1)")

    return label1, vt1, label2, vt2, score, shift


def run_plot_worker(recording1, recording2):
    """Build the figure and show it (blocks THIS detached process only)."""
    disable_power_throttling()
    fig = build_figure(*compute(recording1, recording2))
    force_to_front(fig)

    import matplotlib.pyplot as plt
    plt.show()


def run_save(recording1, recording2, save_path):
    import matplotlib
    matplotlib.use("Agg")  # no display needed when just saving to a file

    fig = build_figure(*compute(recording1, recording2))
    fig.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def spawn_detached(recording1, recording2):
    cmd = [background_interpreter(), __file__, recording1, recording2, _PLOT_WORKER_FLAG]

    if sys.platform == "win32":
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Belt-and-suspenders: allow the child to steal foreground focus if it needs to.
        ctypes.windll.user32.AllowSetForegroundWindow(proc.pid)
    else:
        subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay two recordings' VoicedTracks feature sets at their best-scoring alignment.")
    parser.add_argument("recording1", help="rec_id (e.g. 42) or a .wav filename/path")
    parser.add_argument("recording2", help="rec_id (e.g. 43) or a .wav filename/path")
    parser.add_argument("--save", help="save the plot to this path instead of showing a window")
    parser.add_argument(_PLOT_WORKER_FLAG, dest="plot_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.plot_worker:
        run_plot_worker(args.recording1, args.recording2)
        return

    if args.save:
        run_save(args.recording1, args.recording2, args.save)
        return

    # Showing a window: run plotting in its own detached background process
    # so this command returns immediately instead of blocking the terminal.
    spawn_detached(args.recording1, args.recording2)
    print(f"Opened comparison plot window for {describe_recording(args.recording1)} vs "
          f"{describe_recording(args.recording2)}.")


if __name__ == "__main__":
    main()
