import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

# Lets this script (and its re-spawned worker processes, which invoke it via
# __file__) resolve the repo-root-level common/ and config.py regardless of
# how it's launched, since it lives one directory down in voicedtracks/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from common.audio_playback import play
from common.recording_lookup import describe_recording, resolve_wav_path


class _ProcessPowerThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def disable_power_throttling():
    """Opt this process out of Windows' automatic "Efficiency Mode" (EcoQoS) throttling.

    Windows deprioritizes CPU scheduling for background/unfocused GUI processes,
    which can starve a newly spawned detached window's paint/event loop (and a
    separate audio worker's playback callback) until *something* -- often just
    clicking anywhere on the desktop -- causes the system to reassess. A process
    can always exempt itself from this, no special permission required.
    """
    if sys.platform != "win32":
        return

    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS enum value

    state = _ProcessPowerThrottlingState(
        Version=1,
        ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask=0,  # 0 = do not throttle
    )
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.kernel32.SetProcessInformation(
        handle, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state)
    )

# Hidden flags: distinguish the actual background workers (plotting, audio)
# from the user-facing invocation that spawns them. Kept as separate detached
# processes so a heavy plot render never steals the GIL from audio playback.
_PLOT_WORKER_FLAG = "--plot-worker"
_AUDIO_WORKER_FLAG = "--audio-worker"


def load_recording(recording, wav_path):
    data, samplerate = sf.read(wav_path)
    label = describe_recording(recording)
    print(label)
    print(f"{wav_path} ({len(data) / samplerate:.2f}s @ {samplerate} Hz)")
    return data, samplerate, label


def build_figure(data, samplerate, label):
    plot_data = data.mean(axis=1) if data.ndim > 1 else data
    time = np.arange(len(plot_data)) / samplerate

    import matplotlib.pyplot as plt

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.suptitle(label)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(label)

    ax_wave.plot(time, plot_data, linewidth=0.5, color="#3a5ba0")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlim(time[0], time[-1])

    nfft = 512
    hop = nfft // 2
    ax_spec.specgram(plot_data, Fs=samplerate, NFFT=nfft, noverlap=hop, cmap="magma")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (s)")

    plot_voiced_tracks(ax_spec, plot_data, samplerate)

    fig.tight_layout()
    add_scroll_zoom(fig)
    return fig


def plot_voiced_tracks(ax, plot_data, samplerate):
    """Overlay the VoicedTracks feature set (Shang & Stevenson) as a scatter of
    surviving harmonic-track points on top of the spectrogram, matching
    tech_report.pdf's Figure 3.1."""
    from voiced_tracks import extract_voiced_tracks

    result = extract_voiced_tracks(plot_data, samplerate)
    bins_idx, frames_idx = np.nonzero(result["matrix"])
    if len(frames_idx) == 0:
        return

    ax.scatter(result["frame_times"][frames_idx], result["freqs"][bins_idx],
               s=6, color="black", marker="o", zorder=2.5)


def add_scroll_zoom(fig):
    """Zoom the axis under the cursor on mouse-wheel scroll (in addition to the
    toolbar's click-drag box zoom). Scrolling over either subplot zooms both in
    time, since they share the x-axis; the y-axis (amplitude / frequency) zooms
    independently per subplot."""
    def on_scroll(event):
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return

        scale = 1.2 if event.button == "down" else 1 / 1.2

        xlim = ax.get_xlim()
        x = event.xdata
        ax.set_xlim(x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale)

        if event.ydata is not None:
            ylim = ax.get_ylim()
            y = event.ydata
            ax.set_ylim(y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)


def force_to_front(fig):
    """Grab focus despite Windows blocking SetForegroundWindow for background/detached processes.

    Must run only after the window is actually mapped on screen, so this is
    scheduled via Tk's own event loop rather than called immediately -- doing
    it before the mainloop starts has no effect on Windows.
    """
    manager = fig.canvas.manager
    window = getattr(manager, "window", None)
    if window is None or not hasattr(window, "attributes"):
        return  # not a Tk-backed window (different backend); nothing we can do generically

    def grab_focus():
        window.attributes("-topmost", True)
        window.lift()
        window.focus_force()
        window.after(200, lambda: window.attributes("-topmost", False))

    window.after(100, grab_focus)


def run_plot_worker(recording, wav_path):
    """Build the figure and show it (blocks THIS detached process only)."""
    disable_power_throttling()
    data, samplerate, label = load_recording(recording, wav_path)
    fig = build_figure(data, samplerate, label)
    force_to_front(fig)

    import matplotlib.pyplot as plt
    plt.show()


def run_save(recording, wav_path, save_path):
    import matplotlib
    matplotlib.use("Agg")  # no display needed when just saving to a file

    data, samplerate, label = load_recording(recording, wav_path)
    fig = build_figure(data, samplerate, label)
    fig.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def run_audio_worker(recording, wav_path):
    """Just play the recording, in a process with no matplotlib in it at all."""
    disable_power_throttling()
    data, samplerate = sf.read(wav_path)
    play(data, samplerate)


def background_interpreter():
    """pythonw.exe if available: Python's own windowless interpreter, built exactly
    for launching a GUI child with no console -- unlike python.exe + DETACHED_PROCESS,
    it doesn't fight with terminals that host a pseudoconsole (VSCode's integrated
    terminal, Windows Terminal), which could leave the child's window stuck until the
    hosting terminal itself got an input event."""
    if sys.platform == "win32":
        pythonw = sys.executable.replace("python.exe", "pythonw.exe")
        if pythonw != sys.executable and os.path.exists(pythonw):
            return pythonw
    return sys.executable


def spawn_detached(recording, worker_flag):
    cmd = [background_interpreter(), __file__, recording, worker_flag]

    if sys.platform == "win32":
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Belt-and-suspenders: allow the child to steal foreground focus if it needs to.
        ctypes.windll.user32.AllowSetForegroundWindow(proc.pid)
    else:
        subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="Inspect a recording: plot its waveform/spectrogram, and optionally play it.")
    parser.add_argument("recording", help="rec_id (e.g. 42) or a .wav filename/path")
    parser.add_argument("--play", action="store_true", help="also play the recording through the speakers")
    parser.add_argument("--save", help="save the plot to this path instead of showing a window")
    parser.add_argument(_PLOT_WORKER_FLAG, dest="plot_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(_AUDIO_WORKER_FLAG, dest="audio_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    wav_path = resolve_wav_path(args.recording)
    if not wav_path.exists():
        raise SystemExit(f"No such file: {wav_path}")

    if args.audio_worker:
        run_audio_worker(args.recording, wav_path)
        return

    if args.plot_worker:
        run_plot_worker(args.recording, wav_path)
        return

    if args.save:
        run_save(args.recording, wav_path, args.save)
        if args.play:
            spawn_detached(args.recording, _AUDIO_WORKER_FLAG)
        return

    # Showing a window: run plotting (and playback) in their own detached
    # background processes so this command returns immediately instead of
    # blocking the terminal. Run it again for as many recordings as you want
    # -- each gets its own window. Audio runs in a separate process from the
    # plot so a heavy render never steals CPU/GIL time from the audio thread.
    spawn_detached(args.recording, _PLOT_WORKER_FLAG)
    if args.play:
        spawn_detached(args.recording, _AUDIO_WORKER_FLAG)

    print(f"Opened plot window for {describe_recording(args.recording)} (run again to open another).")


if __name__ == "__main__":
    main()
