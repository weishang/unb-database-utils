from pathlib import Path

import pandas as pd

from config import PROTOCOL_FILE, RECORDINGS_FOLDER

from .get_filename import get_filename


def resolve_wav_path(recording: str) -> Path:
    """Resolve a CLI argument (rec_id, bare filename, or path) to a wav file path."""
    path = Path(recording)
    if path.exists():
        return path

    filename = get_filename(int(recording)) if recording.isdigit() else recording.removesuffix(".wav")
    return RECORDINGS_FOLDER / f"{filename}.wav"


def lookup_metadata(rec_id: int) -> dict | None:
    protocol = pd.read_csv(PROTOCOL_FILE)
    row = protocol.loc[protocol["rec_id"] == rec_id]
    return row.iloc[0].to_dict() if len(row) else None


def describe_recording(recording: str) -> str:
    """Human-readable label for a recording: metadata row if it's a known rec_id, else the path."""
    if recording.isdigit():
        meta = lookup_metadata(int(recording))
        if meta:
            return (f"rec_id={meta['rec_id']} client={meta['client_id']} phrase={meta['phrase_id']} "
                     f"rec_type={meta['rec_type']} with_ir={meta['with_ir']}")
    return str(resolve_wav_path(recording))
