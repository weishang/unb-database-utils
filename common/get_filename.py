"""Get filename given the recording id."""


def get_filename(rec_id: int) -> str:
    return f"{int(rec_id):05d}"
