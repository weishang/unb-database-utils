import pandas as pd


def get_recording_ids(protocol: pd.DataFrame, client_id="", phrase_id="",
                       rec_type="", channel_type="", pb_device=""):
    mask = pd.Series(True, index=protocol.index)

    if client_id:
        mask &= protocol["client_id"] == client_id

    if phrase_id:
        mask &= protocol["phrase_id"] == phrase_id

    if rec_type == "stored":
        mask &= protocol["with_ir"].astype(str) == "True"
    elif rec_type == "authentic":
        mask &= protocol["with_ir"].astype(str) == "False"
    elif rec_type == "playback":
        mask &= protocol["rec_type"] == "playback"
    # else: no filter on rec_type (matches the MATLAB `otherwise` branch)

    if channel_type:
        mask &= protocol["channel_type"] == channel_type

    if pb_device:
        mask &= protocol["pb_device"] == pb_device

    return protocol.loc[mask, "rec_id"].tolist()
