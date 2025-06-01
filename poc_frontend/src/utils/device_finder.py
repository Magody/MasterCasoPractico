# src/utils/device_finder.py
import sounddevice as sd
from typing import Optional, Literal, Union

ChannelKind = Literal["input", "output"]

def find_device(
    substr: str,
    kind: ChannelKind = "output",
    raise_on_missing: bool = False
) -> Optional[int]:
    """
    Search sounddevice.query_devices() for the first device whose name
    contains `substr` (case-insensitive) and that has at least one
    {kind} channel.

    Parameters
    ----------
    substr : str
        Substring to match against device["name"].
    kind : "input" | "output"
        Whether to look for max_input_channels or max_output_channels.
    raise_on_missing : bool
        If True, raise RuntimeError when no matching device is found.
        If False, return None.

    Returns
    -------
    idx : Optional[int]
        The device index, or None if not found (unless raise_on_missing).
    """
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    s_lower = substr.lower()

    for idx, dev in enumerate(sd.query_devices()):
        name = dev.get("name", "").lower()
        if dev.get(key, 0) > 0 and s_lower in name:
            return idx

    if raise_on_missing:
        raise RuntimeError(f"No {kind} device matching {substr!r} found")
    return None


def find_device_audio_player(pa, substr: str):
    """
    Return the first PyAudio device index
    whose name contains `substr` (case-insensitive).
    """
    s = substr.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) and s in info["name"].lower():
            return i
    return None