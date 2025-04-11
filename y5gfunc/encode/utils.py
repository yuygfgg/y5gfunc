import subprocess
from pathlib import Path


def get_language_by_trackid(m2ts_path: Path, ffprobe_id: str) -> str:
    """
    Get the language code for a specific track ID within an M2TS file.

    Uses tsMuxeR to analyze the M2TS file and extract the language tag
    associated with the given track ID.

    Args:
        m2ts_path: Path to the M2TS file.
        ffprobe_id: The track ID. Can be a hexadecimal string (e.g., '0x1011') or an integer representing the 0-based index from ffprobe.

    Returns:
        The language code (e.g., "eng", "jpn") or "und" if not found or if tsMuxeR fails.
    """
    if str(ffprobe_id).startswith("0x"):
        track_id = int(ffprobe_id, 16)
    else:
        track_id = int(ffprobe_id) + 1

    try:
        tsmuxer_output = subprocess.check_output(["tsMuxeR", str(m2ts_path)], text=True)
    except subprocess.CalledProcessError:
        return "und"

    current_track = None
    track_langs = {}

    for line in tsmuxer_output.splitlines():
        line = line.strip()

        if line.startswith("Track ID:"):
            current_track = int(line.split(":")[1].strip())
        elif line.startswith("Stream lang:") and current_track is not None:
            lang = line.split(":")[1].strip()
            track_langs[current_track] = lang
            current_track = None

    return track_langs.get(track_id, "und") or "und"
