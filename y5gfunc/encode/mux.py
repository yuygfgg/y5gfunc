from typing import Union, Optional
from pathlib import Path
import subprocess
from ..utils import resolve_path


def mux_mkv(
    output_path: Union[str, Path],
    videos: Optional[
        Union[
            list[dict[str, Union[str, Path, bool]]], dict[str, Union[str, Path, bool]]
        ]
    ] = None,
    audios: Optional[
        Union[
            list[dict[str, Union[str, Path, bool]]], dict[str, Union[str, Path, bool]]
        ]
    ] = None,
    subtitles: Optional[
        Union[
            list[dict[str, Union[str, Path, bool]]], dict[str, Union[str, Path, bool]]
        ]
    ] = None,
    fonts_dir: Optional[Union[str, Path]] = None,
    chapters: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Muxes video, audio, subtitle tracks, chapters, and fonts into an MKV file.

    Uses mkvmerge for muxing tracks and chapters, and mkvpropedit for attaching fonts.

    Args:
        output_path: Path for the output MKV file.
        videos: Video track(s). Can be a single dict or a list of dicts.
        audios: Audio track(s). Can be a single dict or a list of dicts.
        subtitles: Subtitle track(s). Can be a single dict or a list of dicts.
        fonts_dir: Optional directory containing TTF/OTF fonts to attach.
        chapters: Optional path to an OGM chapter file.

    Track dictionary structure:
        {
            "path": str | Path,       # Path to the track file (Required)
            "language": str,          # Language code
            "track_name": str,        # Name for the track
            "default": bool,          # True to set the default flag
            "comment": bool,          # True to set the commentary flag
            "timecode": str | Path    # Path to a timecode file
        }
        The first track of each type (video, audio, subtitle) will be marked as default unless explicitly set otherwise via the "default" key.

    Returns:
        Path to the created MKV file.

    Raises:
        ValueError: If no video, audio, subtitle, chapter, or font inputs are provided.
        FileNotFoundError: If an input file (track, chapter) is not found.
        RuntimeError: If mkvmerge or mkvpropedit encounters an error during execution.
    """
    output_path = resolve_path(output_path)
    if fonts_dir:
        fonts_dir = resolve_path(fonts_dir)
    if chapters:
        chapters = resolve_path(chapters)

    if not any((videos, audios, subtitles, chapters, fonts_dir)):
        raise ValueError(
            "mux_mkv: At least one input (videos, audios, subtitles, chapters, fonts_dir) must be provided."
        )

    def _normalize_inputs(inputs):
        if isinstance(inputs, dict):
            return [inputs]
        return inputs or []

    videos = _normalize_inputs(videos)
    audios = _normalize_inputs(audios)
    subtitles = _normalize_inputs(subtitles)

    for track_list in (videos, audios, subtitles):
        for track in track_list:
            track["path"] = resolve_path(track["path"])  # type: ignore

    all_files = [track["path"] for track in videos + audios + subtitles] + (
        [chapters] if chapters else []
    )
    for file in all_files:
        if not file.exists():  # type: ignore
            raise FileNotFoundError(f"mux_mkv: Required file not found: {file}")

    mkvmerge_cmd = ["mkvmerge", "-o", str(output_path)]

    def _process_tracks(tracks) -> None:
        first_default_set = False
        for i, track in enumerate(tracks):
            if "language" in track:
                mkvmerge_cmd.extend(["--language", f"0:{track['language']}"])
            if "track_name" in track:
                mkvmerge_cmd.extend(["--track-name", f"0:{track['track_name']}"])
            if "comment" in track:
                mkvmerge_cmd.extend(["--commentary-flag", "0:yes"])
            if "timecode" in track:
                mkvmerge_cmd.extend(["--timestamps", f"0:{str(track['timecode'])}"])

            if track.get("default") is True:
                mkvmerge_cmd.extend(["--default-track", "0:yes"])
                first_default_set = True
            elif track.get("default") is False:
                mkvmerge_cmd.extend(["--default-track", "0:no"])
            elif not first_default_set and i == 0:
                mkvmerge_cmd.extend(["--default-track", "0:yes"])
                first_default_set = True
            else:
                mkvmerge_cmd.extend(["--default-track", "0:no"])

            mkvmerge_cmd.append(str(track["path"]))

    _process_tracks(videos)
    _process_tracks(audios)
    _process_tracks(subtitles)

    if chapters:
        mkvmerge_cmd.extend(["--chapters", str(chapters)])

    result = subprocess.run(mkvmerge_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mux_mkv: Error executing mkvmerge:\n{result.stdout}")

    if fonts_dir and fonts_dir.exists():
        for font_ext in ["ttf", "otf"]:
            for font_file in fonts_dir.glob(f"*.{font_ext}"):
                font_cmd = [
                    "mkvpropedit",
                    str(output_path),
                    "--attachment-mime-type",
                    f"font/{font_ext}",
                    "--add-attachment",
                    str(font_file),
                ]
                font_result = subprocess.run(font_cmd, capture_output=True, text=True)
                if font_result.returncode != 0:
                    raise RuntimeError(
                        f"mux_mkv: Error adding font {font_file}:\n{font_result.stderr}"
                    )

    return output_path
