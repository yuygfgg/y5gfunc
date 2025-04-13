from vstools import vs
from vstools import core
from .wobbly import load_and_process
from typing import Optional, Union
from pathlib import Path
from ..utils import resolve_path


def wobbly_source(
    wob_project_path: Union[str, Path],
    timecodes_v2_path: Optional[Union[str, Path]] = None,
) -> vs.VideoNode:
    """
    Loads a video from a wobbly .wob project file.

    Args:
        wob_project_path: Path to the .wob project file.
        timecodes_v2_path: Optional path to a V2 timecodes file.

    Returns:
        A VapourSynth VideoNode representing the processed video clip.
    """
    clip = load_and_process(
        wob_project_path, timecodes_v2_path, timecode_version="v2"
    ).std.SetFieldBased(False)
    return clip


def bestsource(
    file_path: Union[Path, str],
    track: int = 0,
    timecodes_v2_path: Optional[Union[Path, str]] = None,
    variableformat: int = -1,
    rff: bool = False,
) -> vs.VideoNode:
    """
    Loads a video source using bestsource (bs.VideoSource).

    Args:
        file_path: Path to the video file.
        track: Index of the video track to load.
        timecodes_v2_path: Optional path to a V2 timecodes file. If provided, it will be passed to `bs.VideoSource`.
        variableformat: `variableformat` parameter for `bs.VideoSource`. See bestsource documentation.
        rff: `rff` parameter for `bs.VideoSource` (Repeat First Field). See bestsource documentation.
    Returns:
        A VapourSynth VideoNode loaded by bestsource.
    """
    if timecodes_v2_path:
        return core.bs.VideoSource(
            str(file_path),
            track,
            variableformat,
            timecodes=str(timecodes_v2_path),
            rff=rff,
        )
    else:
        return core.bs.VideoSource(str(file_path), track, variableformat, rff=rff)


# TODO: auto matrix handle
def load_source(
    file_path: Union[Path, str],
    track: int = 0,
    matrix_s: str = "709",
    matrix_in_s: str = "709",
    timecodes_v2_path: Optional[Union[Path, str]] = None,
) -> vs.VideoNode:
    """
    Bestsource and Wobbly wrapper to load a video source.

    This function acts as a primary interface for loading video sources.
    It checks the file extension:
        - If it's a ".wob" file, it uses `_wobbly_source`.
        - Otherwise, it uses `_bestsource`, attempting to automatically detect if the source uses RFF (Repeat First Field) based on frame counts.

    After loading, it applies a color matrix conversion.

    Args:
        file_path: Path to the video file or .wob project file.
        track: Index of the video track to load. Ignored for .wob files.
        matrix_s: Target color matrix string (e.g., "709", "bt470bg").
        matrix_in_s: Input color matrix string (e.g., "709", "bt470bg").
        timecodes_v2_path: Optional path to a V2 timecodes file.

    Returns:
        A VapourSynth VideoNode representing the loaded and matrix-converted video clip.

    Raises:
        FileNotFoundError: If the resolved `file_path` does not exist.
        AssertionError: If a `.wob` file is specified but `track` is not 0.
    """
    file_path = resolve_path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"load_source: File {file_path} does not exist.")

    if file_path.suffix.lower() == ".wob":
        assert track == 0
        clip = wobbly_source(file_path, timecodes_v2_path)
    else:
        # modified from https://guides.vcb-s.com/basic-guide-10/#%E6%A3%80%E6%B5%8B%E6%98%AF%E5%90%A6%E4%B8%BA%E5%85%A8%E7%A8%8B-soft-pulldownpure-film
        a = bestsource(file_path, rff=False)
        b = bestsource(file_path, rff=True)
        rff = False if abs(b.num_frames * 0.8 - a.num_frames) < 1 else True

        clip = bestsource(file_path, track, timecodes_v2_path, rff=rff)

    return clip.resize2.Spline36(matrix_s=matrix_s, matrix_in_s=matrix_in_s)
