from vstools import vs
from pathlib import Path
from typing import Union
from ..utils import PickFrames, resolve_path


def screen_shot(
    clip: vs.VideoNode,
    frames: Union[list[int], int],
    path: Union[Path, str],
    file_name: str,
    overwrite: bool = True,
) -> None:
    """
    Takes screenshots of specified frames from a VapourSynth clip.

    This function selects one or more frames from the input video clip, and saves each selected frame as a PNG image using the `fpng` writer.

    The output filenames are generated based on the `file_name` format string, where a format specifier (like %d) is replaced by the corresponding frame number.

    Args:
        clip: The input VapourSynth video node.
        frames: The frame number or list of frame numbers to capture. If an integer is provided, it's treated as a single-element list.
        path: The directory path where the screenshots will be saved. Can be a string or a pathlib.Path object.
        file_name: A format string for the output filename. Must contain a C-style format specifier that will be replaced by the frame number.
            Example: 'screenshot_%05d' will produce filenames like 'screenshot_00100.png' for frame 100.
        overwrite: overwrite parameter for `fpng.Write`.

    Returns:
        None
    """
    if isinstance(frames, int):
        frames = [frames]

    output_path = resolve_path(path)

    clip = clip.resize2.Spline36(format=vs.RGB24)
    clip = PickFrames(clip=clip, indices=frames)

    for i, _ in enumerate(clip.frames()):
        tmp = clip.std.Trim(first=i, last=i).fpng.Write(
            filename=(output_path / (file_name % frames[i])).with_suffix(".png"),  # type: ignore[union-attr]
            overwrite=overwrite,
            compression=2,
        )
        for f in tmp.frames():
            pass
