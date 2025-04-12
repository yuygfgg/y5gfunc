from typing import List, Union
from vstools import vs
from pathlib import Path

def ranger(start: Union[int, float], end: Union[int, float], step: Union[int, float]) -> List[Union[int, float]]:
    """
    Generates a sequence of numbers similar to range(), but allows floats.

    Creates a list of numbers starting from `start`, incrementing by `step`, and stopping before `end`.

    Args:
        start: The starting value of the sequence (inclusive).
        end: The end value of the sequence (exclusive). The sequence generated will contain values strictly 
            less than `end` if `step` is positive, or strictly greater than `end` if `step` is negative.
        step: The step/increment between consecutive numbers. Must not be zero. Can be negative for descending sequences.

    Returns:
        List[Union[int, float]]: A list of numbers (integers or floats) representing the generated sequence.

    Raises:
        ValueError: If `step` is 0.
    """

    if step == 0:
        raise ValueError("ranger: Step cannot be zero.")
    return list(map(lambda i: round(start + i * step, 10), range(int((end - start) / step))))

# https://discord.com/channels/1168547111139283026/1168591112160690227/1356645786342920202
def PickFrames(clip: vs.VideoNode, indices: list[int]) -> vs.VideoNode:
    """
    Return a new clip with frames picked from input clip from the indices array.

    Args:
        clip: Input clip where frames are from.
        indices: The indices array representing the frames to be picked.

    Returns:
        New clip with frames picked from input clip from the indices array.
    """    
    return clip.std.SelectEvery(cycle=clip.num_frames, offsets=indices)

def resolve_path(path: Union[Path, str]) -> Path:
    """
    Resolves a path to an absolute path and ensures necessary directories exist.

    Args:
        path: The input path string or Path object to resolve and for which to ensure directory structure exists.

    Returns:
        Path: The absolute, resolved pathlib.Path object corresponding to the input path, after ensuring the relevant directory structure exists.
    """
    path = Path(path).resolve()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path