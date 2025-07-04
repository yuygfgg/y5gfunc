from collections import deque
from vstools import vs
from typing import Literal, Optional
import functools
import fractions
import collections
from ..utils import resolve_path


# modified from https://github.com/OrangeChannel/acsuite/blob/e40f50354a2fc26f2a29bf3a2fe76b96b2983624/acsuite/__init__.py#L252
def get_frame_timestamp(
    frame_num: int,
    clip: vs.VideoNode,
    precision: Literal[
        "second", "millisecond", "microsecond", "nanosecond"
    ] = "millisecond",
    timecodes_v2_file: Optional[str] = None,
) -> str:
    """
    Get the timestamp of a frame in a video clip.

    Args:
        frame_num: The frame number to get the timestamp for.
        clip: The video clip to get the timestamp for.
        precision: The precision of the timestamp.
        timecodes_v2_file: The optional path to the timecodes file. If provided, the timestamp will be read from the file.

    Returns:
        The timestamp of the frame.
    """
    assert frame_num >= 0
    assert timecodes_v2_file is None or resolve_path(timecodes_v2_file).exists()

    if frame_num == 0:
        s = 0.0
    elif clip.fps != fractions.Fraction(0, 1):
        t = round(float(10**9 * frame_num * clip.fps**-1))
        s = t / 10**9
    else:
        if timecodes_v2_file is not None:
            timecodes = [
                float(x) / 1000
                for x in open(timecodes_v2_file, "r").read().splitlines()[1:]
            ]
            s = timecodes[frame_num]
        else:
            s = clip_to_timecodes(clip)[frame_num]

    m = s // 60
    s %= 60
    h = m // 60
    m %= 60

    if precision == "second":
        return f"{h:02.0f}:{m:02.0f}:{round(s):02}"
    elif precision == "millisecond":
        return f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
    elif precision == "microsecond":
        return f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
    elif precision == "nanosecond":
        return f"{h:02.0f}:{m:02.0f}:{s:012.9f}"


# TODO: use fps for CFR clips
# modified from https://github.com/OrangeChannel/acsuite/blob/e40f50354a2fc26f2a29bf3a2fe76b96b2983624/acsuite/__init__.py#L305
@functools.lru_cache
def clip_to_timecodes(clip: vs.VideoNode, path: Optional[str] = None) -> deque[float]:
    """
    Generate timecodes for a video clip.

    Args:
        clip: The video clip to generate timecodes for.
        path: The optional path to the timecodes file. If provided, the timecodes will be written to the file.

    Returns:
        A deque of timecodes.
    """
    if path:
        path = resolve_path(path)  # type: ignore

    timecodes = collections.deque([0.0], maxlen=clip.num_frames + 1)
    curr_time = fractions.Fraction()
    init_percentage = 0

    with open(path, "w", encoding="utf-8") if path else None as file:  # type: ignore
        if file:
            file.write("# timecode format v2\n")

        for _, frame in enumerate(clip.frames()):
            num: int = frame.props["_DurationNum"]  # type: ignore
            den: int = frame.props["_DurationDen"]  # type: ignore
            curr_time += fractions.Fraction(num, den)
            timecode = float(curr_time)
            timecodes.append(timecode)

            if file:
                file.write(f"{timecode:.6f}\n")

            percentage_done = round(100 * len(timecodes) / clip.num_frames)
            if percentage_done % 10 == 0 and percentage_done != init_percentage:
                print(
                    f"Finding timecodes for variable-framerate clip: {percentage_done}% done"
                )
                init_percentage = percentage_done

    return timecodes
