import functools
from vsdehalo import fine_dehalo, hq_dering
from vsdenoise import Prefilter
from vstools import core, vs, iterate
from typing import Literal
from .mask import AnimeMask


def simple_dehalo(
    clip: vs.VideoNode,
    mode: Literal["fine_dehalo", "hq_dering"] = "fine_dehalo",
    maximum: int = 1,
) -> vs.VideoNode:
    """
    Apply a simple dehalo filter to the input clip.

    Args:
        clip: The input video clip to be processed.

    Returns:
        The processed video clip with reduced halos.
    """
    func = (
        functools.partial(hq_dering, smooth=Prefilter.BILATERAL)
        if mode == "hq_dering"
        else functools.partial(fine_dehalo)
    )
    return core.akarin.Expr(
        [
            clip.std.MaskedMerge(
                func(clip),
                iterate(
                    AnimeMask(clip, -0.75, 1).std.Binarize(1), core.std.Maximum, maximum
                ),
                first_plane=True,
            ),
            clip,
        ],
        "x y min",
    )
