from vsdehalo import fine_dehalo
from vstools import core, vs
from .mask import AnimeMask


def simple_dehalo(clip: vs.VideoNode) -> vs.VideoNode:
    """
    Apply a simple dehalo filter to the input clip.

    Args:
        clip: The input video clip to be processed.

    Returns:
        The processed video clip with reduced halos.
    """
    return core.akarin.Expr(
        [
            clip.std.MaskedMerge(
                fine_dehalo(clip),
                AnimeMask(clip, -0.75, 1).std.Binarize(1).std.Maximum(),
                first_plane=True,
            ),
            clip,
        ],
        "x y min",
    )
