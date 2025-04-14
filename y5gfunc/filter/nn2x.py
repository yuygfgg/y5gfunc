import functools
from vstools import core
from vstools import vs


def nn2x(
    clip: vs.VideoNode,
    opencl: bool = True,
    nnedi3_args: dict[str, int] = {"field": 1, "nsize": 4, "nns": 4, "qual": 2},
) -> vs.VideoNode:
    """Doubles the resolution of input clip with nnedi3."""
    if hasattr(core, "sneedif") and opencl:
        nnedi3 = functools.partial(core.sneedif.NNEDI3, **nnedi3_args)
        return nnedi3(nnedi3(clip, dh=True), dw=True)
    elif hasattr(core, "nnedi3cl") and opencl:
        nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)
        return nnedi3(nnedi3(clip, dh=True), dw=True)
    else:
        nnedi3 = functools.partial(core.nnedi3.nnedi3, **nnedi3_args)
        return nnedi3(nnedi3(clip, dh=True).std.Transpose(), dh=True).std.Transpose()
