from typing import Optional
from vstools import core
from vstools import vs
from vstools import depth, join, get_y
from .mask import prewitt
from .nn2x import nn2x
from muvsfunc import SSIM_downsample


def nn2x_aa(clip: vs.VideoNode, mask: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    """
    Apply light anti-aliasing to input video clip. Suitable for recent non-descalable anime.
    
    The function first doubles the resolution of the input clip with nnedi3, then downscales back with SSIM_downsample.
    Chroma planes are not touched.

    Args:
        clip: Input video clip.
        mask: If provided, will be used as mask to merge anti-aliased clip and source clip. Otherwise, Prewitt mask is used.

    Returns:
        Anti-aliased input video clip.
    """

    return core.std.MaskedMerge(clip, join(depth(SSIM_downsample(nn2x(get_y(clip)), w=clip.width, h=clip.height, src_left=-0.5, src_top=-0.5), clip), clip), mask or prewitt(clip), first_plane=True)
    
    