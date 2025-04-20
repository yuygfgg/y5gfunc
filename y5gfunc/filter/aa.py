from typing import Callable, Optional
from vstools import core
from vstools import vs
from vstools import depth, join, get_y
from .mask import prewitt
from .morpho import maximum
from .resample import SSIM_downsample, nn2x


def double_aa(clip: vs.VideoNode, mask: Optional[vs.VideoNode] = None, doubler: Callable[[vs.VideoNode], vs.VideoNode] = nn2x) -> vs.VideoNode:
    """
    Apply light anti-aliasing to input video clip. Suitable for recent non-descalable anime.
    
    The function first doubles the resolution of the input clip with `doubler`, then downscales back with SSIM_downsample.
    Chroma planes are not touched.

    Args:
        clip: Input video clip.
        mask: If provided, will be used as mask to merge anti-aliased clip and source clip. Otherwise, Prewitt mask is used.
        doubler: Function to double the clip.
    Returns:
        Anti-aliased input video clip.
    """

    return core.std.MaskedMerge(clip, join(depth(SSIM_downsample(doubler(get_y(clip)), width=clip.width, height=clip.height, src_left=-0.5, src_top=-0.5), clip), clip), mask or maximum(prewitt(clip)), first_plane=True)
    
    