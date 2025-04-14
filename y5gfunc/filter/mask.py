from vstools import vs
from vstools import core
import vstools
from typing import Optional, Union
from vsrgtools import remove_grain
from vsmasktools import retinex
import functools
from .morpho import minimum, maximum, convolution, inflate
from .utils import get_peak_value_full
from vstools import get_peak_value, scale_mask, replace_ranges
from ..expr import ex_planes


# modified from LoliHouse: https://share.dmhy.org/topics/view/478666_LoliHouse_LoliHouse_1st_Anniversary_Announcement_and_Gift.html
def DBMask(clip: vs.VideoNode) -> vs.VideoNode:
    """Lolihouse's deband mask"""
    nrmasks = core.tcanny.TCanny(
        clip, sigma=0.8, op=2, mode=1, planes=[0, 1, 2]
    ).std.Binarize(scale_mask(7, 8, clip), planes=[0])
    nrmaskb = core.tcanny.TCanny(clip, sigma=1.3, t_h=6.5, op=2, planes=0)
    nrmaskg = core.tcanny.TCanny(clip, sigma=1.1, t_h=5.0, op=2, planes=0)
    nrmask = core.akarin.Expr(
        [nrmaskg, nrmaskb, nrmasks, clip],
        [
            f"a {scale_mask(20, 8, clip)} < {get_peak_value_full(clip)} a {scale_mask(48, 8, clip)} < x {scale_mask(256, 16, clip)} * a {scale_mask(96, 8, clip)} < y {scale_mask(256, 16, clip)} * z ? ? ?",
            "",
        ],
        clip.format.id,
    )
    nrmask = minimum(
        vstools.iterate(nrmask, functools.partial(maximum, planes=[0]), 2), planes=[0]
    )
    nrmask = remove_grain(nrmask, [20, 0])
    nrmask = vstools.get_y(
        nrmask
    )  # first_plane=True in [LoliHouse] Anime_WebSource_deband_1080P_10bit_adcance.vpy: L33
    return nrmask


# modified from muvsfunc.AnimeMask
def AnimeMask(clip: vs.VideoNode, shift: float = 0, mode: int = 1) -> vs.VideoNode:
    """
    Generates edge/ringing mask for anime based on gradient operator.

    For Anime's ringing mask, it's recommended to set "shift" between 0.5 and 1.0.

    Args:
        clip: Source clip. Only the First plane will be processed.
        shift: (float, -1.5 ~ 1.5) The distance of translation. Default is 0.
        mode: (-1 or 1) Type of the kernel, which simply inverts the pixel values and "shift".
            Typically, -1 is for edge, 1 is for ringing. Default is 1.

    Returns:
        Generated mask.

    Raises:
        ValueError: If mode(-1 or 1) is invalid.
    """

    if clip.format.color_family != vs.GRAY:
        clip = vstools.get_y(clip)

    if mode not in [-1, 1]:
        raise ValueError("AnimeMask: 'mode' have not a correct value! [-1 or 1]")

    if mode == -1:
        clip = core.std.Invert(clip)
        shift = -shift

    mask1 = convolution(
        clip, [0, 0, 0, 0, 2, -1, 0, -1, 0], saturate=True
    ).resize2.Bicubic(src_left=shift, src_top=shift, range_s="full", range_in_s="full")
    mask2 = convolution(
        clip, [0, -1, 0, -1, 2, 0, 0, 0, 0], saturate=True
    ).resize2.Bicubic(
        src_left=-shift, src_top=-shift, range_s="full", range_in_s="full"
    )
    mask3 = convolution(
        clip, [0, -1, 0, 0, 2, -1, 0, 0, 0], saturate=True
    ).resize2.Bicubic(src_left=shift, src_top=-shift, range_s="full", range_in_s="full")
    mask4 = convolution(
        clip, [0, 0, 0, -1, 2, 0, 0, -1, 0], saturate=True
    ).resize2.Bicubic(src_left=-shift, src_top=shift, range_s="full", range_in_s="full")

    calc_expr = "src0 2 ** src01 2 ** + src2 2 ** + src3 2 ** + sqrt "

    mask = core.akarin.Expr([mask1, mask2, mask3, mask4], [calc_expr])

    return mask


def get_oped_mask(
    clip: vs.VideoNode,
    ncop: vs.VideoNode,
    nced: vs.VideoNode,
    op_start: int,
    ed_start: int,
    threshold: int = 7,
) -> tuple[vs.VideoNode, vs.VideoNode]:
    assert clip.format == ncop.format == nced.format
    assert clip.format.color_family == vs.YUV

    op_end = op_start + ncop.num_frames
    ed_end = ed_start + nced.num_frames

    assert 0 <= op_start <= op_end < ed_start <= ed_end < clip.num_frames

    if op_start != 0:
        ncop = (
            core.std.Trim(clip, first=0, last=op_start - 1)
            + ncop
            + core.std.Trim(clip, first=op_end + 1, last=clip.num_frames - 1)
        )
    else:
        ncop = ncop + core.std.Trim(clip, first=op_end, last=clip.num_frames - 1)

    if ed_end != clip.num_frames - 1:
        nced = (
            core.std.Trim(clip, first=0, last=ed_start - 1)
            + nced
            + core.std.Trim(clip, first=ed_end + 1, last=clip.num_frames - 1)
        )
    else:
        nced = core.std.Trim(clip, first=0, last=ed_start - 1) + nced

    nc = replace_ranges(clip, ncop, [op_start, op_end])
    nc = replace_ranges(nc, nced, [ed_start, ed_end])

    thr = scale_mask(threshold, 8, clip)
    max = get_peak_value_full(clip)

    diff = core.akarin.Expr([nc, clip], f"x y - abs {thr} < 0 {max} ?")
    diff = vstools.get_y(diff)

    diff = vstools.iterate(diff, maximum, 5)
    diff = vstools.iterate(diff, minimum, 6)

    return nc, diff


# modified from kagefunc.kirsch()
def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    kirsch1 = convolution(src, matrix=[5, 5, 5, -3, 0, -3, -3, -3, -3], saturate=False)
    kirsch2 = convolution(src, matrix=[-3, 5, 5, -3, 0, 5, -3, -3, -3], saturate=False)
    kirsch3 = convolution(src, matrix=[-3, -3, 5, -3, 0, 5, -3, -3, 5], saturate=False)
    kirsch4 = convolution(src, matrix=[-3, -3, -3, -3, 0, 5, -3, 5, 5], saturate=False)
    return core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4], "x y max z max a max")


# modified from vsTaambk
def prewitt(clip: vs.VideoNode, mthr: Union[int, float] = 24) -> vs.VideoNode:
    mthr = scale_mask(mthr, 8, clip)
    prewitt1 = convolution(
        clip, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor=1, saturate=False
    )
    prewitt2 = convolution(
        clip, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False
    )
    prewitt3 = convolution(
        clip, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False
    )
    prewitt4 = convolution(
        clip, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor=1, saturate=False
    )
    prewitt = core.akarin.Expr(
        [prewitt1, prewitt2, prewitt3, prewitt4], "x y max z max a max"
    )
    prewitt = inflate(
        remove_grain(core.akarin.Expr(prewitt, f"x {mthr} <= x 2 / x 1.4 pow ?"), 4)
    )
    return prewitt


# modified from kagefunc.retinex_edgemask()
def retinex_edgemask(src: vs.VideoNode) -> vs.VideoNode:
    luma = vstools.get_y(src)
    max_value = get_peak_value_full(src)
    ret = retinex(luma, sigma=[50, 200, 350], upper_thr=0.005)
    tcanny = minimum(
        ret.tcanny.TCanny(mode=1, sigma=1), coordinates=[1, 0, 1, 0, 0, 1, 0, 1]
    )
    return core.akarin.Expr([kirsch(luma), tcanny], f"x y + {max_value} min")


# modified from kegefunc._generate_descale_mask()
def generate_detail_mask(
    source: vs.VideoNode, upscaled: vs.VideoNode, threshold: float = 0.05
) -> vs.VideoNode:
    assert source.format == upscaled.format
    threshold = scale_mask(threshold, 32, source)
    mask = core.akarin.Expr([source, upscaled], "src0 src1 - abs").std.Binarize(
        threshold=threshold
    )
    mask = vstools.iterate(mask, maximum, 3)
    mask = vstools.iterate(mask, inflate, 3)
    return mask


# modified from jvsfunc.comb_mask()
def comb_mask(
    clip: vs.VideoNode,
    cthresh: int = 6,
    mthresh: int = 9,
    expand: bool = True,
    metric: int = 0,
    planes: Optional[Union[int, list[int]]] = None,
) -> vs.VideoNode:
    """
    Comb mask from TIVTC/TFM plugin.

    Args:
        clip: Input clip.
        cthresh: Spatial combing threshold.
        mthresh: Motion adaptive threshold.
        expand: Assume left and right pixels of combed pixel as combed too.
        metric: Sets which spatial combing metric is used to detect combed pixels.
                - Metric 0 is what TFM used previous to v0.9.12.0.
                - Metric 1 is from Donald Graft's decomb.dll.
        planes: Planes to process.
    """

    cth_max = 65025 if metric else 255
    if (cthresh > cth_max) or (cthresh < 0):
        raise ValueError(
            f"comb_mask: cthresh must be between 0 and {cth_max} when metric = {metric}."
        )
    if (mthresh > 255) or (mthresh < 0):
        raise ValueError("comb_mask: mthresh must be between 0 and 255.")
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]

    peak = get_peak_value(clip)
    ex_m0 = [
        f"x[0,-2] a! x[0,-1] b! x c! x[0,1] d! x[0,2] e! "
        f"c@ b@ - d1! c@ d@ - d2! "
        f"c@ 4 * a@ + e@ + b@ d@ + 3 * - abs fd! "
        f"d1@ {cthresh} > d2@ {cthresh} > and "
        f"d1@ -{cthresh} < d2@ -{cthresh} < and or "
        f"fd@ {cthresh * 6} > and {peak} 0 ?"
    ]

    ex_m1 = [f"x[0,-1] x - x[0,1] x - * {cthresh} > {peak} 0 ?"]
    ex_motion = [f"x y - abs {mthresh} > {peak} 0 ?"]
    ex_spatial = ex_m1 if metric else ex_m0

    spatial_mask = clip.akarin.Expr(ex_planes(clip, ex_spatial, planes))
    if mthresh == 0:
        return (
            spatial_mask
            if not expand
            else maximum(
                spatial_mask, planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]
            )
        )

    motion_mask = core.akarin.Expr(
        [clip, clip[0] + clip], ex_planes(clip, ex_motion, planes)
    )
    motion_mask = maximum(
        motion_mask, planes=planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]
    )
    comb_mask = core.akarin.Expr([spatial_mask, motion_mask], "x y min")

    return (
        comb_mask
        if not expand
        else maximum(comb_mask, planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    )
