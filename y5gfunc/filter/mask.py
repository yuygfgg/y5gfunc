from vstools import vs
from vstools import core
import vsutil
from typing import Callable, Any
from vsrgtools import remove_grain
import functools
from .morpho import minimum, maximum, convolution
from .utils import scale_value_full, get_peak_value_full

# modified from LoliHouse: https://share.dmhy.org/topics/view/478666_LoliHouse_LoliHouse_1st_Anniversary_Announcement_and_Gift.html
def DBMask(clip: vs.VideoNode) -> vs.VideoNode:
    nrmasks = core.tcanny.TCanny(clip, sigma=0.8, op=2, mode=1, planes=[0, 1, 2]).std.Binarize(scale_value_full(7, 8, clip), planes=[0])
    nrmaskb = core.tcanny.TCanny(clip, sigma=1.3, t_h=6.5, op=2, planes=0)
    nrmaskg = core.tcanny.TCanny(clip, sigma=1.1, t_h=5.0, op=2, planes=0)
    nrmask = core.akarin.Expr([nrmaskg, nrmaskb, nrmasks, clip],[f"a {scale_value_full(20, 8, clip)} < {get_peak_value_full(clip)} a {scale_value_full(48, 8, clip)} < x {scale_value_full(256, 16, clip)} * a {scale_value_full(96, 8, clip)} < y {scale_value_full(256, 16, clip)} * z ? ? ?",""], clip.format.id)
    nrmask = minimum(vsutil.iterate(nrmask, functools.partial(maximum, planes=[0]), 2), planes=[0])
    nrmask = remove_grain(nrmask, [20, 0])
    nrmask = vsutil.get_y(nrmask) # first_plane=True in [LoliHouse] Anime_WebSource_deband_1080P_10bit_adcance.vpy: L33
    return nrmask

def get_oped_mask(
    clip: vs.VideoNode,
    ncop: vs.VideoNode,
    nced: vs.VideoNode,
    op_start: int,
    ed_start: int,
    threshold: int = 7
) -> tuple[vs.VideoNode, vs.VideoNode]:

    from fvsfunc import rfs
    
    assert clip.format == ncop.format == nced.format
    assert clip.format.color_family == vs.YUV
    
    op_end = op_start + ncop.num_frames
    ed_end = ed_start + nced.num_frames
    
    assert 0 <= op_start <= op_end < ed_start <= ed_end < clip.num_frames
    
    if op_start != 0:
        ncop = core.std.Trim(clip, first=0, last=op_start-1) + ncop + core.std.Trim(clip, first=op_end+1, last=clip.num_frames-1)
    else:
        ncop = ncop + core.std.Trim(clip, first=op_end, last=clip.num_frames-1)
    
    if ed_end != clip.num_frames - 1:
        nced = core.std.Trim(clip, first=0, last=ed_start-1) + nced + core.std.Trim(clip, first=ed_end+1, last=clip.num_frames-1)
    else:
        nced = core.std.Trim(clip, first=0, last=ed_start-1) + nced
    
    nc = rfs(clip, ncop, f"[{op_start} {op_end}]")
    nc = rfs(nc, nced, f"[{ed_start} {ed_end}]")
    
    thr = scale_value_full(threshold, 8, clip)
    max = get_peak_value_full(clip)

    diff = core.akarin.Expr([nc, clip], f"x y - abs {thr} < 0 {max} ?")
    diff = vsutil.get_y(diff)
    
    diff = vsutil.iterate(diff, maximum, 5)
    diff = vsutil.iterate(diff, minimum, 6)
    
    return nc, diff

# modified from vardefunc.cambi_mask
def cambi_mask(
    clip: vs.VideoNode,
    scale: int = 1,
    merge_previous: bool = True,
    blur_func: Callable[[vs.VideoNode], vs.VideoNode] = lambda clip: core.std.BoxBlur(clip, planes=0, hradius=2, hpasses=3),
    **cambi_args: Any
) -> vs.VideoNode:
    
    assert 0 <= scale < 5
    assert callable(blur_func)
    
    if vsutil.get_depth(clip) > 10:
        clip = vsutil.depth(clip, 10, dither_type="none")

    scores = core.akarin.Cambi(clip, scores=True, **cambi_args)
    if merge_previous:
        cscores = [
            blur_func(scores.std.PropToClip(f'CAMBI_SCALE{i}').std.Deflate().std.Deflate())
            for i in range(scale + 1)
        ]
        expr_parts = [f"src{i} {scale + 1} /" for i in range(scale + 1)]
        expr = " ".join(expr_parts) + " " + " ".join(["+"] * (scale))
        deband_mask = core.akarin.Expr([core.resize2.Bilinear(c, scores.width, scores.height) for c in cscores], expr)
    else:
        deband_mask = blur_func(scores.std.PropToClip(f'CAMBI_SCALE{scale}').std.Deflate().std.Deflate())

    return deband_mask.std.CopyFrameProps(scores)

# modified from kagefunc.kirsch()
def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    kirsch1 = convolution(src, matrix=[ 5,  5,  5, -3,  0, -3, -3, -3, -3], saturate=False)
    kirsch2 = convolution(src, matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], saturate=False)
    kirsch3 = convolution(src, matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], saturate=False)
    kirsch4 = convolution(src, matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], saturate=False)
    return core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4], 'x y max z max a max')

# modified from kagefunc.retinex_edgemask()
def retinex_edgemask(src: vs.VideoNode) -> vs.VideoNode:
    luma = vsutil.get_y(src)
    max_value = get_peak_value_full(src)
    ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    tcanny = minimum(ret.tcanny.TCanny(mode=1, sigma=1), coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    return core.akarin.Expr([kirsch(luma), tcanny], f'x y + {max_value} min')

# modified from kegefunc._generate_descale_mask()
def generate_detail_mask(source: vs.VideoNode, upscaled: vs.VideoNode, threshold: float = 0.05) -> vs.VideoNode:
    assert source.format == upscaled.format
    threshold = scale_value_full(threshold, 32, source)
    mask = core.akarin.Expr([source, upscaled], 'src0 src1 - abs').std.Binarize(threshold=threshold)
    mask = vsutil.iterate(mask, maximum, 3)
    mask = vsutil.iterate(mask, core.std.Inflate, 3)
    return mask