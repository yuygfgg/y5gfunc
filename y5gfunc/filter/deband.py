import vapoursynth as vs
from vapoursynth import core
import vsutil
from typing import Union, Optional
import functools
import mvsfunc as mvf

from .masks import retinex_edgemask
from vsrgtools import remove_grain

# modified from rksfunc.SynDeband()
def SynDeband(
    clip: vs.VideoNode, 
    r1: int = 14, 
    y1: int = 72, 
    uv1: int = 48, 
    r2: int = 30,
    y2: int = 48, 
    uv2: int = 32, 
    mstr: int = 6000, 
    inflate: int = 2,
    include_mask: bool = False, 
    kill: Optional[vs.VideoNode] = None, 
    bmask: Optional[vs.VideoNode] = None,
    limit: bool = True,
    limit_thry: float = 0.12,
    limit_thrc: float = 0.1,
    limit_elast: float = 20,
) -> Union[vs.VideoNode, tuple[vs.VideoNode, vs.VideoNode]]:
    
    assert clip.format.id == vs.YUV420P16
    
    if kill is None:
        kill = vsutil.iterate(clip, functools.partial(remove_grain, mode=[20, 11]), 2)
    elif not kill:
        kill = clip
    
    assert isinstance(kill, vs.VideoNode)
    grain = core.std.MakeDiff(clip, kill)
    f3kdb_params = {
        'grainy': 0,
        'grainc': 0,
        'sample_mode': 2,
        'blur_first': True,
        'dither_algo': 2,
    }
    f3k1 = kill.neo_f3kdb.Deband(r1, y1, uv1, uv1, **f3kdb_params)
    f3k2 = f3k1.neo_f3kdb.Deband(r2, y2, uv2, uv2, **f3kdb_params)
    if limit:
        f3k2 = mvf.LimitFilter(f3k2, kill, thr=limit_thry, thrc=limit_thrc, elast=limit_elast)
    if bmask is None:
        bmask = retinex_edgemask(kill).std.Binarize(mstr)
        bmask = vsutil.iterate(bmask, core.std.Inflate, inflate)
    deband = core.std.MaskedMerge(f3k2, kill, bmask)
    deband = core.std.MergeDiff(deband, grain)
    if include_mask:
        return deband, bmask
    else:
        return deband