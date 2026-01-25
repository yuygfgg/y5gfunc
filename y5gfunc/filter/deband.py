from vstools import vs
from vstools import core
import vstools
from typing import Union, Optional, Literal, Callable
import functools
import mvsfunc as mvf
from vsdeband import deband_detail_mask


from .mask import retinex_edgemask
from vsrgtools import remove_grain
from .morpho import inflate as _inflate  # Fucking naming conflict


# modified from rksfunc.SynDeband()
def sakiko_deband(
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
    """Sakiko Togawa is good at de-band"""
    assert clip.format.id == vs.YUV420P16

    if kill is None:
        kill = vstools.iterate(clip, functools.partial(remove_grain, mode=[20, 11]), 2)
    elif not kill:
        kill = clip

    assert isinstance(kill, vs.VideoNode)
    grain = core.std.MakeDiff(clip, kill)
    f3kdb_params = {
        "grainy": 0,
        "grainc": 0,
        "sample_mode": 2,
        "blur_first": True,
        "dither_algo": 2,
    }
    f3k1 = kill.neo_f3kdb.Deband(r1, y1, uv1, uv1, **f3kdb_params)
    f3k2 = f3k1.neo_f3kdb.Deband(r2, y2, uv2, uv2, **f3kdb_params)
    if limit:
        f3k2 = mvf.LimitFilter(
            f3k2, kill, thr=limit_thry, thrc=limit_thrc, elast=limit_elast
        )
    if bmask is None:
        bmask = retinex_edgemask(kill).std.Binarize(mstr)
        bmask = vstools.iterate(bmask, _inflate, inflate)
    deband = core.std.MaskedMerge(f3k2, kill, bmask)
    deband = core.std.MergeDiff(deband, grain)
    if include_mask:
        return deband, bmask
    else:
        return deband


# modified from rksfunc.SynDebandv2()
def sakiko_deband_v2(
    clip: vs.VideoNode,
    preset: Literal["low", "high", "mid"] = "low",
    banding_mask: Optional[vs.VideoNode] = None,
    debander: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
    killer: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
    ampo: float = 1.0,
    include_mask: bool = False,
) -> Union[vs.VideoNode, tuple[vs.VideoNode, vs.VideoNode]]:
    if banding_mask is None:
        banding_mask = (
            deband_detail_mask(clip).std.Maximum().std.Maximum().std.Deflate()
        )
    kill = clip if killer is None else killer(clip)
    if debander is None:
        deband = vstools.depth(kill, 32)
        _d = deband.vszip.Deband
        match preset:
            case "low":
                deband = _d(
                    range=12,
                    thr=0.6,
                    grain=0,
                    sample_mode=7,
                    thr1=1.9,
                    thr2=1.2,
                    angle_boost=1.9,
                )
                deband = _d(
                    range=22,
                    thr=0.5,
                    grain=0,
                    sample_mode=7,
                    thr1=1.7,
                    thr2=1.1,
                    angle_boost=1.8,
                )
            case "mid":
                deband = _d(
                    range=12,
                    thr=1.8,
                    grain=0,
                    sample_mode=7,
                    thr1=4.0,
                    thr2=2.0,
                    angle_boost=1.6,
                )
                deband = _d(
                    range=22,
                    thr=1.6,
                    grain=0,
                    sample_mode=7,
                    thr1=3.6,
                    thr2=1.8,
                    angle_boost=1.5,
                )
            case "high":
                deband = _d(
                    range=12, thr=3.4, grain=0, sample_mode=6, thr1=6.8, thr2=3.3
                )
                deband = _d(
                    range=12, thr=3.2, grain=0, sample_mode=6, thr1=6.4, thr2=3.1
                )
            case _:
                raise ValueError(f"{preset = } is not available.")
        origin = kill.format.bits_per_sample
        if origin < 32:
            deband = deband.fmtc.bitdepth(bits=origin, dmode=0, ampn=0.1, ampo=ampo)
    else:
        deband = debander(kill)
    deband = core.std.MaskedMerge(deband, kill, banding_mask)
    if include_mask:
        if killer is not None:
            return deband.std.MergeFullDiff(clip.std.MakeFullDiff(kill)), banding_mask
        else:
            return deband, banding_mask
    else:
        if killer is not None:
            return deband.std.MergeFullDiff(clip.std.MakeFullDiff(kill))
        else:
            return deband
