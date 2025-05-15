import functools
from varname.core import argname
from varname.utils import ImproperUseError
from vsdenoise.prefilters import PrefilterPartial
from vstools import vs
from vstools import core
import vstools
from vsdenoise import (
    AnalyzeArgs,
    DFTTest,
    FilterType,
    MVToolsPreset,
    MotionMode,
    RFilterMode,
    RecalculateArgs,
    SADMode,
    SearchMode,
    SharpMode,
    SuperArgs,
    mc_degrain,
    Prefilter,
    prefilter_to_full_range,
)
from vsrgtools import remove_grain
import vsrgtools
from vsmasktools import adg_mask
from typing import Callable, Optional, Union
from enum import StrEnum
from .resample import (
    ColorMatrixManager,
    yuv2opp,
    opp2yuv,
    rgb2opp,
    opp2rgb,
    default_opp,
)
from .mask import GammaMask
from .morpho import maximum


class BM3DPreset(StrEnum):
    """
    BM3D speed vs quality presets. `MAGIC` is better classified as tune but whatever.

    modified from <https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D?tab=readme-ov-file#profile-default>
    preset 'magic' is from rksfunc
    """

    FAST = "fast"
    LC = "lc"
    NP = "np"
    HIGH = "high"
    MAGIC = "magic"


bm3d_presets: dict[str, dict[BM3DPreset, dict[str, int]]] = {
    "basic": {
        BM3DPreset.FAST: {"block_step": 8, "bm_range": 9, "ps_num": 2, "ps_range": 4},
        BM3DPreset.LC: {"block_step": 6, "bm_range": 9, "ps_num": 2, "ps_range": 4},
        BM3DPreset.NP: {"block_step": 4, "bm_range": 16, "ps_num": 2, "ps_range": 5},
        BM3DPreset.HIGH: {"block_step": 3, "bm_range": 16, "ps_num": 2, "ps_range": 7},
        BM3DPreset.MAGIC: {"block_step": 3, "bm_range": 12, "ps_num": 2, "ps_range": 8},
    },
    "vbasic": {
        BM3DPreset.FAST: {"block_step": 8, "bm_range": 7, "ps_num": 2, "ps_range": 4},
        BM3DPreset.LC: {"block_step": 6, "bm_range": 9, "ps_num": 2, "ps_range": 4},
        BM3DPreset.NP: {"block_step": 4, "bm_range": 12, "ps_num": 2, "ps_range": 5},
        BM3DPreset.HIGH: {"block_step": 3, "bm_range": 16, "ps_num": 2, "ps_range": 7},
        BM3DPreset.MAGIC: {"block_step": 3, "bm_range": 12, "ps_num": 2, "ps_range": 8},
    },
    "final": {
        BM3DPreset.FAST: {"block_step": 7, "bm_range": 9, "ps_num": 2, "ps_range": 5},
        BM3DPreset.LC: {"block_step": 5, "bm_range": 9, "ps_num": 2, "ps_range": 5},
        BM3DPreset.NP: {"block_step": 3, "bm_range": 16, "ps_num": 2, "ps_range": 6},
        BM3DPreset.HIGH: {"block_step": 2, "bm_range": 16, "ps_num": 2, "ps_range": 8},
        BM3DPreset.MAGIC: {"block_step": 2, "bm_range": 8, "ps_num": 2, "ps_range": 6},
    },
    "vfinal": {
        BM3DPreset.FAST: {"block_step": 7, "bm_range": 7, "ps_num": 2, "ps_range": 5},
        BM3DPreset.LC: {"block_step": 5, "bm_range": 9, "ps_num": 2, "ps_range": 5},
        BM3DPreset.NP: {"block_step": 3, "bm_range": 12, "ps_num": 2, "ps_range": 6},
        BM3DPreset.HIGH: {"block_step": 2, "bm_range": 16, "ps_num": 2, "ps_range": 8},
        BM3DPreset.MAGIC: {"block_step": 2, "bm_range": 8, "ps_num": 2, "ps_range": 6},
    },
}


# modified from rksfunc.BM3DWrapper()
def Fast_BM3DWrapper(
    clip: vs.VideoNode,
    bm3d: Callable = core.lazy.bm3dcpu,
    chroma: bool = True,
    sigma_Y: Union[float, int] = 1.2,
    radius_Y: int = 1,
    delta_sigma_Y: Union[float, int] = 0.6,
    preset_Y_basic: BM3DPreset = BM3DPreset.FAST,
    preset_Y_final: BM3DPreset = BM3DPreset.FAST,
    sigma_chroma: Union[float, int] = 2.4,
    radius_chroma: int = 0,
    delta_sigma_chroma: Union[float, int] = 1.2,
    preset_chroma_basic: BM3DPreset = BM3DPreset.FAST,
    preset_chroma_final: BM3DPreset = BM3DPreset.FAST,
    ref: Optional[vs.VideoNode] = None,
    opp_matrix: ColorMatrixManager = default_opp,
    fast: Optional[bool] = None,
) -> vs.VideoNode:
    """
    BM3D/V-BM3D denoising

    This function performs a two-step BM3D (or V-BM3D if radius > 0) denoising process. Luma (Y) is processed directly.
    Chroma (U/V) is processed by downscaling the denoised luma, joining it with the original chroma, converting to OPP color space
    at half resolution, denoising in OPP, converting back to YUV, and finally joining the denoised luma and denoised chroma planes.

    The 'delta_sigma' parameters add extra strength specifically to the first (basic) denoising step for both luma and chroma.

    Args:
        clip: Input video clip. Must be in YUV420P16 format.
        bm3d: The BM3D plugin implementation to use.
        chroma: If True, process chroma planes. If False, only luma is processed and original chroma is retained.
        sigma_Y: Denoising strength for the luma plane's final step.
        radius_Y: Temporal radius for luma processing. If > 0, V-BM3D is used.
        delta_sigma_Y: Additional sigma added to `sigma_Y` for the luma plane's basic step.
        preset_Y_basic: BM3D parameter preset for the luma basic step.
        preset_Y_final: BM3D parameter preset for the luma final step.
        sigma_chroma: Denoising strength for the chroma planes' final step.
        radius_chroma: Temporal radius for chroma processing. If > 0, V-BM3D is used.
        delta_sigma_chroma: Additional sigma added to `sigma_chroma` for the chroma planes' basic step.
        preset_chroma_basic: BM3D parameter preset for the chroma basic step.
        preset_chroma_final: BM3D parameter preset for the chroma final step.
        ref: Ref for final BM3D step. If provided, basic step is bypassed.
        opp_matrix: OPP transform type to use.
        fast: Multi-threaded copy between CPU and GPU at the expense of 4x memory consumption.

    Returns:
        Denoised video clip in YUV420P16 format.

    Raises:
        ValueError: If the input clip format is not YUV420P16.
        ValueError: If any provided preset name is invalid.
        ValueError: If `fast` is set for non-gpu backend.

    Notes:
        - The `delta_sigma_xxx` value is added to `sigma_xxx` only for the 'basic' denoising step. The 'final' step uses `sigma_xxx` directly.
    """

    if clip.format.id != vs.YUV420P16:
        raise ValueError("Fast_BM3DWrapper: Input clip format must be YUV420P16.")
    if ref:
        if ref.format.id != clip.format.id:
            raise ValueError(
                f"Fast_BM3DWrapper: Input clip and ref must have the same format. Got {ref.format.id} and {clip.format.id}"
            )

    matrix = vstools.Matrix.from_video(clip, strict=True)

    try:
        to_opp = functools.partial(yuv2opp, matrix_in=matrix, opp_manager=opp_matrix)
        to_yuv = functools.partial(
            opp2yuv,
            target_matrix=matrix,
            opp_manager=opp_matrix,
        )
    except ValueError:

        def to_opp(clip) -> vs.VideoNode:
            return rgb2opp(
                core.resize2.Bicubic(clip, format=vs.RGBS, matrix_in=matrix),
                opp_manager=opp_matrix,
            )

        def to_yuv(clip) -> vs.VideoNode:
            return opp2rgb(clip, opp_manager=opp_matrix).resize2.Spline36(
                format=vs.YUV444PS, matrix=matrix
            )

    for preset in [
        preset_Y_basic,
        preset_Y_final,
        preset_chroma_basic,
        preset_chroma_final,
    ]:
        if preset not in [
            BM3DPreset.FAST,
            BM3DPreset.LC,
            BM3DPreset.NP,
            BM3DPreset.HIGH,
            BM3DPreset.MAGIC,
        ]:
            raise ValueError(f"Fast_BM3DWrapper: Unknown preset {preset}.")

    try:
        bm3d_s = argname("bm3d")
    except ImproperUseError:
        bm3d_s = "bm3dcpu"

    if "cpu" in bm3d_s:
        if fast is not None:
            raise ValueError(
                "Fast_BM3DWrapper: bm3dcpu does not support argument `fast`"
            )
        params = {
            "y_basic": bm3d_presets["vbasic" if radius_Y > 0 else "basic"][
                preset_Y_basic
            ],
            "y_final": bm3d_presets["vfinal" if radius_Y > 0 else "final"][
                preset_Y_final
            ],
            "chroma_basic": bm3d_presets["vbasic" if radius_chroma > 0 else "basic"][
                preset_chroma_basic
            ],
            "chroma_final": bm3d_presets["vfinal" if radius_chroma > 0 else "final"][
                preset_chroma_final
            ],
        }
    else:
        if fast is None:
            fast = True
        params = {
            "y_basic": bm3d_presets["vbasic" if radius_Y > 0 else "basic"][
                preset_Y_basic
            ]
            | {"fast": fast},
            "y_final": bm3d_presets["vfinal" if radius_Y > 0 else "final"][
                preset_Y_final
            ]
            | {"fast": fast},
            "chroma_basic": bm3d_presets["vbasic" if radius_chroma > 0 else "basic"][
                preset_chroma_basic
            ]
            | {"fast": fast},
            "chroma_final": bm3d_presets["vfinal" if radius_chroma > 0 else "final"][
                preset_chroma_final
            ]
            | {"fast": fast},
        }

    half_width = clip.width // 2
    half_height = clip.height // 2
    srcY_float, srcU_float, srcV_float = vstools.split(vstools.depth(clip, 32))

    if ref is None:
        basic_y = bm3d.BM3Dv2(
            clip=srcY_float,
            ref=srcY_float,
            sigma=sigma_Y + delta_sigma_Y,
            radius=radius_Y,
            **params["y_basic"],
        )
    else:
        basic_y = vstools.depth(vstools.get_y(ref), 32)

    final_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=basic_y,
        sigma=sigma_Y,
        radius=radius_Y,
        **params["y_final"],
    )

    vyhalf = final_y.resize2.Spline36(half_width, half_height, src_left=-0.5)
    srchalf_444 = vstools.join([vyhalf, srcU_float, srcV_float])
    srchalf_opp = to_opp(srchalf_444)
    if ref:
        refhalf444 = vstools.join(vyhalf, vstools.depth(ref, 32))
        refhalf_opp = to_opp(refhalf444)

    if ref is None:
        basic_half = bm3d.BM3Dv2(
            clip=srchalf_opp,
            ref=srchalf_opp,
            sigma=sigma_chroma + delta_sigma_chroma,
            chroma=chroma,
            radius=radius_chroma,
            zero_init=0,
            **params["chroma_basic"],
        )
    else:
        basic_half = refhalf_opp

    final_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=basic_half,
        sigma=sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0,
        **params["chroma_final"],
    )

    final_half = to_yuv(final_half)
    _, final_u, final_v = vstools.split(final_half)
    vfinal = vstools.join([final_y, final_u, final_v])
    return vstools.depth(vfinal, 16)


# Inspired by mawen1250's bm3d readme, Vodesfunc and EoEfunc
def hybrid_denoise(
    clip: vs.VideoNode,
    mc_degrain_prefilter: PrefilterPartial = Prefilter.DFTTEST(),
    mc_degrain_preset: Optional[MVToolsPreset] = None,
    mc_degrain_refine: int = 2,
    mc_degrain_thsad: int = 100,
    show_ref: bool = False,
    bm3d_sigma: Union[float, int] = 2,
    bm3d_preset: BM3DPreset = BM3DPreset.FAST,
    bm3d_opp_matrix: ColorMatrixManager = default_opp,
) -> Union[vs.VideoNode, tuple[vs.VideoNode, vs.VideoNode]]:
    """
    mc_degrain + bm3d denoise
    """
    if clip.format.id != vs.YUV420P16:
        raise ValueError("hybrid_denoise: Input clip format must be YUV420P16.")

    if clip.width <= 1024 and clip.height <= 576:
        block_size = 32
        overlap = 16
    elif clip.width <= 2048 and clip.height <= 1536:
        block_size = 64
        overlap = 32
    else:
        block_size = 128
        overlap = 64

    if mc_degrain_preset is None:
        mc_degrain_preset = MVToolsPreset(
            search_clip=prefilter_to_full_range,  # type: ignore
            pel=2,
            super_args=SuperArgs(sharp=SharpMode.WIENER, rfilter=RFilterMode.TRIANGLE),
            analyze_args=AnalyzeArgs(
                blksize=block_size,
                overlap=overlap,
                search=SearchMode.DIAMOND,
                dct=SADMode.ADAPTIVE_SPATIAL_MIXED,
                truemotion=MotionMode.SAD,
                pelsearch=2,
            ),
            recalculate_args=RecalculateArgs(
                blksize=int(block_size / 2),
                overlap=int(overlap / 2),
                search=SearchMode.DIAMOND,
                dct=SADMode.ADAPTIVE_SPATIAL_MIXED,
                truemotion=MotionMode.SAD,
                searchparam=1,
            ),
        )

    ref = mc_degrain(
        clip=clip,
        prefilter=mc_degrain_prefilter,
        preset=mc_degrain_preset,
        thsad=mc_degrain_thsad,
        thsad_recalc=mc_degrain_thsad,
        blksize=block_size,
        refine=mc_degrain_refine,
    )

    bm3d = Fast_BM3DWrapper(
        clip,
        sigma_Y=bm3d_sigma,
        sigma_chroma=bm3d_sigma,
        preset_Y_final=bm3d_preset,
        preset_chroma_final=bm3d_preset,
        ref=ref,
        opp_matrix=bm3d_opp_matrix,
    )
    if show_ref:
        return bm3d, ref
    else:
        return bm3d


# modified from soifunc
def magic_denoise(clip: vs.VideoNode) -> vs.VideoNode:
    """
    Uses dark magic to denoise heavy grain from videos.
    """
    super = core.mv.Super(clip, hpad=16, vpad=16, rfilter=4)

    backward2 = core.mv.Analyse(
        super, isb=True, blksize=16, overlap=8, delta=2, search=3, dct=6
    )
    backward = core.mv.Analyse(super, isb=True, blksize=16, overlap=8, search=3, dct=6)
    forward = core.mv.Analyse(super, isb=False, blksize=16, overlap=8, search=3, dct=6)
    forward2 = core.mv.Analyse(
        super, isb=False, blksize=16, overlap=8, delta=2, search=3, dct=6
    )

    backward2 = core.mv.Recalculate(
        super, backward2, blksize=8, overlap=4, search=3, divide=2, dct=6
    )
    backward = core.mv.Recalculate(
        super, backward, blksize=8, overlap=4, search=3, divide=2, dct=6
    )
    forward = core.mv.Recalculate(
        super, forward, blksize=8, overlap=4, search=3, divide=2, dct=6
    )
    forward2 = core.mv.Recalculate(
        super, forward2, blksize=8, overlap=4, search=3, divide=2, dct=6
    )

    backward_re2 = core.mv.Finest(backward2)
    backward_re = core.mv.Finest(backward)
    forward_re = core.mv.Finest(forward)
    forward_re2 = core.mv.Finest(forward2)

    clip = core.mv.Degrain2(
        clip,
        super,
        backward_re,
        forward_re,
        backward_re2,
        forward_re2,
        thsad=220,
        thscd1=300,
    )

    return DFTTest(
        sloc=[(0.0, 0.8), (0.06, 1.1), (0.12, 1.0), (1.0, 1.0)],
    ).denoise(
        clip,
        pmax=1000000,
        pmin=1.25,
        ftype=FilterType.MULT_RANGE,
        tbsize=3,
        ssystem=1,
    )


# modified from rksfunc
def adaptive_denoise(clip: vs.VideoNode, denoised: vs.VideoNode) -> vs.VideoNode:
    bilateral = vsrgtools.bilateral(clip, denoised, 0.5)
    amask = vstools.iterate(denoised, functools.partial(remove_grain, mode=[20, 11]), 2)
    amask = adg_mask(amask, luma_scaling=12)
    degrain = core.std.MaskedMerge(denoised, bilateral, amask, first_plane=True)
    clear_edge = core.std.MaskedMerge(degrain, denoised, maximum(GammaMask(denoised)))
    return vstools.join(clear_edge, denoised)
