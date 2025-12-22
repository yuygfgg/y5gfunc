import functools
import torch
import numpy as np
from varname.core import argname
from varname.utils import ImproperUseError
from vsdenoise.prefilters import PrefilterPartial
from vstools import vs
from vstools import core
import vstools
from vsdenoise import (
    AnalyzeArgs,
    DFTTest,
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
from scipy.stats import rv_continuous
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
from .utils import get_peak_value_full


def _get_bm3d_backend() -> tuple[Callable, str]:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        if any(
            s in device_name
            for s in [
                "nvidia",
                "geforce",
                "quadro",
                "tesla",
                "rtx",
                "gtx",
                "rt",
                "gt",
                "titan",
            ]
        ):
            if hasattr(core, "bm3dcuda_rtc"):
                return core.lazy.bm3dcuda_rtc, "bm3dcuda_rtc"  # type: ignore[attr-defined]
            if hasattr(core, "bm3dcuda"):
                return core.lazy.bm3dcuda, "bm3dcuda"  # type: ignore[attr-defined]
        elif any(
            s in device_name
            for s in [
                "amd",
                "radeon",
                "vega",
                "firepro",
                "rx",
                "r9",
                "r7",
                "r5",
                "r3",
                "r2",
                "r1",
                "r0",
            ]
        ):
            if hasattr(core, "bm3dhip"):
                return core.lazy.bm3dhip, "bm3dhip"  # type: ignore[attr-defined]

    if hasattr(core, "bm3dsycl") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return core.lazy.bm3dsycl, "bm3dsycl"  # type: ignore[attr-defined]

    if (
        hasattr(core, "bm3dmetal")
        and hasattr(torch, "mps")
        and torch.mps.is_available()
    ):
        return core.lazy.bm3dmetal, "bm3dmetal"  # type: ignore[attr-defined]

    return core.lazy.bm3dcpu, "bm3dcpu"  # type: ignore[attr-defined]


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


_bm3d_presets: dict[str, dict[BM3DPreset, dict[str, int]]] = {
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
    bm3d: Optional[Callable] = None,
    chroma: bool = True,
    sigma_Y: Union[float, int] = 1.2,
    radius_Y: int = 1,
    delta_sigma_Y: Union[float, int] = 0.6,
    preset: Optional[BM3DPreset] = None,
    preset_Y_basic: Optional[BM3DPreset] = None,
    preset_Y_final: Optional[BM3DPreset] = None,
    sigma_chroma: Union[float, int] = 2.4,
    radius_chroma: int = 0,
    delta_sigma_chroma: Union[float, int] = 1.2,
    preset_chroma_basic: Optional[BM3DPreset] = None,
    preset_chroma_final: Optional[BM3DPreset] = None,
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
        bm3d: The BM3D plugin implementation to use. If `None` (default), it will automatically detect and
            select the best available backend (`bm3dcuda_rtc`, `bm3dcuda`, `bm3dhip`, `bm3dsycl`, `bm3dmetal`).
            If no compatible GPU hardware is found, it falls back to `bm3dcpu`.
        chroma: If True, process chroma planes. If False, only luma is processed and original chroma is retained.
        sigma_Y: Denoising strength for the luma plane's final step.
        radius_Y: Temporal radius for luma processing. If > 0, V-BM3D is used.
        delta_sigma_Y: Additional sigma added to `sigma_Y` for the luma plane's basic step.
        preset: BM3D parameter preset for all steps (basic and final, for both luma and chroma).
            This is a convenience parameter that sets all four preset_* parameters at once.
            Individual preset_* parameters will override this if specified.
            If `None` (default), individual presets default to `BM3DPreset.LC` for GPU backends and `BM3DPreset.FAST` for CPU backends.
        preset_Y_basic: BM3D parameter preset for the luma basic step.
            If `None`, uses `preset` if specified, otherwise defaults based on backend.
        preset_Y_final: BM3D parameter preset for the luma final step.
            If `None`, uses `preset` if specified, otherwise defaults based on backend.
        sigma_chroma: Denoising strength for the chroma planes' final step.
        radius_chroma: Temporal radius for chroma processing. If > 0, V-BM3D is used.
        delta_sigma_chroma: Additional sigma added to `sigma_chroma` for the chroma planes' basic step.
        preset_chroma_basic: BM3D parameter preset for the chroma basic step.
            If `None`, uses `preset` if specified, otherwise defaults based on backend.
        preset_chroma_final: BM3D parameter preset for the chroma final step.
            If `None`, uses `preset` if specified, otherwise defaults based on backend.
        ref: Ref for final BM3D step. If provided, basic step is bypassed.
        opp_matrix: OPP transform type to use.
        fast: Multi-threaded copy between CPU and GPU at the expense of 4x memory consumption. Only available for GPU backends.
            If `None` (default), it's set to `True` for non-metal GPU backends and `False` for other backends.

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

    bm3d_s: str
    _bm3d: Callable
    if bm3d is None:
        _bm3d, bm3d_s = _get_bm3d_backend()
    else:
        _bm3d = bm3d
        try:
            bm3d_s = argname("bm3d")
        except ImproperUseError:
            bm3d_s = "bm3dcpu"

    is_gpu = "cpu" not in bm3d_s
    default_preset = BM3DPreset.LC if is_gpu else BM3DPreset.FAST

    # Specific preset_* > preset > default_preset
    if preset_Y_basic is None:
        preset_Y_basic = preset if preset is not None else default_preset
    if preset_Y_final is None:
        preset_Y_final = preset if preset is not None else default_preset
    if preset_chroma_basic is None:
        preset_chroma_basic = preset if preset is not None else default_preset
    if preset_chroma_final is None:
        preset_chroma_final = preset if preset is not None else default_preset

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

    for p in [
        preset,
        preset_Y_basic,
        preset_Y_final,
        preset_chroma_basic,
        preset_chroma_final,
    ]:
        if p is not None and p not in [
            BM3DPreset.FAST,
            BM3DPreset.LC,
            BM3DPreset.NP,
            BM3DPreset.HIGH,
            BM3DPreset.MAGIC,
        ]:
            raise ValueError(f"Fast_BM3DWrapper: Unknown preset {p}.")

    if "cpu" in bm3d_s:
        if fast is not None:
            raise ValueError(
                "Fast_BM3DWrapper: bm3dcpu does not support argument `fast`"
            )
        params = {
            "y_basic": _bm3d_presets["vbasic" if radius_Y > 0 else "basic"][
                preset_Y_basic
            ],
            "y_final": _bm3d_presets["vfinal" if radius_Y > 0 else "final"][
                preset_Y_final
            ],
            "chroma_basic": _bm3d_presets["vbasic" if radius_chroma > 0 else "basic"][
                preset_chroma_basic
            ],
            "chroma_final": _bm3d_presets["vfinal" if radius_chroma > 0 else "final"][
                preset_chroma_final
            ],
        }
    else:
        if fast is None:
            fast = (
                "metal" not in bm3d_s
            )  # TODO: get a Intel Mac with AMD GPU to test. For Apple SoC `fast` should be False
        params = {
            "y_basic": _bm3d_presets["vbasic" if radius_Y > 0 else "basic"][
                preset_Y_basic
            ]
            | {"fast": fast},
            "y_final": _bm3d_presets["vfinal" if radius_Y > 0 else "final"][
                preset_Y_final
            ]
            | {"fast": fast},
            "chroma_basic": _bm3d_presets["vbasic" if radius_chroma > 0 else "basic"][
                preset_chroma_basic
            ]
            | {"fast": fast},
            "chroma_final": _bm3d_presets["vfinal" if radius_chroma > 0 else "final"][
                preset_chroma_final
            ]
            | {"fast": fast},
        }

    half_width = clip.width // 2
    half_height = clip.height // 2
    srcY_float, srcU_float, srcV_float = vstools.split(vstools.depth(clip, 32))

    if ref is None:
        basic_y = _bm3d.BM3Dv2(
            clip=srcY_float,
            ref=srcY_float,
            sigma=sigma_Y + delta_sigma_Y,
            radius=radius_Y,
            **params["y_basic"],
        )
    else:
        basic_y = vstools.depth(vstools.get_y(ref), 32)

    final_y = _bm3d.BM3Dv2(
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
        basic_half = _bm3d.BM3Dv2(
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

    final_half = _bm3d.BM3Dv2(
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
            search_clip=prefilter_to_full_range,
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
        ftype=4,
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


# modified from https://forum.doom9.org/showthread.php?p=1992601#post1992601
def remove_dirt(
    clip: vs.VideoNode, repmode: int = 16, remgrainmode: int = 17, limit: int = 10
) -> vs.VideoNode:
    cleansed = vsrgtools.clense(clip)
    sbegin = vsrgtools.clense(clip, mode=vsrgtools.rgtools.Clense.Mode.FORWARD)
    send = vsrgtools.clense(clip, mode=vsrgtools.rgtools.Clense.Mode.BACKWARD)
    scenechange = core.rdvs.SCSelect(clip, sbegin, send, cleansed)
    alt = vsrgtools.repair(scenechange, clip, mode=[repmode, repmode, 1])
    restore = vsrgtools.repair(cleansed, clip, mode=[repmode, repmode, 1])
    corrected = core.rdvs.RestoreMotionBlocks(
        cleansed,
        restore,
        neighbour=clip,
        alternative=alt,
        gmthreshold=70,
        dist=1,
        dmode=2,
        noise=limit,
        noisy=12,
    )
    return remove_grain(corrected, mode=[remgrainmode, remgrainmode, 1])


def immerkaer(
    clip: vs.VideoNode,
    planes: list[int] | int | None = None,
    prop_name: str = "NoiseSigma",
    gpu: bool = False,
) -> vs.VideoNode:
    r"""
    Estimate the noise standard deviation of a clip using Immerkaer's method.
    
    Noise strength (sigma) is defined as the standard deviation of the AWGN component, 
    implying $Noise \sim \mathcal{N}(0, \sigma^2)$ on a $0\text{-}255$ scale.

    Args:
        clip: The input video clip.
        planes: The index or list of indices of the planes to process.
            If None, all planes are processed.
        prop_name: The prefix for the generated frame property.
            The final property name will be f"{prop_name}{plane_index}"
            (e.g., "NoiseSigma0" for the Y plane).
        gpu: Whether to use GPU acceleration.

    Returns:
        The input clip with the estimated noise sigma attached as frame
        properties for the requested planes.

    Raises:
        ValueError: If a specified plane index is larger than the number of planes in the clip.
    
    Reference:
        John Immerkær, "Fast Noise Variance Estimation," Computer Vision and Image
        Understanding, vol. 64, no. 2, pp. 300-302, 1996.
        https://doi.org/10.1006/cviu.1996.0060
    """

    if planes is None:
        planes = list(range(0, clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if min(planes) < 0 or max(planes) >= clip.format.num_planes:
        raise ValueError("immerkaer: Plane index out of range.")

    conv_rpn = "x[-1,-1] x[1,-1] + x[-1,1] + x[1,1] + x[0,-1] x[-1,0] + x[1,0] + x[0,1] + -2 * + x 4 * + abs"

    expr_list = [
        conv_rpn if i in planes else "0" for i in range(clip.format.num_planes)
    ]

    conv = core.llvmexpr.VkExpr(clip, expr_list) if gpu else core.llvmexpr.Expr(clip, expr_list)

    conv = conv.std.Crop(left=2, right=2, top=2, bottom=2)

    prop = conv
    for i in planes:
        prop = core.std.PlaneStats(prop, plane=i, prop=f"Immerkaer{i}")

    assert prop

    sigma = core.llvmexpr.SingleExpr(
        prop,
        " ".join(
            [
                f"x.Immerkaer{i}Average pi 0.5 * sqrt * 6 / 255 * {prop_name}{i}$f"
                for i in planes
            ]
        ),
    )

    return core.std.CopyFrameProps(clip, sigma, [f"{prop_name}{i}" for i in planes])

def rmt_analyze(
    clip: vs.VideoNode,
    prop_name: str = "RMTSigma",
    patch_size: int = 10,
    n_patches: int = 2000,
    cutoff: float = 0.8,
    seed: int = 42,
) -> vs.VideoNode:
    r"""
    Estimate the noise standard deviation of a clip using Random Matrix Theory.
    
    Noise strength (sigma) is defined as the standard deviation of the AWGN component, 
    implying $Noise \sim \mathcal{N}(0, \sigma^2)$ on a $0\text{-}1$ scale. The estimation
    is performed by analyzing the eigenvalue distribution of the covariance matrix of
    random patches extracted from the frame and fitting it to the Marchenko-Pastur law.

    Args:
        clip: The input video clip. All planes will be processed.
        prop_name: The prefix for the generated frame property.
            The final property name will be f"{prop_name}{plane_index}"
            (e.g., "RMTSigma0" for the Y plane).
        patch_size: The size of the square patches used for analysis.
        n_patches: The number of random patches to extract from each frame.
        cutoff: The upper quantile of the eigenvalue distribution to consider for fitting.
        seed: The seed for the random number generator used for selecting patch locations.

    Returns:
        The input clip with the estimated noise sigma attached as frame
        properties for each plane.

    Raises:
        ValueError: If `cutoff` is not between 0 and 1.
        ValueError: If `patch_size` is not positive or is larger than the smallest plane dimension.
        ValueError: If `n_patches` is not positive or is greater than the number of available unique patches.
    """
    if not (0 < cutoff < 1):
        raise ValueError(f"rmt_analyze: cutoff must be between 0 and 1 (exclusive), but got {cutoff}")
    if patch_size <= 0:
        raise ValueError(f"rmt_analyze: patch_size must be a positive integer, but got {patch_size}")
    if n_patches <= 0:
        raise ValueError(f"rmt_analyze: n_patches must be a positive integer, but got {n_patches}")

    min_width = clip.width >> clip.format.subsampling_w
    min_height = clip.height >> clip.format.subsampling_h

    if patch_size >= min_width or patch_size >= min_height:
        raise ValueError(
            f"rmt_analyze: patch_size ({patch_size}) must be strictly smaller than the smallest plane's dimensions "
            f"({min_width}x{min_height})."
        )

    max_patches = (min_width - patch_size + 1) * (min_height - patch_size + 1)
    if n_patches > max_patches:
        raise ValueError(
            f"rmt_analyze: n_patches ({n_patches}) is greater than the maximum number of available unique patches "
            f"({max_patches}) for the smallest plane."
        )

    def _relu(x: Union[float, np.ndarray]):
        return (np.abs(x) + x) / 2

    def _indicator(
        x: np.ndarray,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        inclusive: str = "both",
    ) -> np.ndarray:
        if start is None and stop is None:
            raise ValueError("Error: provide start and/or stop for the indicator function.")
        left_inclusive = inclusive in {"both", "left"}
        right_inclusive = inclusive in {"both", "right"}

        if start is not None:
            left_condition = (x >= start) if left_inclusive else (x > start)
        else:
            left_condition = np.ones_like(x, dtype=bool)

        if stop is not None:
            right_condition = (x <= stop) if right_inclusive else (x < stop)
        else:
            right_condition = np.ones_like(x, dtype=bool)

        return np.where(left_condition & right_condition, 1.0, 0.0)

    class _mp_gen(rv_continuous):
        ARCTAN_OF_INFTY = np.pi / 2

        def __init__(
            self,
            momentum=0,
            a=0,
            b=None,
            xtol=1e-14,
            badvalue=None,
            name=None,
            shapes=None,
            seed=None,
        ):
            if shapes is None:
                shapes = "beta, sigma, ratio"
            super().__init__(momentum, a, b, xtol, badvalue, name, shapes, seed)

        @staticmethod
        def get_lambdas(beta, sigma, ratio):
            ratio = np.abs(ratio)
            sigma_sq = sigma**2
            lambda_minus = beta * sigma_sq * (1 - np.sqrt(ratio)) ** 2
            lambda_plus = beta * sigma_sq * (1 + np.sqrt(ratio)) ** 2
            return lambda_minus, lambda_plus

        @staticmethod
        def get_var(beta, sigma):
            return beta * sigma**2

        def _argcheck(self, beta, sigma, ratio):
            return (beta > 0) & (sigma > 0) & (ratio > 0) & (ratio <= 1.0)

        def _fitstart(self, x):
            x_clean = x[x > 1e-9]
            if len(x_clean) == 0:
                return 1.0, 0.1, 0.5, 0.0, 1.0

            lm, lp = np.quantile(x_clean, [0.05, 0.95])
            lm = max(lm, 1e-9)

            a = np.sqrt(lm)
            b = np.sqrt(lp)

            sigma_est = (a + b) / 2.0

            ratio_sqrt = (b - a) / (a + b)
            ratio_est = ratio_sqrt**2

            ratio_est = max(0.01, min(ratio_est, 0.99))
            sigma_est = max(1e-6, sigma_est)

            return 1.0, sigma_est, ratio_est, 0.0, 1.0

        def _pdf(self, x, beta, sigma, ratio):
            lambda_minus, lambda_plus = self.get_lambdas(beta, sigma, ratio)
            var = self.get_var(beta, sigma)

            with np.errstate(divide="ignore", invalid="ignore"):
                numerator = np.sqrt(_relu(lambda_plus - x) * _relu(x - lambda_minus))
                denominator = 2.0 * np.pi * ratio * var * x
                val = numerator / denominator

            return np.where(x < 1e-9, 0, val)

        def _cdf(self, x, beta, sigma, ratio):
            lambda_minus, lambda_plus = self.get_lambdas(beta, sigma, ratio)
            with np.errstate(divide="ignore", invalid="ignore"):
                acum = _indicator(x, start=lambda_plus, inclusive="left")
                mask = _indicator(
                    x, start=lambda_minus, stop=lambda_plus, inclusive="left"
                ).astype(bool)
                if np.any(mask):
                    acum[mask] += self._cdf_aux_f(x[mask], beta, sigma, ratio)

                return acum

        def _cdf_aux_f(self, x, beta, sigma, ratio):
            lambda_minus, lambda_plus = self.get_lambdas(beta, sigma, ratio)
            var = self.get_var(beta, sigma)
            cdf_aux_r = self._cdf_aux_r(x, lambda_minus, lambda_plus)

            first_term = np.arctan((cdf_aux_r**2 - 1) / (2 * cdf_aux_r))
            second_term = np.arctan(
                (lambda_minus * cdf_aux_r**2 - lambda_plus)
                / (2 * var * (1 - ratio) * cdf_aux_r)
            )

            first_term = np.nan_to_num(first_term, nan=self.ARCTAN_OF_INFTY)
            second_term = np.nan_to_num(second_term, nan=self.ARCTAN_OF_INFTY)

            result = (
                1
                / (2 * np.pi * ratio)
                * (
                    np.pi * ratio
                    + (1 / var) * np.sqrt(_relu(lambda_plus - x) * _relu(x - lambda_minus))
                    - (1 + ratio) * first_term
                    + (1 - ratio) * second_term
                )
            )
            return result

        def _cdf_aux_r(self, x, lambda_minus, lambda_plus):
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.sqrt((lambda_plus - x) / (x - lambda_minus))
            return np.nan_to_num(val, posinf=1e9)

    _marchenkopastur = _mp_gen(name="marchenkopastur")

    def _mp_fit_routine(eigenvalues: np.ndarray, cutoff_val: float):
        x = eigenvalues[np.isfinite(eigenvalues)]
        x = x[x > 1e-9]

        if len(x) < 20:
            return None

        if cutoff_val < 1.0:
            ub = np.quantile(x, cutoff_val)
        else:
            idx = int(min(len(x) - 1, cutoff_val))
            ub = x[idx]

        xp = x[x < ub]
        if len(xp) < 10:
            return None

        try:
            params = _marchenkopastur.fit(xp, fbeta=1.0, floc=0.0, fscale=1.0)
            return params
        except Exception:
            return None

    def _get_random_patches(
        img_plane: np.ndarray, patch_sz: int, n_pts: int, seed_val: int = 42
    ) -> np.ndarray:
        h, w = img_plane.shape
        sz = patch_sz

        valid_h = h - sz
        valid_w = w - sz

        if valid_h <= 0 or valid_w <= 0:
            return np.zeros((0, sz * sz))

        rng = np.random.default_rng(seed_val)
        r_indices = rng.integers(0, valid_h, size=n_pts)
        c_indices = rng.integers(0, valid_w, size=n_pts)

        patches = []
        for r, c in zip(r_indices, c_indices):
            patch = img_plane[r : r + sz, c : c + sz]
            flat_patch = patch.flatten().astype(np.float64)
            flat_patch -= np.mean(flat_patch)
            patches.append(flat_patch)

        return np.array(patches)

    max_val = get_peak_value_full(clip)

    def _estimator(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()

        for p in range(fout.format.num_planes):
            plane_array = np.asarray(fout[p])

            if max_val != 1.0:
                plane_norm = plane_array.astype(np.float64) / max_val
            else:
                plane_norm = plane_array.astype(np.float64)

            X = _get_random_patches(
                plane_norm, patch_size, n_patches, seed + n
            )
            N, M = X.shape

            sigma_result = 0.0

            if N >= M:
                try:
                    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
                    eigenvalues = (s**2) / N

                    params = _mp_fit_routine(eigenvalues, cutoff_val=cutoff)

                    if params:
                        estimated_sigma = params[1]
                        if np.isfinite(estimated_sigma) and estimated_sigma >= 0:
                            sigma_result = estimated_sigma
                except np.linalg.LinAlgError:
                    pass

            key = f"{prop_name}{p}"
            fout.props[key] = float(sigma_result) * 255.0

        return fout

    return vs.core.std.ModifyFrame(clip, clip, _estimator)