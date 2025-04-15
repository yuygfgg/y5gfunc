from vstools import vs
from vstools import core
import mvsfunc as mvf
import vstools
from typing import Callable, Union
from enum import StrEnum
from .resample import rgb2opp, opp2rgb


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

    Returns:
        Denoised video clip in YUV420P16 format.

    Raises:
        ValueError: If the input clip format is not YUV420P16.
        ValueError: If any provided preset name is invalid.

    Notes:
        - The `delta_sigma_xxx` value is added to `sigma_xxx` only for the 'basic' denoising step. The 'final' step uses `sigma_xxx` directly.
    """

    if clip.format.id != vs.YUV420P16:
        raise ValueError("Fast_BM3DWrapper: Input clip format must be YUV420P16.")

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

    params = {
        "y_basic": bm3d_presets["vbasic" if radius_Y > 0 else "basic"][preset_Y_basic],
        "y_final": bm3d_presets["vfinal" if radius_Y > 0 else "final"][preset_Y_final],
        "chroma_basic": bm3d_presets["vbasic" if radius_chroma > 0 else "basic"][
            preset_chroma_basic
        ],
        "chroma_final": bm3d_presets["vfinal" if radius_chroma > 0 else "final"][
            preset_chroma_final
        ],
    }

    half_width = clip.width // 2  # half width
    half_height = clip.height // 2  # half height
    srcY_float, srcU_float, srcV_float = vstools.split(vstools.depth(clip, 32))

    basic_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=srcY_float,
        sigma=sigma_Y + delta_sigma_Y,
        radius=radius_Y,
        **params["y_basic"],
    )

    final_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=basic_y,
        sigma=sigma_Y,
        radius=radius_Y,
        **params["y_final"],
    )

    vyhalf = final_y.resize2.Spline36(half_width, half_height, src_left=-0.5)
    srchalf_444 = vstools.join([vyhalf, srcU_float, srcV_float])
    srchalf_opp = rgb2opp(
        mvf.ToRGB(input=srchalf_444, depth=32, matrix="709", sample=vs.FLOAT)
    )

    basic_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=srchalf_opp,
        sigma=sigma_chroma + delta_sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0,
        **params["chroma_basic"],
    )

    final_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=basic_half,
        sigma=sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0,
        **params["chroma_final"],
    )

    final_half = opp2rgb(final_half).resize2.Spline36(format=vs.YUV444PS, matrix=1)
    _, final_u, final_v = vstools.split(final_half)
    vfinal = vstools.join([final_y, final_u, final_v])
    return vstools.depth(vfinal, 16)

