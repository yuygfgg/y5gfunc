import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import vsutil
from typing import Literal, Union

# modified from yvsfunc
def _rgb2opp(clip: vs.VideoNode) -> vs.VideoNode:
    coef = [1/3, 1/3, 1/3, 0, 1/2, -1/2, 0, 0, 1/4, 1/4, -1/2, 0]
    opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.YUV, coef=coef)
    opp = core.std.SetFrameProps(opp, _Matrix=vs.MATRIX_UNSPECIFIED, BM3D_OPP=1)
    return opp

# modified from yvsfunc
def _opp2rgb(clip: vs.VideoNode) -> vs.VideoNode:
    coef = [1, 1, 2/3, 0, 1, -1, 2/3, 0, 1, 0, -4/3, 0]
    rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.RGB, coef=coef)
    rgb = core.std.SetFrameProps(rgb, _Matrix=vs.MATRIX_RGB)
    rgb = core.std.RemoveFrameProps(rgb, 'BM3D_OPP')
    return rgb

# modified from https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D?tab=readme-ov-file#profile-default
# preset 'magic' is from rksfunc
bm3d_presets = {
    "basic": {
        "fast":  {"block_step": 8, "bm_range": 9,  "ps_num": 2, "ps_range": 4},
        "lc":    {"block_step": 6, "bm_range": 9,  "ps_num": 2, "ps_range": 4},
        "np":    {"block_step": 4, "bm_range": 16, "ps_num": 2, "ps_range": 5},
        "high":  {"block_step": 3, "bm_range": 16, "ps_num": 2, "ps_range": 7},
        "magic": {"block_step": 3, "bm_range": 12, "ps_num": 2, "ps_range": 8},
    },
    "vbasic": {
        "fast":  {"block_step": 8, "bm_range": 7,  "ps_num": 2, "ps_range": 4},
        "lc":    {"block_step": 6, "bm_range": 9,  "ps_num": 2, "ps_range": 4},
        "np":    {"block_step": 4, "bm_range": 12, "ps_num": 2, "ps_range": 5},
        "high":  {"block_step": 3, "bm_range": 16, "ps_num": 2, "ps_range": 7},
        "magic": {"block_step": 3, "bm_range": 12, "ps_num": 2, "ps_range": 8},
    },
    "final": {
        "fast":  {"block_step": 7, "bm_range": 9,  "ps_num": 2, "ps_range": 5},
        "lc":    {"block_step": 5, "bm_range": 9,  "ps_num": 2, "ps_range": 5},
        "np":    {"block_step": 3, "bm_range": 16, "ps_num": 2, "ps_range": 6},
        "high":  {"block_step": 2, "bm_range": 16, "ps_num": 2, "ps_range": 8},
        "magic": {"block_step": 2, "bm_range": 8,  "ps_num": 2, "ps_range": 6},
    },
    "vfinal": {
        "fast":  {"block_step": 7, "bm_range": 7,  "ps_num": 2, "ps_range": 5},
        "lc":    {"block_step": 5, "bm_range": 9,  "ps_num": 2, "ps_range": 5},
        "np":    {"block_step": 3, "bm_range": 12, "ps_num": 2, "ps_range": 6},
        "high":  {"block_step": 2, "bm_range": 16, "ps_num": 2, "ps_range": 8},
        "magic": {"block_step": 2, "bm_range": 8,  "ps_num": 2, "ps_range": 6},
    }
}

# modified from rksfunc.BM3DWrapper()
def Fast_BM3DWrapper(
    clip: vs.VideoNode,
    bm3d=core.bm3dcpu,
    chroma: bool = True,
    
    sigma_Y: Union[float, int] = 1.2,
    radius_Y: int = 1,
    delta_sigma_Y: Union[float, int] = 0.6,
    preset_Y_basic: Literal["fast", "lc", "np", "high", "magic"] = "fast",
    preset_Y_final: Literal["fast", "lc", "np", "high", "magic"] = "fast",
    
    sigma_chroma: Union[float, int] = 2.4,
    radius_chroma: int = 0,
    delta_sigma_chroma: Union[float, int] = 1.2,
    preset_chroma_basic: Literal["fast", "lc", "np", "high", "magic"] = "fast",
    preset_chroma_final: Literal["fast", "lc", "np", "high", "magic"] = "fast",
) -> vs.VideoNode:

    '''
    Note: delta_sigma_xxx is added to sigma_xxx in step basic.
    '''

    assert clip.format.id == vs.YUV420P16

    assert all(preset in ["fast", "lc", "np", "high", "magic"] for preset in [preset_Y_basic, preset_Y_final, preset_chroma_basic, preset_chroma_final])

    params = {
        "y_basic": bm3d_presets["vbasic" if radius_Y > 0 else "basic"][preset_Y_basic],
        "y_final": bm3d_presets["vfinal" if radius_Y > 0 else "final"][preset_Y_final],
        "chroma_basic": bm3d_presets["vbasic" if radius_chroma > 0 else "basic"][preset_chroma_basic],
        "chroma_final": bm3d_presets["vfinal" if radius_chroma > 0 else "final"][preset_chroma_final],
    }

    half_width = clip.width // 2  # half width
    half_height = clip.height // 2  # half height
    srcY_float, srcU_float, srcV_float = vsutil.split(vsutil.depth(clip, 32))

    basic_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=srcY_float,
        sigma=sigma_Y + delta_sigma_Y,
        radius=radius_Y,
        **params["y_basic"]
    )

    final_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=basic_y,
        sigma=sigma_Y,
        radius=radius_Y,
        **params["y_final"]
    )
    
    vyhalf = final_y.resize2.Spline36(half_width, half_height, src_left=-0.5)
    srchalf_444 = vsutil.join([vyhalf, srcU_float, srcV_float])
    srchalf_opp = _rgb2opp(mvf.ToRGB(input=srchalf_444, depth=32, matrix="709", sample=1))

    basic_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=srchalf_opp,
        sigma=sigma_chroma + delta_sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0,
        **params["chroma_basic"]
    )

    final_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=basic_half,
        sigma=sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0,
        **params["chroma_final"]
    )

    final_half = _opp2rgb(final_half).resize2.Spline36(format=vs.YUV444PS, matrix=1)
    _, final_u, final_v = vsutil.split(final_half)
    vfinal = vsutil.join([final_y, final_u, final_v])
    return vsutil.depth(vfinal, 16)
