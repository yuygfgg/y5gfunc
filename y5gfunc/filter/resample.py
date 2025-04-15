import functools
from vstools import vs
from vstools import core
from vstools import (
    depth,
    get_y,
    get_peak_value,
    get_lowest_value,
    ColorRange,
    join,
    scale_mask,
)
from vsrgtools import box_blur
from typing import Any, Optional, Union, Callable


# modified from yvsfunc
def rgb2opp(clip: vs.VideoNode) -> vs.VideoNode:
    coef = [1 / 3, 1 / 3, 1 / 3, 0, 1 / 2, -1 / 2, 0, 0, 1 / 4, 1 / 4, -1 / 2, 0]
    opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.YUV, coef=coef)
    opp = core.std.SetFrameProps(opp, _Matrix=vs.MATRIX_UNSPECIFIED, BM3D_OPP=1)
    return opp


# modified from yvsfunc
def opp2rgb(clip: vs.VideoNode) -> vs.VideoNode:
    coef = [1, 1, 2 / 3, 0, 1, -1, 2 / 3, 0, 1, 0, -4 / 3, 0]
    rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.RGB, coef=coef)
    rgb = core.std.SetFrameProps(rgb, _Matrix=vs.MATRIX_RGB)
    rgb = core.std.RemoveFrameProps(rgb, "BM3D_OPP")
    return rgb


def nn2x(
    clip: vs.VideoNode,
    opencl: bool = True,
    nnedi3_args: dict[str, int] = {"field": 1, "nsize": 4, "nns": 4, "qual": 2},
) -> vs.VideoNode:
    """Doubles the resolution of input clip with nnedi3."""
    if hasattr(core, "sneedif") and opencl:
        nnedi3 = functools.partial(core.sneedif.NNEDI3, **nnedi3_args)
        return nnedi3(nnedi3(clip, dh=True), dw=True)
    elif hasattr(core, "nnedi3cl") and opencl:
        nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)
        return nnedi3(nnedi3(clip, dh=True), dw=True)
    else:
        nnedi3 = functools.partial(core.nnedi3.nnedi3, **nnedi3_args)
        return nnedi3(nnedi3(clip, dh=True).std.Transpose(), dh=True).std.Transpose()


# modified from rksfunc
def Gammarize(clip: vs.VideoNode, gamma, tvrange=False) -> vs.VideoNode:
    if clip.format.name.startswith("YUV"):
        is_yuv = True
        y = get_y(clip)
    elif clip.format.name.startswith("Gray"):
        is_yuv = False
        y = clip
    else:
        raise ValueError("Gammarize: Input clip must be either YUV or GRAY.")

    range = ColorRange.LIMITED if tvrange else ColorRange.FULL

    thrl = get_lowest_value(clip, range_in=range)
    thrh = get_peak_value(clip, range_in=range)
    rng = scale_mask(219, 8, clip) if tvrange else scale_mask(255, 8, clip)
    corrected = y.akarin.Expr(
        f"x {rng} / {gamma} pow {rng} * {thrl} + {thrl} max {thrh} min"
    )

    return join(corrected, clip) if is_yuv else corrected


# modified from muvsfunc
def SSIM_downsample(
    clip: vs.VideoNode,
    width: int,
    height: int,
    smooth: Union[float, Callable] = 1,
    gamma: bool = True,
    fulls: bool = False,
    fulld: bool = False,
    curve: str = "709",
    sigmoid: bool = True,
    epsilon: float = 1e-6,
    **rersample_args: Any,
) -> vs.VideoNode:
    """
    SSIM downsampler

    SSIM downsampler is an image downscaling technique that aims to optimize for the perceptual quality of the downscaled results.
    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured using famous Structural SIMilarity (SSIM) index.
    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.

    All the internal calculations are done at 32-bit float, except gamma correction is done at integer.

    Args:
        clip: The input clip.
        width: The width of the output clip.
        height: The height of the output clip
        smooth: The method to smooth the image.
            If it's an int, it specifies the "radius" of the internel used boxfilter, i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
            If it's a float, it specifies the "sigma" of core.tcanny.TCanny, i.e. the standard deviation of gaussian blur.
            If it's a function, it acs as a general smoother.
        gamma: Set to true to turn on gamma correction for the y channel.
        fulls: Specifies if the luma is limited range (False) or full range (True)
        fulld: Same as fulls, but for output.
        curve: Type of gamma mapping.
        sigmoid: When True, applies a sigmoidal curve after the power-like curve (or before when converting from linear to gamma-corrected).
            This helps reducing the dark halo artefacts around sharp edges caused by resizing in linear luminance.
        resample_args: Additional arguments passed to `core.resize2` in the form of keyword arguments.

    Returns:
        Downsampled clip in 32-bit format.

    Raises:
        TypeError: If `smooth` is neigher a int, float nor a function.

    Ref:
        [1] Oeztireli, A. C., & Gross, M. (2015). Perceptually based downscaling of images. ACM Transactions on Graphics (TOG), 34(4), 77.

    """
    if callable(smooth):
        Filter = smooth
    elif isinstance(smooth, int):
        Filter = functools.partial(box_blur, radius=smooth + 1)
    elif isinstance(smooth, float):
        Filter = functools.partial(core.tcanny.TCanny, sigma=smooth, mode=-1)
    else:
        raise TypeError('SSIM_downsample: "smooth" must be a int, float or a function!')

    if gamma:
        import nnedi3_resample as nnrs

        clip = nnrs.GammaToLinear(
            depth(clip, 16),
            fulls=fulls,
            fulld=fulld,
            curve=curve,
            sigmoid=sigmoid,
            planes=[0],
        )

    clip = depth(clip, 32)

    l1 = core.resize2.Bicubic(clip, width, height, **rersample_args)  # type: ignore
    l2 = core.resize2.Bicubic(
        core.akarin.Expr([clip], ["x 2 **"]),
        width,
        height,
        **rersample_args,  # type: ignore
    )

    m = Filter(l1)
    sl_plus_m_square = Filter(core.akarin.Expr([l1], ["x 2 **"]))
    sh_plus_m_square = Filter(l2)
    m_square = core.akarin.Expr([m], ["x 2 **"])
    r = core.akarin.Expr(
        [sl_plus_m_square, sh_plus_m_square, m_square],
        [
            f"x z - {epsilon} < 0 y z - x z - / sqrt ?"
        ],  # akarin.Expr adds "0 max" to sqrt by default
    )
    t = Filter(core.akarin.Expr([r, m], ["x y *"]))
    m = Filter(m)
    r = Filter(r)
    d = core.akarin.Expr([m, r, l1, t], ["x y z * + a -"])

    if gamma:
        d = nnrs.LinearToGamma(
            depth(d, 16),
            fulls=fulls,
            fulld=fulld,
            curve=curve,
            sigmoid=sigmoid,
            planes=[0],
        )

    return depth(d, 32)


def Descale(
    src: vs.VideoNode,
    width: int,
    height: int,
    kernel: str,
    custom_kernel: Optional[Callable] = None,
    taps: int = 3,
    b: Union[int, float] = 0.0,
    c: Union[int, float] = 0.5,
    blur: Union[int, float] = 1.0,
    post_conv: Optional[list[Union[float, int]]] = None,
    src_left: Union[int, float] = 0.0,
    src_top: Union[int, float] = 0.0,
    src_width: Optional[Union[int, float]] = None,
    src_height: Optional[Union[int, float]] = None,
    border_handling: int = 0,
    ignore_mask: Optional[vs.VideoNode] = None,
    force: bool = False,
    force_h: bool = False,
    force_v: bool = False,
    opt: int = 0,
) -> vs.VideoNode:
    def _get_resize_name(kernal_name: str) -> str:
        if kernal_name == "Decustom":
            return "ScaleCustom"
        if kernal_name.startswith("De"):
            return kernal_name[2:].capitalize()
        return kernal_name

    def _get_descaler_name(kernal_name: str) -> str:
        if kernal_name == "ScaleCustom":
            return "Decustom"
        if kernal_name.startswith("De"):
            return kernal_name
        return "De" + kernal_name[0].lower() + kernal_name[1:]

    assert width > 0 and height > 0
    assert opt in [0, 1, 2]
    assert isinstance(src, vs.VideoNode) and src.format.id == vs.GRAYS

    kernel = kernel.capitalize()

    if src_width is None:
        src_width = width
    if src_height is None:
        src_height = height

    if width > src.width or height > src.height:
        kernel = _get_resize_name(kernel)
    else:
        kernel = _get_descaler_name(kernel)

    descaler = getattr(core.descale, kernel)
    assert callable(descaler)
    extra_params: dict[str, dict[str, Union[float, int, Callable]]] = {}
    if _get_descaler_name(kernel) == "Debicubic":
        extra_params = {
            "dparams": {"b": b, "c": c},
        }
    elif _get_descaler_name(kernel) == "Delanczos":
        extra_params = {
            "dparams": {"taps": taps},
        }
    elif _get_descaler_name(kernel) == "Decustom":
        assert callable(custom_kernel)
        extra_params = {
            "dparams": {"custom_kernel": custom_kernel},
        }
    descaled = descaler(
        src=src,
        width=width,
        height=height,
        blur=blur,
        post_conv=post_conv,
        src_left=src_left,
        src_top=src_top,
        src_width=src_width,
        src_height=src_height,
        border_handling=border_handling,
        ignore_mask=ignore_mask,
        force=force,
        force_h=force_h,
        force_v=force_v,
        opt=opt,
        **extra_params.get("dparams", {}),
    )

    assert isinstance(descaled, vs.VideoNode)

    return descaled
