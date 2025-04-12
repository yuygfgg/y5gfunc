from vstools import vs
from vstools import core
from typing import Optional, Union, Callable


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
