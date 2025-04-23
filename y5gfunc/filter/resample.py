import functools
import vstools
from vstools import vs
from vstools import core
from vstools import (
    depth,
    get_y,
    get_peak_value,
    get_lowest_value,
    ColorRange,
    Primaries,
    Transfer,
    join,
    scale_mask,
    get_prop,
)
from vsrgtools import box_blur
from typing import Any, Optional, Union, Callable
import sympy
from sympy import S, Rational, sqrt


class ColorMatrixManager:
    """Manager for a specific RGB-OPP conversion matrix and its related transformations."""

    def __init__(self, rgb_opp_matrix: sympy.Matrix, name: str):
        """
        Initialize the ColorMatrixManager with a specific RGB to OPP matrix.

        Args:
            rgb_opp_matrix: A 3x3 sympy.Matrix for RGB to OPP conversion
            name: Identifier for this specific OPP transform variant
        """
        self._rgb_opp_matrix = rgb_opp_matrix
        self._rgb_from_opp_matrix = rgb_opp_matrix.inv()
        self.name = name

        self._yuv_to_rgb_matrices: dict[vstools.Matrix, sympy.Matrix] = {}
        self._yuv_to_opp_matrices: dict[vstools.Matrix, sympy.Matrix] = {}
        self._opp_to_yuv_matrices: dict[vstools.Matrix, sympy.Matrix] = {}

        self._yuv_to_opp_coefs: dict[vstools.Matrix, list[float]] = {}
        self._opp_to_yuv_coefs: dict[vstools.Matrix, list[float]] = {}

        self._rgb_to_opp_coefs: Optional[list[float]] = None
        self._opp_to_rgb_coefs: Optional[list[float]] = None

        self._initialized = False

    def register_yuv_matrix(
        self,
        matrix_ids: Union[vstools.Matrix, list[vstools.Matrix]],
        yuv_to_rgb_matrix: sympy.Matrix,
    ) -> None:
        """
        Register a YUV to RGB matrix type.

        Args:
            matrix_id: The matrix id
            yuv_to_rgb_matrix: A sympy 3x3 Matrix representing YUV->RGB conversion
        """
        if isinstance(matrix_ids, vstools.Matrix):
            matrix_ids = [matrix_ids]

        for matrix_id in matrix_ids:
            self._yuv_to_rgb_matrices[matrix_id] = yuv_to_rgb_matrix
            if self._initialized:
                self._recalculate_for_matrix(matrix_id)

    def _matrix_to_coefs(self, matrix: sympy.Matrix) -> list[float]:
        """Convert a 3x3 matrix to fmtc.matrix compatible coefficients."""
        coefs = []
        for i in range(3):
            for j in range(3):
                coefs.append(float(sympy.N(matrix[i, j])))
            coefs.append(0.0)  # Bias term
        return coefs

    def _recalculate_for_matrix(self, matrix_id: vstools.Matrix) -> None:
        """Recalculate conversion matrices and coefficients for a specific matrix ID."""
        m_yuv_rgb = self._yuv_to_rgb_matrices[matrix_id]
        m_rgb_opp = self._rgb_opp_matrix
        m_rgb_from_opp = self._rgb_from_opp_matrix

        self._yuv_to_opp_matrices[matrix_id] = m_rgb_opp * m_yuv_rgb

        m_yuv_from_rgb = m_yuv_rgb.inv()
        self._opp_to_yuv_matrices[matrix_id] = m_yuv_from_rgb * m_rgb_from_opp

        self._yuv_to_opp_coefs[matrix_id] = self._matrix_to_coefs(
            self._yuv_to_opp_matrices[matrix_id]
        )
        self._opp_to_yuv_coefs[matrix_id] = self._matrix_to_coefs(
            self._opp_to_yuv_matrices[matrix_id]
        )

    def initialize(self) -> None:
        """Initialize all matrices and coefficients."""
        if self._initialized:
            return

        # Calculate direct RGB↔OPP coefficients
        self._rgb_to_opp_coefs = self._matrix_to_coefs(self._rgb_opp_matrix)
        self._opp_to_rgb_coefs = self._matrix_to_coefs(self._rgb_from_opp_matrix)

        # Calculate matrices and coefficients for all registered matrix types
        for matrix_type in self._yuv_to_rgb_matrices:
            self._recalculate_for_matrix(matrix_type)

        self._initialized = True

    def get_yuv_to_opp_coefs(self, matrix_type: vstools.Matrix) -> list[float]:
        """Get YUV to OPP conversion coefficients for a specific matrix type."""
        if not self._initialized:
            self.initialize()

        if matrix_type not in self._yuv_to_opp_coefs:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")

        return self._yuv_to_opp_coefs[matrix_type]

    def get_opp_to_yuv_coefs(self, matrix_type: vstools.Matrix) -> list[float]:
        """Get OPP to YUV conversion coefficients for a specific matrix type."""
        if not self._initialized:
            self.initialize()

        if matrix_type not in self._opp_to_yuv_coefs:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")

        return self._opp_to_yuv_coefs[matrix_type]

    def get_rgb_to_opp_coefs(self) -> list[float]:
        """Get RGB to OPP conversion coefficients."""
        if not self._initialized:
            self.initialize()

        return self._rgb_to_opp_coefs  # type: ignore

    def get_opp_to_rgb_coefs(self) -> list[float]:
        """Get OPP to RGB conversion coefficients."""
        if not self._initialized:
            self.initialize()

        return self._opp_to_rgb_coefs  # type: ignore


# Dictionary to store all OPP matrix managers by name
_opp_managers = {}


def _register_standard_matrices(manager: ColorMatrixManager) -> None:
    """
    Register standard YUV->RGB matrices to a matrix manager.

    Args:
        manager: The ColorMatrixManager to register matrices to
    """
    manager.register_yuv_matrix(
        vstools.Matrix.BT709,
        sympy.Matrix([[1.0, 0.0, 1.5748], [1.0, -0.1873, -0.4681], [1.0, 1.8556, 0.0]]),
    )

    manager.register_yuv_matrix(
        [
            vstools.Matrix.BT470BG,
            vstools.Matrix.SMPTE170M,
            vstools.Matrix.BT601_525,
            vstools.Matrix.BT601_625,
        ],
        sympy.Matrix(
            [[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]]
        ),
    )

    manager.register_yuv_matrix(
        [vstools.Matrix.BT2020NCL, vstools.Matrix.BT2020CL],
        sympy.Matrix(
            [[1.0, 0.0, 1.47493], [1.0, -0.16479, -0.57135], [1.0, 1.8814, 0.0]]
        ),
    )


def register_opp_matrix(name: str, rgb_opp_matrix: sympy.Matrix) -> ColorMatrixManager:
    """
    Register a new OPP matrix variant and create its manager.

    Args:
        name: Name identifier for this OPP variant
        rgb_opp_matrix: The 3x3 RGB to OPP conversion matrix

    Returns:
        The created ColorMatrixManager instance
    """
    if name in _opp_managers:
        raise ValueError(f"OPP matrix variant '{name}' already registered")

    manager = ColorMatrixManager(rgb_opp_matrix, name)
    _register_standard_matrices(manager)
    _opp_managers[name] = manager
    return manager


STANDARD_OPP = register_opp_matrix(
    "standard",
    sympy.Matrix(
        [
            [Rational(1, 3), Rational(1, 3), Rational(1, 3)],
            [Rational(1, 2), Rational(-1, 2), 0],
            [Rational(1, 4), Rational(1, 4), Rational(-1, 2)],
        ]
    ),
)

NORMALIZED_OPP = register_opp_matrix(
    "normalized",
    sympy.Matrix(
        [
            [Rational(1, 3), Rational(1, 3), Rational(1, 3)],
            [S(1) / sqrt(6), S(-1) / sqrt(6), 0],
            [S(1) / sqrt(18), S(1) / sqrt(18), S(-2) / sqrt(18)],
        ]
    ),
)

MAWEN_OPP = register_opp_matrix(
    "mawen",
    sympy.Matrix(
        [
            [Rational(1, 3), Rational(1, 3), Rational(1, 3)],
            [Rational(1, 2), 0, Rational(-1, 2)],
            [Rational(1, 4), Rational(-1, 2), Rational(1, 4)],
        ]
    ),
)

default_opp = STANDARD_OPP


def yuv2opp(
    clip: vs.VideoNode,
    matrix_in: Optional[vstools.Matrix] = None,
    range_in: Optional[ColorRange] = None,
    opp_manager: ColorMatrixManager = default_opp,
) -> vs.VideoNode:
    """
    Convert YUV to OPP color space.

    Args:
        clip: Input YUV clip
        matrix_in: Source matrix override, if None, read from clip properties
        range_in: Source range override, if None, read from clip properties
        opp_manager: ColorMatrixManager for the specific OPP variant to use

    Returns:
        Clip in OPP color space with appropriate frame properties
    """
    if matrix_in is None:
        matrix_in = get_prop(clip, "_Matrix", int, vstools.Matrix)
    if range_in is None:
        range_in = ColorRange.from_video(clip, strict=True)

    coef = opp_manager.get_yuv_to_opp_coefs(matrix_in)

    opp = core.fmtc.matrix(
        clip, fulls=not range_in, fulld=True, col_fam=vs.YUV, coef=coef
    )
    return opp.std.SetFrameProps(
        _Matrix=vs.MATRIX_UNSPECIFIED,
        BM3D_OPP=1,
        BM3D_OPP_VARIANT=opp_manager.name,
        BM3D_OPP_SRC_RANGE=range_in,
        BM3D_OPP_SRC_MATRIX=matrix_in,
    )


def opp2yuv(
    clip: vs.VideoNode,
    target_matrix: Optional[vstools.Matrix] = None,
    target_range: Optional[ColorRange] = None,
    opp_manager: Optional[ColorMatrixManager] = None,
) -> vs.VideoNode:
    """
    Convert OPP to YUV color space.

    Args:
        clip: Input OPP clip
        target_matrix: Target YUV color space, if None, read from clip properties
        target_range: Target range, if None, read from clip properties
        opp_manager: ColorMatrixManager for the specific OPP variant, if None, read from clip properties

    Returns:
        Clip in YUV color space
    """
    assert (
        get_prop(clip, "BM3D_OPP", int) == 1
    ), "Input must be an OPP clip (BM3D_OPP=1)"

    if opp_manager is None:
        variant_name = get_prop(clip, "BM3D_OPP_VARIANT", str, str)
        if variant_name not in _opp_managers:
            raise ValueError(f"Unknown OPP variant: {variant_name}")
        opp_manager = _opp_managers[variant_name]

    if target_matrix is None:
        target_matrix = get_prop(clip, "BM3D_OPP_SRC_MATRIX", int, vstools.Matrix)

    primaries = Primaries.from_matrix(target_matrix)
    transfer = Transfer.from_matrix(target_matrix)

    if target_range is None:
        target_range = get_prop(clip, "BM3D_OPP_SRC_RANGE", int, ColorRange)

    coef = opp_manager.get_opp_to_yuv_coefs(target_matrix)  # type: ignore

    yuv = core.fmtc.matrix(
        clip, fulls=True, fulld=not target_range, col_fam=vs.YUV, coef=coef
    )
    return yuv.std.SetFrameProps(
        _Matrix=target_matrix, _Primaries=primaries, _Transfer=transfer
    ).std.RemoveFrameProps(
        ["BM3D_OPP", "BM3D_OPP_VARIANT", "BM3D_OPP_SRC_MATRIX", "BM3D_OPP_SRC_RANGE"]
    )


def rgb2opp(
    clip: vs.VideoNode,
    opp_manager: ColorMatrixManager = default_opp,
) -> vs.VideoNode:
    """
    Convert RGB to OPP color space.

    Args:
        clip: Input RGB clip
        opp_manager: ColorMatrixManager for the specific OPP variant to use

    Returns:
        Clip in OPP color space
    """
    assert clip.format.color_family == vs.RGB, "Input must be in RGB format"

    coef = opp_manager.get_rgb_to_opp_coefs()

    opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.YUV, coef=coef)
    return opp.std.SetFrameProps(
        _Matrix=vs.MATRIX_UNSPECIFIED, BM3D_OPP=1, BM3D_OPP_VARIANT=opp_manager.name
    )


def opp2rgb(
    clip: vs.VideoNode,
    opp_manager: Optional[ColorMatrixManager] = None,
) -> vs.VideoNode:
    """
    Convert OPP to RGB color space.

    Args:
        clip: Input OPP clip
        opp_manager: ColorMatrixManager for the specific OPP variant, if None, read from clip properties

    Returns:
        Clip in RGB color space
    """
    assert (
        get_prop(clip, "BM3D_OPP", int) == 1
    ), "Input must be an OPP clip (BM3D_OPP=1)"

    if opp_manager is None:
        variant_name = get_prop(clip, "BM3D_OPP_VARIANT", str, str, "standard")
        if variant_name not in _opp_managers:
            raise ValueError(f"Unknown OPP variant: {variant_name}")
        opp_manager = _opp_managers[variant_name]

    coef = opp_manager.get_opp_to_rgb_coefs()  # type: ignore

    rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.RGB, coef=coef)
    return rgb.std.SetFrameProps(
        _Matrix=vstools.Matrix.RGB, _Transfer=Transfer.SRGB
    ).std.RemoveFrameProps(
        ["BM3D_OPP", "BM3D_OPP_VARIANT", "BM3D_OPP_SRC_MATRIX", "BM3D_OPP_SRC_RANGE"]
    )


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
