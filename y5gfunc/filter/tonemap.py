from enum import IntEnum
from typing import Optional, Union
from vstools import vs
from vstools import core
import vstools

class ColorSpace(IntEnum):
    """
    Color spaces used in tonemapping operations.

    Attributes:
        SDR: Standard Dynamic Range
        HDR10: High Dynamic Range (HDR10)
        HLG: Hybrid Log-Gamma
        DOLBY_VISION: Dolby Vision
    """

    SDR = 0
    HDR10 = 1
    HLG = 2
    DOLBY_VISION = 3


class ColorPrimaries(IntEnum):
    """
    Color primaries standards.

    Attributes:
        UNKNOWN: Unknown primaries
        BT601_525: ITU-R Rec. BT.601 (525-line = NTSC, SMPTE-C)
        BT601_625: ITU-R Rec. BT.601 (625-line = PAL, SECAM)
        BT709: ITU-R Rec. BT.709 (HD), also sRGB
        BT470M: ITU-R Rec. BT.470 M
        EBU_TECH_3213E: EBU Tech. 3213-E / JEDEC P22 phosphors
        BT2020: ITU-R Rec. BT.2020 (UltraHD)
        APPLE_RGB: Apple RGB
        ADOBE_RGB: Adobe RGB (1998)
        PROPHOTO_RGB: ProPhoto RGB (ROMM)
        CIE1931_RGB: CIE 1931 RGB primaries
        DCI_P3: DCI-P3 (Digital Cinema)
        DCI_P3_D65: DCI-P3 (Digital Cinema) with D65 white point
        V_GAMUT: Panasonic V-Gamut (VARICAM)
        S_GAMUT: Sony S-Gamut
        FILM_C: Traditional film primaries with Illuminant C
        ACES0: ACES Primaries #0 (ultra wide)
        ACES1: ACES Primaries #1
    """

    UNKNOWN = 0
    BT601_525 = 1
    BT601_625 = 2
    BT709 = 3
    BT470M = 4
    EBU_TECH_3213E = 5
    BT2020 = 6
    APPLE_RGB = 7
    ADOBE_RGB = 8
    PROPHOTO_RGB = 9
    CIE1931_RGB = 10
    DCI_P3 = 11
    DCI_P3_D65 = 12
    V_GAMUT = 13
    S_GAMUT = 14
    FILM_C = 15
    ACES0 = 16
    ACES1 = 17


class GamutMapping(IntEnum):
    """
    Gamut mapping functions.

    Attributes:
        CLIP: Performs no gamut-mapping, just hard clips out-of-range colors per-channel
        PERCEPTUAL: Performs a perceptually balanced (saturation) gamut mapping with a soft knee function
        SOFTCLIP: Performs perceptually balanced gamut mapping using a soft knee function and hue shifting
        RELATIVE: Performs relative colorimetric clipping, maintaining exponential brightness/chromaticity relationship
        SATURATION: Performs simple RGB->RGB saturation mapping; never clips but may distort hues
        ABSOLUTE: Performs absolute colorimetric clipping without adapting white point
        DESATURATE: Performs constant-luminance colorimetric clipping, desaturating colors towards white
        DARKEN: Uniformly darkens the input to prevent highlight clipping, then clamps colorimetrically
        HIGHLIGHT: Performs no gamut mapping, but highlights out-of-gamut pixels
        LINEAR: Linearly/uniformly desaturates the image to bring it into the target gamut
    """

    CLIP = 0
    PERCEPTUAL = 1
    SOFTCLIP = 2
    RELATIVE = 3
    SATURATION = 4
    ABSOLUTE = 5
    DESATURATE = 6
    DARKEN = 7
    HIGHLIGHT = 8
    LINEAR = 9


class ToneMappingFunction(IntEnum):
    """
    Tone mapping functions.

    Attributes:
        CLIP: No tone-mapping, just clips out-of-range colors
        SPLINE: Simple spline consisting of two polynomials joined by a pivot point
        ST2094_40: EETF from SMPTE ST 2094-40 Annex B, using Bezier curves
        ST2094_10: EETF from SMPTE ST 2094-10 Annex B.2
        BT2390: EETF from ITU-R Report BT.2390, a hermite spline roll-off with linear segment
        BT2446A: EETF from ITU-R Report BT.2446, method A
        REINHARD: Simple non-linear curve named after Erik Reinhard
        MOBIUS: Generalization of reinhard algorithm (legacy/low-quality)
        HABLE: Piece-wise, filmic tone-mapping algorithm by John Hable (legacy/low-quality)
        GAMMA: Fits a gamma (power) function between source and target (legacy/low-quality)
        LINEAR: Linearly stretches input range to output range in PQ space
        LINEARLIGHT: Like LINEAR but in linear light instead of PQ
    """

    CLIP = 0
    SPLINE = 1
    ST2094_40 = 2
    ST2094_10 = 3
    BT2390 = 4
    BT2446A = 5
    REINHARD = 6
    MOBIUS = 7
    HABLE = 8
    GAMMA = 9
    LINEAR = 10
    LINEARLIGHT = 11


class Metadata(IntEnum):
    """
    Metadata sources for tone-mapping.

    Attributes:
        AUTO: Automatic selection based on available metadata
        NONE: No metadata (disabled)
        HDR10: HDR10 static metadata
        HDR10_PLUS: HDR10+ (MaxRGB) dynamic metadata
        LUMINANCE: Luminance (CIE Y) derived metadata
    """

    AUTO = 0
    NONE = 1
    HDR10 = 2
    HDR10_PLUS = 3
    LUMINANCE = 4


def tonemap(
    clip: vs.VideoNode,
    src_csp: ColorSpace,
    dst_csp: ColorSpace,
    dst_prim: Optional[Union[ColorPrimaries, vstools.Primaries]] = None,
    src_max: Optional[Union[float, int]] = None,
    src_min: Optional[Union[float, int]] = None,
    dst_max: Optional[Union[float, int]] = None,
    dst_min: Optional[Union[float, int]] = None,
    dynamic_peak_detection: bool = True,
    smoothing_period: Union[float, int] = 20.0,
    scene_threshold_low: Union[float, int] = 1.0,
    scene_threshold_high: Union[float, int] = 3.0,
    percentile: Union[float, int] = 99.995,
    gamut_mapping: GamutMapping = GamutMapping.PERCEPTUAL,
    tone_mapping_function: ToneMappingFunction = ToneMappingFunction.SPLINE,
    tone_mapping_param: Optional[Union[float, int]] = None,
    metadata: Metadata = Metadata.AUTO,
    use_dovi: Optional[bool] = None,
    visualize_lut: bool = False,
    show_clipping: bool = False,
    contrast_recovery: Union[float, int] = 0.3,
    log_level: int = 2,
) -> vs.VideoNode:
    """
    Color mapping using placebo.Tonemap.

    Args:
        clip: Input video clip
        src_csp: Source colorspace (SDR, HDR10, HLG, DOLBY_VISION)
        dst_csp: Destination colorspace (SDR, HDR10, HLG, DOLBY_VISION)
        dst_prim: Destination color primaries
        src_max: Source maximum display level in nits (cd/m²)
        src_min: Source minimum display level in nits (cd/m²)
        dst_max: Destination maximum display level in nits (cd/m²)
        dst_min: Destination minimum display level in nits (cd/m²)
        dynamic_peak_detection: Enables HDR peak detection
        smoothing_period: Smoothing coefficient for detected values (in frames)
        scene_threshold_low: Lower bound for scene change detection (in units of 1% PQ)
        scene_threshold_high: Upper bound for scene change detection (in units of 1% PQ)
        percentile: Percentile of brightness histogram to consider as true peak
        gamut_mapping: Gamut mapping function to handle out-of-gamut colors
        tone_mapping_function: Tone mapping function
        tone_mapping_param: Optional parameter for tone mapping function
        metadata: Data source to use when tone-mapping
        use_dovi: Whether to use Dolby Vision RPU for ST2086 metadata
        visualize_lut: Display a (PQ-PQ) graph of the active tone-mapping LUT
        show_clipping: Highlight hard-clipped pixels during tone-mapping
        contrast_recovery: HDR contrast recovery strength
        log_level: Logging verbosity level

    Returns:
        Tonemapped video clip
    """
    src_csp_int = int(src_csp)
    dst_csp_int = int(dst_csp)
    
    if isinstance(dst_prim, ColorPrimaries):
        dst_prim_int = int(dst_prim)
    elif isinstance(dst_prim, vstools.Primaries):
        dst_prim_int = dst_prim.value_libplacebo
    else:
        dst_prim_int = None
    
    gamut_mapping_int = int(gamut_mapping)
    metadata_int = int(metadata)

    tone_mapping_function_int = int(tone_mapping_function)

    return core.placebo.Tonemap(
        clip=clip,
        src_csp=src_csp_int,
        dst_csp=dst_csp_int,
        dst_prim=dst_prim_int,  # type: ignore
        src_max=src_max,  # type: ignore
        src_min=src_min,  # type: ignore
        dst_max=dst_max,  # type: ignore
        dst_min=dst_min,  # type: ignore
        dynamic_peak_detection=dynamic_peak_detection,
        smoothing_period=smoothing_period,
        scene_threshold_low=scene_threshold_low,
        scene_threshold_high=scene_threshold_high,
        percentile=percentile,
        gamut_mapping=gamut_mapping_int,
        tone_mapping_function=tone_mapping_function_int,
        tone_mapping_param=tone_mapping_param,  # type: ignore
        metadata=metadata_int,
        use_dovi=use_dovi,  # type: ignore
        visualize_lut=visualize_lut,
        show_clipping=show_clipping,
        contrast_recovery=contrast_recovery,
        log_level=log_level,
    )
