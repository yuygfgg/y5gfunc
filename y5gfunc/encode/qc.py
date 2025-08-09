from typing import Union, Optional
from vstools import vs
from vstools import core
import vstools
from enum import StrEnum
from ..utils import PickFrames
from muvsfunc import SSIM


class QcMode(StrEnum):
    """Which metrics to use for quality check"""

    SSIM = "SSIM"
    CAMBI = "CAMBI"
    BOTH = "BOTH"


class ReturnType(StrEnum):
    """
    What to return after quality check

    Attributes:
        ENCODED: Return the `encoded` clip with frame properties added (CAMBI, PlaneSSIM, *_err flags)
        ERROR: Return a clip containing only the frames flagged as errors
        BOTH: Return a tuple containing both the annotated `encoded` clip and the `error` clip.
    """

    ENCODED = "encoded"
    ERROR = "error"
    BOTH = "both"


def encode_check(
    encoded: vs.VideoNode,
    source: Optional[vs.VideoNode] = None,
    mode: QcMode = QcMode.BOTH,
    threshold_cambi: float = 5,
    threshold_ssim: float = 0.9,
    return_type: ReturnType = ReturnType.ENCODED,
) -> Union[vs.VideoNode, tuple[vs.VideoNode, vs.VideoNode]]:
    """
    Perform a quality check on an encoded video using SSIM and/or CAMBI metrics.

    This function compares the encoded video against optional source video (for SSIM) and calculates CAMBI scores. It identifies frames
    that fall below the SSIM threshold or exceed the CAMBI threshold, printing information about problematic frames to the console.

    Args:
        encoded: The encoded video clip to check.
        source: The source video clip for SSIM comparison. Required if mode includes SSIM. Must have the same format as `encoded`.
        mode: Which metrics to use.
        threshold_cambi: The maximum allowed CAMBI score. Frames exceeding this are flagged. Must be between 0 and 24.
        threshold_ssim: The minimum allowed PlaneSSIM score. Frames below this are flagged. Must be between 0 and 1.
        return_type: What to return:

    Returns:
        Depending on `return_type`, either the annotated encoded clip, a clip with only error frames, or a tuple containing both.

    Raises:
        AssertionError: If input parameters are invalid.
    """

    assert 0 <= threshold_cambi <= 24
    assert 0 <= threshold_ssim <= 1

    if mode == QcMode.BOTH:
        enable_ssim = enable_cambi = True
    elif mode == QcMode.SSIM:
        enable_ssim = True
        enable_cambi = False
    else:
        enable_ssim = False
        enable_cambi = True

    if enable_ssim:
        assert source
        assert encoded.format.id == source.format.id
        ssim = SSIM(encoded, source)

    if enable_cambi:
        cambi = core.cambi.Cambi(
            (
                encoded
                if vstools.get_depth(encoded) <= 10
                else vstools.depth(encoded, 10, dither_type="none")
            ),
            prop="CAMBI",
        )

    error_frames = []

    def _chk(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        def print_red_bold(text) -> None:
            print("\033[1;31m" + text + "\033[0m")

        fout = f[0].copy()

        ssim_err = cambi_err = False

        if enable_ssim:
            fout.props["PlaneSSIM"] = ssim_val = f[2].props["PlaneSSIM"]
            fout.props["ssim_err"] = ssim_err = (
                1 if threshold_ssim > f[2].props["PlaneSSIM"] else 0  # type: ignore
            )

        if enable_cambi:
            fout.props["CAMBI"] = cambi_val = f[1].props["CAMBI"]
            fout.props["cambi_err"] = cambi_err = (
                1 if threshold_cambi < f[1].props["CAMBI"] else 0  # type: ignore
            )

        if cambi_err and enable_cambi:
            print_red_bold(
                f"frame {n}: Banding detected! CAMBI: {cambi_val}"
                f"    Note: banding threshold is {threshold_cambi}"
            )
        if ssim_err and enable_ssim:
            print_red_bold(
                f"frame {n}: Distortion detected! SSIM: {ssim_val}"
                f"    Note: distortion threshold is {threshold_ssim}"
            )
        if not (cambi_err or ssim_err):
            print(f"Frame {n}: OK!")
        else:
            error_frames.append(n)

        return fout

    if enable_ssim and enable_cambi:
        output = core.std.ModifyFrame(encoded, [encoded, cambi, ssim], _chk)
    elif enable_cambi:
        output = core.std.ModifyFrame(encoded, [encoded, cambi, cambi], _chk)
    else:
        output = core.std.ModifyFrame(encoded, [encoded, ssim, ssim], _chk)

    if return_type == ReturnType.ENCODED:
        return output

    for _ in output.frames():
        pass

    err = PickFrames(encoded, error_frames)

    if return_type == ReturnType.BOTH:
        return output, err
    else:
        return err
