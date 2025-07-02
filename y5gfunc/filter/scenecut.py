from typing import Callable
from vstools import core
from vstools import vs
import numpy as np
from functools import partial
from ..filter import AnimeMask
import muvsfunc


def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """cv2.HISTCMP_CORREL"""
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()

    x_mean = np.mean(hist1)
    y_mean = np.mean(hist2)

    numerator = np.sum((hist1 - x_mean) * (hist2 - y_mean))
    denominator = np.sqrt(np.sum((hist1 - x_mean) ** 2) * np.sum((hist2 - y_mean) ** 2))

    if denominator == 0:
        return 0
    return numerator / denominator


def scd_koala(
    clip: vs.VideoNode,
    filter_size: int = 3,
    window_size: int = 8,
    deviation: float = 3.0,
    edge_func: Callable[[vs.VideoNode], vs.VideoNode] = partial(AnimeMask, mode=-1),
) -> vs.VideoNode:
    """
    Koala-36M Based Scenecut Detector

    Args:
        clip: Input video clip.
        filter_size: Boxcar filter size.
        window_size: Window to use for calculating threshold.
        deviation: Multiplier for standard deviations when calculating threshold.
        edge_func: Function to detect edges in the input clip.

    Returns:
        Input video clip with frame prop "_Scenecut" set to 1 at scenecuts.
    """
    if filter_size > window_size:
        raise ValueError("scd_koala: filter_size should be <= window_size.")

    resized_clip = core.resize.Bicubic(clip, width=256, height=256, format=vs.RGB24, dither_type="none")

    gray_clip = core.std.ShufflePlanes(resized_clip, planes=0, colorfamily=vs.GRAY)
    small_gray = core.resize.Bicubic(gray_clip, width=128, height=128)

    edges = edge_func(small_gray)

    combined_edges = core.akarin.Expr([small_gray, edges], expr="x y max")
    prev_edge_clip = core.std.Splice([combined_edges[0], combined_edges[:-1]], mismatch=True)
    ssim_result = muvsfunc.SSIM(prev_edge_clip, combined_edges)

    def _get_score(curr_array: np.ndarray, prev_array: np.ndarray, delta_edges: float) -> float:
        prev_hist = []
        curr_hist = []

        for c in range(3):
            channel_data = prev_array[:, :, c]
            mask = (channel_data > 0) & (channel_data < 255)
            hist, _ = np.histogram(channel_data[mask], bins=254, range=(1, 255)) if np.any(mask) else (np.zeros(254), None)
            prev_hist.append(hist)

            channel_data = curr_array[:, :, c]
            mask = (channel_data > 0) & (channel_data < 255)
            hist, _ = np.histogram(channel_data[mask], bins=254, range=(1, 255)) if np.any(mask) else (np.zeros(254), None)
            curr_hist.append(hist)

        delta_histogram = histogram_correlation(np.array(prev_hist), np.array(curr_hist))

        return 4.61480465 * delta_histogram + 3.75211168 * delta_edges - 5.485968377115124

    def _scenecut_eval(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        if n == 0:
            fout.props._Scenecut = 0
            return fout

        k = (filter_size - 1) // 2
        num_scores = window_size + filter_size

        resized_frames = f[1: 1 + num_scores + 1]
        ssim_frames = f[1 + num_scores + 1:]

        scores = []
        for i in range(num_scores):
            prev_frame = resized_frames[i]
            curr_frame = resized_frames[i+1]
            ssim_frame = ssim_frames[i]

            prev_array = np.dstack(
                [np.array(prev_frame[p]) for p in range(prev_frame.format.num_planes)]  # type: ignore
            )
            curr_array = np.dstack(
                [np.array(curr_frame[p]) for p in range(curr_frame.format.num_planes)]  # type: ignore
            )
            delta_edges = ssim_frame.props.get("PlaneSSIM", 0.0)

            score = _get_score(curr_array, prev_array, delta_edges)
            scores.append(score)

        current_idx = window_size + k
        is_cut = scores[current_idx] < 0.0

        filter_kernel = np.ones(filter_size) / filter_size
        filtered = np.convolve(scores, filter_kernel, mode="same")

        current_filtered_score = filtered[current_idx]

        if n >= window_size and current_filtered_score < float(filter_size) / float(filter_size + 1):
            window_start = current_idx - window_size
            window_end = current_idx
            window = filtered[window_start:window_end]
            threshold = np.mean(window) - (deviation * np.std(window))
            if current_filtered_score < threshold:
                is_cut = True

        fout.props._Scenecut = 1 if is_cut else 0
        return fout

    k = (filter_size - 1) // 2
    prop_clips: list[vs.VideoNode] = [clip]

    for i in range(window_size + filter_size + 1):
        rel_idx = i - (window_size + k + 1)
        shifted = core.std.Splice([core.std.BlankClip(resized_clip, length=abs(rel_idx)), resized_clip], mismatch=True) if rel_idx < 0 else resized_clip[rel_idx:]
        prop_clips.append(shifted)

    for i in range(window_size + filter_size):
        rel_idx = i - (window_size + k)
        shifted = core.std.Splice([core.std.BlankClip(ssim_result, length=abs(rel_idx)), ssim_result], mismatch=True) if rel_idx < 0 else ssim_result[rel_idx:]
        prop_clips.append(shifted)

    return core.std.ModifyFrame(clip, clips=prop_clips, selector=_scenecut_eval)
