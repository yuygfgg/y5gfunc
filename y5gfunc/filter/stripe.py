from vstools import vs
from vstools import core
from typing import Union
import functools


# inspired by https://skyeysnow.com/forum.php?mod=redirect&goto=findpost&ptid=13824&pid=333218
def is_stripe(
    clip: vs.VideoNode,
    threshold: Union[float, int] = 2,
    freq_range: Union[int, float] = 0.25,
    scenecut_threshold: Union[float, int] = 0.1,
) -> vs.VideoNode:
    """
    Detects scenes potentially containing strong vertical stripes using FFT analysis.

    This function analyzes the frequency spectrum of each frame to identify scenes where the energy in high vertical frequencies
    significantly outweighs the energy in high horizontal frequencies. This pattern is often indicative of vertical stripes.

    The analysis is performed per scene, meaning the ratio of vertical to horizontal frequency energy is averaged over all frames
    within a detected scene. All frames in that scene are then marked based on this average ratio.

    Args:
        clip: Input video clip. Must be 32-bit per sample.
        threshold: The threshold for the ratio of average vertical high-frequency energy to average horizontal
            high-frequency energy within a scene. If a scene's ratio exceeds this value, it's marked as potentially containing stripes.
        freq_range: Defines the proportion of the spectrum (from each edge towards the center) considered as "high frequency".
            For example, 0.25 means the outer 25% of frequencies horizontally and vertically are analyzed.
            Must be between 0 and 0.5 (exclusive of 0, inclusive of 0.5 isn't practically useful).
        scenecut_threshold: Threshold used for scene change detection via `core.misc.SCDetect`.
            Controls how sensitive the scene detection is. Lower values detect more scene changes.

    Returns:
        The input clip with a frame property `_Stripe` added. `_Stripe` is 1 (True) if the frame belongs to a scene detected as
            potentially containing stripes (ratio > threshold), and 0 (False) otherwise.

    Raises:
        ValueError: If input clip is not 32-bit per sample.
        ValueError: if freq_range is not between 0 and 0.5.
    """

    def scene_fft(
        n: int, f: list[vs.VideoFrame], cache: list[float], prefetch: vs.VideoNode
    ) -> vs.VideoFrame:
        fout = f[0].copy()
        if n == 0 or n == prefetch.num_frames:
            fout.props["_SceneChangePrev"] = 1

        if cache[n] == -1.0:  # not cached
            i = n
            scene_start = n
            while i >= 0:
                frame = prefetch.get_frame(i)
                if frame.props["_SceneChangePrev"] == 1:  # scene start
                    scene_start = i
                    break
                i -= 1
            i = scene_start
            scene_length = 0
            hor_accum = 1e-9
            ver_accum = 0
            while i < prefetch.num_frames:
                frame = prefetch.get_frame(i)
                hor_accum += frame.props["hor"]  # type: ignore[index]
                ver_accum += frame.props["ver"]  # type: ignore[index]
                scene_length += 1
                i += 1
                if frame.props["_SceneChangeNext"] == 1:  # scene end
                    break

            ratio = ver_accum / hor_accum

            cache[scene_start : scene_start + scene_length] = [ratio] * scene_length

        fout.props["ratio"] = cache[n]
        return fout

    if clip.format.bits_per_sample != 32:
        raise ValueError("is_stripe: input clip must be 32-bit per sample.")
    if not 0 < freq_range < 0.5:
        raise ValueError("is_stripe: freq_range must be between 0 and 0.5.")

    freq_drop_range = 1 - freq_range
    freq_drop_lr = int(clip.width * freq_drop_range)
    freq_drop_bt = int(clip.height * freq_drop_range)

    fft = core.fftspectrum_rs.FFTSpectrum(clip)

    left = core.std.Crop(fft, right=freq_drop_lr)
    right = core.std.Crop(fft, left=freq_drop_lr)
    hor = core.std.StackHorizontal([left, right]).std.PlaneStats()

    top = core.std.Crop(fft, bottom=freq_drop_bt)
    bottom = core.std.Crop(fft, top=freq_drop_bt)
    ver = core.std.StackHorizontal([top, bottom]).std.PlaneStats()

    scene = core.misc.SCDetect(clip, threshold=scenecut_threshold)

    prefetch = core.std.BlankClip(clip)
    prefetch = core.akarin.PropExpr(
        [hor, ver, scene],
        lambda: {
            "hor": "x.PlaneStatsAverage",
            "ver": "y.PlaneStatsAverage",
            "_SceneChangeNext": "z._SceneChangeNext",
            "_SceneChangePrev": "z._SceneChangePrev",
        },
    )

    cache = [-1.0] * scene.num_frames

    ret = core.std.ModifyFrame(
        scene,
        [scene, scene],
        functools.partial(scene_fft, prefetch=prefetch, cache=cache),
    )
    ret = core.akarin.PropExpr(
        [ret], lambda: {"_Stripe": f"x.ratio {threshold} >"}
    )  # x.ratio > threshold: Stripe

    return ret
