from typing import Union
import numpy as np
from vapoursynth import ColorRange
from vstools import core, vs
from .utils import get_peak_value_full

def add_noise(
    clip: vs.VideoNode, sigma: Union[int, float] | list[Union[int, float]] = 5
) -> vs.VideoNode:
    r"""
    Add Gaussian noise to a video clip using NumPy.

    Args:
        clip: The input video clip.
        sigma: Noise strength defined as the standard deviation of the AWGN component,
                implying $Noise \sim \mathcal{N}(0, \sigma^2)$ on a $0\text{-}255$ scale.

    Returns:
        A new clip with the added noise.
    """
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * clip.format.num_planes
    elif len(sigma) != clip.format.num_planes:
        raise ValueError(
            "add_noise: Length of sigma list must match the number of planes in the clip, or a single value that applies to all planes."
        )

    max_val = get_peak_value_full(clip)

    def noise_eval(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        for p in range(fout.format.num_planes):
            plane_arr = np.asarray(fout[p])

            noise = np.random.normal(0, sigma[p] / 255.0, plane_arr.shape)

            if clip.format.sample_type == vs.INTEGER:
                img_norm = plane_arr.astype(np.float32) / max_val
                noisy = np.clip(img_norm + noise, 0, 1.0) * max_val
                plane_arr[:] = noisy.astype(plane_arr.dtype)
            else:
                plane_arr[:] = plane_arr + noise
        return fout

    return core.std.ModifyFrame(clip, clip, noise_eval)
