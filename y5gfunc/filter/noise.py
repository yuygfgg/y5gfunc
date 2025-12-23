from typing import Union
from vstools import core, vs
import vstools


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

    Raises:
        ValueError: If sigma values are not positive
        ValueError: If sigma is not a single value or the length of sigma list does not match the number of planes in the clip.
    """
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * clip.format.num_planes
    elif len(sigma) != clip.format.num_planes:
        raise ValueError(
            "add_noise: Length of sigma list must match the number of planes in the clip, or a single value that applies to all planes."
        )

    if min(sigma) <= 0:
        raise ValueError("add_noise: Sigma values must be positive.")

    planes = vstools.split(clip)
    out_planes = []
    for i, p in enumerate(planes):
        out_planes.append(core.grain.Add(p, sigma[i] ** 2))

    return vstools.join(out_planes, clip.format.color_family)
