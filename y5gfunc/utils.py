from typing import Union
from vstools import vs, Matrix, Primaries, Transfer
from pathlib import Path

_MATRIX_TRANSFER_MAP: dict[Matrix, Transfer] = {
    Matrix.RGB: Transfer.IEC_61966_2_1,
    Matrix.BT709: Transfer.BT709,
    Matrix.BT470_BG: Transfer.BT601,
    Matrix.ST170_M: Transfer.BT601,
    Matrix.ST240_M: Transfer.ST240_M,
    Matrix.CHROMATICITY_DERIVED_NC: Transfer.IEC_61966_2_1,
    Matrix.ICTCP: Transfer.BT2020_10,
}


def primaries_from_matrix(matrix: Matrix, strict: bool = False) -> Primaries:
    """
    Derive the ``Primaries`` for a given ``Matrix``.

    Args:
        matrix: Source matrix object.
        strict: When True, raise on matrices whose value has no ``Primaries``
            counterpart instead of falling back to ``Primaries.UNSPECIFIED``.

    Returns:
        The matching ``Primaries`` value.

    Raises:
        ValueError: If ``strict`` is True and ``matrix`` cannot be mapped.
    """
    try:
        return Primaries(matrix.value)
    except ValueError:
        if strict:
            raise
        return Primaries.UNSPECIFIED


def transfer_from_matrix(matrix: Matrix, strict: bool = False) -> Transfer:
    """
    Derive the ``Transfer`` for a given ``Matrix``.

    Args:
        matrix: Source matrix object.
        strict: When True, raise on matrices missing from the mapping table
            instead of falling back to ``Transfer(matrix.value)``.

    Returns:
        The matching ``Transfer`` value.

    Raises:
        ValueError: If ``strict`` is True and ``matrix`` is not in the mapping.
    """
    transfer = _MATRIX_TRANSFER_MAP.get(matrix)
    if transfer is not None:
        return transfer
    if strict:
        raise ValueError(f"{matrix} is not supported by transfer_from_matrix")
    return Transfer(matrix.value)


def ranger(
    start: Union[int, float], end: Union[int, float], step: Union[int, float]
) -> list[Union[int, float]]:
    """
    Generates a sequence of numbers similar to range(), but allows floats.

    Creates a list of numbers starting from `start`, incrementing by `step`, and stopping before `end`.

    Args:
        start: The starting value of the sequence (inclusive).
        end: The end value of the sequence (exclusive). The sequence generated will contain values strictly
            less than `end` if `step` is positive, or strictly greater than `end` if `step` is negative.
        step: The step/increment between consecutive numbers. Must not be zero. Can be negative for descending sequences.

    Returns:
        A list of numbers (integers or floats) representing the generated sequence.

    Raises:
        ValueError: If `step` is 0.
    """

    if step == 0:
        raise ValueError("ranger: Step cannot be zero.")
    return list(
        map(lambda i: round(start + i * step, 10), range(int((end - start) / step)))
    )


# https://discord.com/channels/1168547111139283026/1168591112160690227/1356645786342920202
def PickFrames(clip: vs.VideoNode, indices: list[int]) -> vs.VideoNode:
    """
    Return a new clip with frames picked from input clip from the indices array.

    Args:
        clip: Input clip where frames are from.
        indices: The indices array representing the frames to be picked.

    Returns:
        New clip with frames picked from input clip from the indices array.
    """
    return clip.std.SelectEvery(cycle=clip.num_frames, offsets=indices)


def resolve_path(path: Union[Path, str]) -> Path:
    """
    Resolves a path to an absolute path and ensures necessary directories exist.

    Args:
        path: The input path string or Path object to resolve and for which to ensure directory structure exists.

    Returns:
        The absolute, resolved pathlib. Path object corresponding to the input path, after ensuring the relevant directory structure exists.
    """
    path = Path(path).resolve()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path
