from vstools import vs
from vstools import core
from typing import Union, Optional
from vstools import get_peak_value
from .utils import is_optimized_cpu

NEIGHBOR_OFFSETS = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def _create_minmax_expr(
    clip: vs.VideoNode,
    process_expr: str,
    threshold_expr: str,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    coordinates: list[int] = [1, 1, 1, 1, 1, 1, 1, 1],
    boundary: int = 1,
) -> vs.VideoNode:
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]

    def _build_neighbor_expr(coordinates: list[int]) -> str:
        return " ".join(
            f"x[{dx},{dy}]"
            for flag, (dx, dy) in zip(coordinates, NEIGHBOR_OFFSETS)
            if flag
        )

    if len(coordinates) != 8:
        raise ValueError("coordinates must contain exactly 8 elements.")

    neighbor_expr = _build_neighbor_expr(coordinates)
    expr = f"x[0,0] {' ' + neighbor_expr if neighbor_expr else ''} sort{sum(coordinates) + 1} {process_expr}"

    if threshold is not None:
        expr += threshold_expr.format(threshold)

    expressions = [
        expr if (i in planes) else "x" for i in range(clip.format.num_planes)
    ]

    return core.akarin.Expr(clips=[clip], expr=expressions, boundary=boundary)


def minimum(
    clip: vs.VideoNode,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    coordinates: list[int] = [1, 1, 1, 1, 1, 1, 1, 1],
    boundary: int = 1,
    use_std: bool = is_optimized_cpu(),
) -> vs.VideoNode:
    if use_std:
        return core.std.Minimum(clip, planes, threshold, coordinates)  # type: ignore
    else:
        return _create_minmax_expr(
            clip,
            "min! drop{} min@".format(sum(coordinates)),
            " x[0,0] {} - swap max",
            planes,
            threshold,
            coordinates,
            boundary,
        )


def maximum(
    clip: vs.VideoNode,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    coordinates: list[int] = [1, 1, 1, 1, 1, 1, 1, 1],
    boundary: int = 1,
    use_std: bool = is_optimized_cpu(),
) -> vs.VideoNode:
    if use_std:
        return core.std.Maximum(clip, planes, threshold, coordinates)  # type: ignore
    else:
        return _create_minmax_expr(
            clip,
            "drop{}".format(sum(coordinates)),
            " x[0,0] {} + swap min",
            planes,
            threshold,
            coordinates,
            boundary,
        )


# TODO: add exprs for other modes
def convolution(
    clip: vs.VideoNode,
    matrix: list[int],
    bias: float = 0.0,
    divisor: float = 0.0,
    planes: Optional[Union[list[int], int]] = None,
    saturate: bool = True,
    mode: str = "s",
    use_std: bool = is_optimized_cpu(),
) -> vs.VideoNode:
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]

    if mode != "s" or len(matrix) != 9 or use_std:
        return core.std.Convolution(clip, matrix, bias, divisor, planes, saturate, mode)

    if len(matrix) == 9:
        if abs(divisor) < 1e-9:
            actual_divisor = sum(matrix) if abs(sum(matrix)) > 1e-9 else 1.0
        else:
            actual_divisor = divisor

        coeffs = [f"{c:.6f}" for c in matrix]

        expr_parts = []

        if len(matrix) == 9:
            offsets = [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]

        for i, (dx, dy) in enumerate(offsets):
            expr_parts.append(f"x[{dx},{dy}] {coeffs[i]} *")
            if i > 0:
                expr_parts.append("+")

        expr_parts.append(f" {actual_divisor:.6f} / {bias:.6f} + ")

        peak = get_peak_value(clip)

        if saturate:
            expr_parts.append(f"0 {peak} clip")
        else:
            expr_parts.append("abs")
            expr_parts.append(f"{peak} min")

        expr = " ".join(expr_parts)
        expressions = [
            expr if i in planes else "x" for i in range(clip.format.num_planes)
        ]

        return core.akarin.Expr(clip, expressions, boundary=1)

    return core.std.Convolution(clip, matrix, bias, divisor, planes, saturate, mode)


def inflate(
    clip: vs.VideoNode,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    boundary: int = 1,
    use_std: bool = is_optimized_cpu(),
) -> vs.VideoNode:
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]

    if use_std:
        return core.std.Inflate(clip, planes, threshold)  # type: ignore

    expr_parts = []

    for i, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
        expr_parts.append(f"x[{dx},{dy}] 8 / ")
        if i > 0:
            expr_parts.append("+ ")

    expr_parts.append("x max")
    if threshold:
        expr_parts.append(f" x {threshold} + min")
    expr = "".join(expr_parts)

    expressions = [
        expr if (i in planes) else "x" for i in range(clip.format.num_planes)
    ]

    return core.akarin.Expr(clips=[clip], expr=expressions, boundary=boundary)


def deflate(
    clip: vs.VideoNode,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    boundary: int = 1,
    use_std: bool = is_optimized_cpu(),
) -> vs.VideoNode:
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]

    if use_std:
        return core.std.Deflate(clip, planes, threshold)  # type: ignore

    expr_parts = []

    for i, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
        expr_parts.append(f"x[{dx},{dy}] 8 / ")
        if i > 0:
            expr_parts.append("+")

    expr_parts.append("x min")
    if threshold:
        expr_parts.append(f" x {threshold} - max")
    expr = "".join(expr_parts)

    expressions = [
        expr if (i in planes) else "x" for i in range(clip.format.num_planes)
    ]

    return core.akarin.Expr(clips=[clip], expr=expressions, boundary=boundary)
