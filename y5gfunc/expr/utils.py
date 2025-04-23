from vstools import vs
from typing import Union, Optional


# modified from jvsfunc.ex_planes()
def ex_planes(
    clip: vs.VideoNode, expr: list[str], planes: Optional[Union[int, list[int]]] = None
) -> list[str]:
    if planes:
        plane_range = range(clip.format.num_planes)
        planes = [planes] if isinstance(planes, int) else planes
        expr = [expr[0] if i in planes else "" for i in plane_range]
    return expr
