from typing import List, Union
from vstools import vs
from pathlib import Path

def ranger(start: Union[int, float], end: Union[int, float], step: Union[int, float]) -> List[Union[int, float]]:
    assert step != 0
    return list(map(lambda i: round(start + i * step, 10), range(int((end - start) / step))))

# https://discord.com/channels/1168547111139283026/1168591112160690227/1356645786342920202
def PickFrames(clip: vs.VideoNode, indices: list[int]) -> vs.VideoNode:
    return clip.std.SelectEvery(cycle=clip.num_frames, offsets=indices)

def resolve_path(path: Union[Path, str]) -> Path:
    path = Path(path).resolve()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path