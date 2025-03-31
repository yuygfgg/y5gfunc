from typing import List, Union
from vstools import vs
from vstools import core
from pathlib import Path

def ranger(start: Union[int, float], end: Union[int, float], step: Union[int, float]) -> List[Union[int, float]]:
    assert step != 0
    return list(map(lambda i: round(start + i * step, 10), range(int((end - start) / step))))

def PickFrames(clip: vs.VideoNode, indices: list[int]) -> vs.VideoNode:
    try: 
        ret = core.akarin.PickFrames(clip, indices=indices) # type: ignore
    except AttributeError:
        # modified from https://github.com/AkarinVS/vapoursynth-plugin/issues/26#issuecomment-1951230729
        new = clip.std.BlankClip(length=len(indices))
        ret = new.std.FrameEval(lambda n: clip[indices[n]], None, clip) # type: ignore
    
    return ret

def resolve_path(path: Union[Path, str]) -> Path:
    path = Path(path).resolve()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path