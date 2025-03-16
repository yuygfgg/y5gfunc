from vstools import vs
from pathlib import Path
from typing import Union
from ..utils import PickFrames, resolve_path

def screen_shot(
    clip: vs.VideoNode,
    frames: Union[list[int], int],
    path: Union[Path, str],
    file_name: str,
    overwrite: bool = True
):
    
    if isinstance(frames, int):
        frames = [frames]
    
    output_path = resolve_path(path)
    
    clip = clip.resize2.Spline36(format=vs.RGB24)
    clip = PickFrames(clip=clip, indices=frames)
        
    for i, _ in enumerate(clip.frames()):
        tmp = clip.std.Trim(first=i, last=i).fpng.Write(filename=(output_path / (file_name%frames[i])).with_suffix('.png'), overwrite=overwrite, compression=2) # type: ignore
        for f in tmp.frames():
            pass