from vstools import vs
from vstools import core
from .wobbly import load_and_process
from typing import Optional, Union
from pathlib import Path
from ..utils import resolve_path

def _wobbly_source(
    wob_project_path: Union[str, Path], 
    timecodes_v2_path: Optional[Union[str, Path]] = None
) -> vs.VideoNode:

    clip = load_and_process(wob_project_path, timecodes_v2_path, timecode_version="v2").std.SetFieldBased(False)
    return clip

def _bestsource(
    file_path: Union[Path, str],
    track: int = 0,
    timecodes_v2_path: Optional[Union[Path, str]] = None,
    variableformat: int = -1,
    rff: bool = False
) -> vs.VideoNode:
    
    if timecodes_v2_path:
        return core.bs.VideoSource(str(file_path), track, variableformat, timecodes=str(timecodes_v2_path), rff=rff)
    else:
        return core.bs.VideoSource(str(file_path), track, variableformat, rff=rff)

# TODO: auto matrix handle
def load_source(
    file_path: Union[Path, str],
    track: int = 0,
    matrix_s: str = "709",
    matrix_in_s: str = "709",
    timecodes_v2_path: Optional[Union[Path, str]] = None
) -> vs.VideoNode:

    file_path = resolve_path(file_path)
    
    assert file_path.exists()
    
    if file_path.suffix.lower() == ".wob":
        assert track == 0
        clip = _wobbly_source(file_path, timecodes_v2_path)
    else:
        # modified from https://guides.vcb-s.com/basic-guide-10/#%E6%A3%80%E6%B5%8B%E6%98%AF%E5%90%A6%E4%B8%BA%E5%85%A8%E7%A8%8B-soft-pulldownpure-film
        a = _bestsource(file_path, rff=False)
        b = _bestsource(file_path, rff=True)
        rff = False if abs(b.num_frames * 0.8 - a.num_frames) < 1 else True
        
        clip = _bestsource(file_path, track, timecodes_v2_path, rff=rff)
    
    return clip.resize.Spline36(matrix_s=matrix_s, matrix_in_s=matrix_in_s)