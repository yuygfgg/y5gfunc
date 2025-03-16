from types import FrameType
from vstools import vs
from vstools import core
from vspreview import set_output
import inspect

_output_index = 0
used_indices = set()

def _get_variable_name(frame: FrameType, clip: vs.VideoNode) -> str:
    for var_name, var_val in frame.f_locals.items():
        if var_val is clip:
            return var_name
    return "Unknown Variable"

def _add_text(clip: vs.VideoNode, text: str, debug: bool) -> vs.VideoNode:
    return core.akarin.Text(clip, text) if debug else clip

def reset_output_index(index: int = 0) -> None:
    global _output_index
    _output_index = index
    global used_indices
    used_indices = set()

def output(*args, debug: bool = True) -> None:
    if debug and __name__ == '__vapoursynth__':
        raise ValueError("Don't set debug=True when encoding!")
    
    frame: FrameType = inspect.currentframe().f_back # type: ignore
    
    global _output_index
    global used_indices
    clips_to_process = []

    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, tuple) and len(item) == 2:
                    clip, index = item
                    if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                        raise TypeError("Tuple must be (VideoNode, int)")
                    if index < 0:
                        raise ValueError("Output index must be non-negative")
                    clips_to_process.append((clip, index))
                elif isinstance(item, vs.VideoNode):
                    clips_to_process.append((item, None))
                else:
                    raise TypeError(f"Invalid element in list/tuple: {type(item)}")
        elif isinstance(arg, vs.VideoNode):
            clips_to_process.append((arg, None))
        elif isinstance(arg, tuple) and len(arg) == 2:
            clip, index = arg
            if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                raise TypeError("Tuple must be (VideoNode, int)")
            if index < 0:
                raise ValueError("Output index must be non-negative")
            clips_to_process.append((clip, index))
        else:
            raise TypeError(f"Invalid argument type: {type(arg)}")

    for clip, index in clips_to_process:
        if index is not None:
            if index in used_indices:
                raise ValueError(f"Output index {index} is already in use")
            variable_name = _get_variable_name(frame, clip)
            clip = _add_text(clip, f"{variable_name}", debug)
            set_output(clip, index, f'{index}: {variable_name}')
            used_indices.add(index)
            
    for clip, index in clips_to_process:
        if index is None:
            while _output_index in used_indices:
                _output_index += 1
            variable_name = _get_variable_name(frame, clip)
            clip = _add_text(clip, f"{variable_name}", debug)            
            set_output(clip, _output_index, f'{_output_index}: {variable_name}')
            used_indices.add(_output_index)
