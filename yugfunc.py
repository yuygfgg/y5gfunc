import inspect
import vapoursynth as vs
from vapoursynth import core

_output_index = 1
debug = 1

def enable_debug():
    global debug
    debug = 1

def disable_debug():
    global debug
    debug = 0

def reset_output_index(index=1):
    global _output_index
    _output_index = index

def output(*args):
    def _get_variable_name(frame, clip):
        for var_name, var_val in frame.f_locals.items():
            if var_val is clip:
                return var_name
        return None

    def _add_text(clip, text):
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(f"_add_text expected a VideoNode, but got {type(clip)}")
        return core.text.Text(clip, text)
    
    global _output_index
    frame = inspect.currentframe().f_back # type: ignore
    used_indices = set()
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
            used_indices.add(index)
            if debug and index != 0:
                variable_name = _get_variable_name(frame, clip)
                if variable_name:
                    clip = _add_text(clip, f"{variable_name}")
                else:
                    clip = _add_text(clip, "Unknown Variable")
            clip.set_output(index)

    for clip, index in clips_to_process:
        if index is None:
            while _output_index in used_indices:
                _output_index += 1
            if debug and _output_index != 0:
                variable_name = _get_variable_name(frame, clip)
                if variable_name:
                    clip = _add_text(clip, f"{variable_name}")
                else:
                    clip = _add_text(clip, "Unknown Variable")
            clip.set_output(_output_index)
            used_indices.add(_output_index)
            _output_index += 1
