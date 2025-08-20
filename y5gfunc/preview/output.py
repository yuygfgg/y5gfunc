from types import FrameType
from typing import Optional
import inspect
from vspreview import set_output
from vstools import vs
from vstools import core

_output_index = 0
used_indices = set()


def _get_variable_name(frame: FrameType, clip: vs.VideoNode) -> str:
    """Tries to find the variable name assigned to the clip in the calling frame."""
    for var_name, var_val in frame.f_locals.items():
        if var_val is clip:
            return var_name
    return "Unknown Variable"


def _add_text(clip: vs.VideoNode, text: str, debug: bool) -> vs.VideoNode:
    """Adds a text overlay to the clip if debug is True."""
    return core.akarin.Text(clip, text) if debug else clip


def reset_output_index(index: int = 0) -> None:
    """
    Resets the global output index counter and the set of used indices.

    Args:
        index: The value to reset the automatic index counter to.
    """
    global _output_index
    _output_index = index
    global used_indices
    used_indices = set()


def set_preview(*args, debug: bool = True) -> None:
    """
    Outputs VapourSynth clips in vspreview.

    This function handles assigning output indices automatically or allows for explicit index specification.
    Nodes are named in `vspreview` as `{index}: {variable_name}`. When debug=True, node name is overlaid onto the video.

    Args:
        *args: A variable number of arguments representing the clips to output.
            Each argument can be:
            - A single `vs.VideoNode`: This clip will be assigned the next available automatic output index.
            - A list or tuple of `vs.VideoNode` objects: Each node in the sequence will be assigned the next available automatic output index.
            - A tuple in the format `(vs.VideoNode, int)`: The specified node will be assigned the given integer index. The index must be non-negative.
            - A list or tuple containing any combination of the above formats, including mixtures of `vs.VideoNode` and `(vs.VideoNode, int)` tuples.
        debug: If True, overlays the inferred variable name onto the output clip for identification during previews.

    Raises:
        ValueError: If `debug` is True when the script is run for encoding (i.e., `__name__ == '__vapoursynth__'`).
        ValueError: If an explicitly provided output index in a `(clip, index)` tuple is negative.
        ValueError: If an explicitly provided output index is already assigned by a prior call to `output` in the same script execution (unless `reset_output_index` has been called).
        TypeError: If any provided argument or an element within a passed list/tuple is not a `vs.VideoNode` or a `(vs.VideoNode, int)` tuple.

    Note:
        This function uses global variables (`_output_index`, `used_indices`) to track assigned indices across multiple calls within a single script run.
        Use `reset_output_index()` to clear this state if you need to reuse indices or manage output sets independently.
    """
    if debug and __name__ == "__vapoursynth__":
        raise ValueError("output: Do not set debug=True when encoding!")

    frame: FrameType = inspect.currentframe().f_back  # type: ignore[union-attr]

    global _output_index
    global used_indices
    clips_to_process: list[tuple[vs.VideoNode, Optional[int]]] = []

    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, tuple) and len(item) == 2:
                    clip, index = item
                    if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                        raise TypeError(
                            "output: Expected a tuple of (VideoNode, int) for explicit indexing."
                        )
                    if index < 0:
                        raise ValueError("output: Output index must be non-negative.")
                    clips_to_process.append((clip, index))
                elif isinstance(item, vs.VideoNode):
                    clips_to_process.append((item, None))
                else:
                    raise TypeError(
                        f"output: Invalid element type in list/tuple: {type(item)}. Expected VideoNode or (VideoNode, int)."
                    )
        elif isinstance(arg, vs.VideoNode):
            clips_to_process.append((arg, None))
        elif isinstance(arg, tuple) and len(arg) == 2:
            clip, index = arg
            if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                raise TypeError(
                    "output: Expected a tuple of (VideoNode, int) for explicit indexing."
                )
            if index < 0:
                raise ValueError("output: Output index must be non-negative.")
            clips_to_process.append((clip, index))
        else:
            raise TypeError(
                f"output: Invalid argument type: {type(arg)}. Expected VideoNode, list/tuple of VideoNodes, or (VideoNode, int)."
            )

    # Explicitly Indexed Clips
    for clip, index in clips_to_process:
        if index is not None:
            if index in used_indices:
                raise ValueError(
                    f"output: Output index {index} is already in use. Reset with reset_output_index() or choose another index."
                )
            variable_name = _get_variable_name(frame, clip)
            processed_clip = _add_text(clip, f"{index}: {variable_name}", debug)
            set_output(processed_clip, index, name=f"{index}: {variable_name}")
            used_indices.add(index)

    # Automatically Indexed Clips
    for clip, index in clips_to_process:
        if index is None:
            while _output_index in used_indices:
                _output_index += 1
            current_auto_index = _output_index
            variable_name = _get_variable_name(frame, clip)
            processed_clip = _add_text(
                clip, f"{current_auto_index}: {variable_name}", debug
            )
            set_output(
                processed_clip,
                current_auto_index,
                name=f"{current_auto_index}: {variable_name}",
            )
            used_indices.add(current_auto_index)
            _output_index += 1
