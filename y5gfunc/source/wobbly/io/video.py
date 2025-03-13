"""
Video loading and source handling.
"""

import os
from typing import Optional

import vapoursynth as vs

from ..types import FramePropertyMap, VideoFormat
from ..errors import WobblyProcessError, WobblyInputError


core = vs.core


def load_source_video(input_file: str, source_filter: str) -> vs.VideoNode:
    """
    Load source video based on specified filter
    
    Args:
        input_file: Path to input video file
        source_filter: Source filter to use (plugin.function)
        
    Returns:
        Loaded video node
        
    Raises:
        WobblyInputError: If the input file doesn't exist
        WobblyProcessError: If loading fails
    """
    if not os.path.exists(input_file):
        raise WobblyInputError(f"Input file does not exist: {input_file}")
        
    try:
        if source_filter == "bs.VideoSource":
            # BestSource loading
            return core.bs.VideoSource(input_file, rff=True, showprogress=False)
        else:
            # Use specified filter
            filter_parts = source_filter.split('.')
            if len(filter_parts) < 2:
                raise WobblyInputError(f"Invalid source filter: {source_filter}, expected format: plugin.function")
                
            plugin = getattr(core, filter_parts[0])
            return getattr(plugin, filter_parts[1])(input_file)
    except Exception as e:
        raise WobblyProcessError(f"Failed to load video: {e}", stage="video_loading", cause=e)


def query_format(
    clip: vs.VideoNode, 
    bits: int, 
    sample_type: VideoFormat,
    subsampling_w: Optional[int] = None,
    subsampling_h: Optional[int] = None
) -> int:
    """
    Query video format ID
    
    Args:
        clip: Reference clip
        bits: Bit depth
        sample_type: Sample type (FLOAT or INTEGER)
        subsampling_w: Horizontal subsampling, or None to use reference clip's value
        subsampling_h: Vertical subsampling, or None to use reference clip's value
        
    Returns:
        Format ID
    """
    if subsampling_w is None:
        subsampling_w = clip.format.subsampling_w
        
    if subsampling_h is None:
        subsampling_h = clip.format.subsampling_h
        
    return core.query_video_format(
        clip.format.color_family,
        sample_type.value,
        bits,
        subsampling_w,
        subsampling_h
    ).id


def apply_frame_properties(clip: vs.VideoNode, frame_props: FramePropertyMap) -> vs.VideoNode:
    """
    Apply frame properties to video frames
    
    Args:
        clip: Input clip
        frame_props: Frame property mapping
        
    Returns:
        Clip with applied frame properties
    """
    def transfer_props(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if n in frame_props:
            fout = f.copy()
            
            # Add all properties
            for key, value in frame_props[n].items():
                if value is not None:  # Skip None values
                    # VapourSynth doesn't accept None as frame property value
                    fout.props[key] = value # type: ignore
                    
            return fout
        return f
    
    # Apply all saved frame properties
    return core.std.ModifyFrame(clip, clip, transfer_props)