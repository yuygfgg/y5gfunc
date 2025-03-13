"""
Resize and bit depth processor.
"""

from typing import Dict, Any, Tuple

import vapoursynth as vs

from .base import BaseProcessor
from ..types import ProjectData, FramePropertyMap, FrameMap, PresetDict, WobblyKeys, VideoFormat
from ..core.context import safe_processing
from ..io.video import query_format

core = vs.core


class ResizeProcessor(BaseProcessor):
    """Resize and bit depth processor implementation"""
    
    def __init__(self):
        """Initialize resize processor"""
        self.keys = WobblyKeys()
    
    def process(
        self, 
        clip: vs.VideoNode, 
        project: ProjectData,
        frame_props: FramePropertyMap,
        frame_mapping: FrameMap,
        presets: PresetDict
    ) -> Tuple[vs.VideoNode, FramePropertyMap, FrameMap]:
        """
        Process resize and bit depth operations
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        with safe_processing("resize_and_depth_processing"):
            Keys = self.keys
            resize_info = project.get(Keys.project.resize, {})
            depth_info = project.get(Keys.project.depth, {})
            
            resize_enabled = resize_info.get(Keys.resize.enabled, False)
            depth_enabled = depth_info.get(Keys.depth.enabled, False)
            
            if not (resize_enabled or depth_enabled):
                return clip, frame_props, frame_mapping
                
            # Record resize and bit depth info
            resize_props: Dict[str, Any] = {}
            
            # Get filter name
            resize_filter_name = resize_info.get(Keys.resize.filter, "Bicubic")
            if resize_filter_name:
                resize_filter_name = resize_filter_name[0].upper() + resize_filter_name[1:]
            else:
                resize_filter_name = "Bicubic"
                
            if not hasattr(core.resize, resize_filter_name):
                resize_filter_name = "Bicubic"
                
            # Prepare resize arguments
            resize_args: Dict[str, Any] = {}
            if resize_enabled:
                resize_width = resize_info.get(Keys.resize.width, clip.width)
                resize_height = resize_info.get(Keys.resize.height, clip.height)
                resize_args["width"] = resize_width
                resize_args["height"] = resize_height
                
                resize_props.update({
                    "WobblyResizeEnabled": True,
                    "WobblyResizeWidth": resize_width,
                    "WobblyResizeHeight": resize_height,
                    "WobblyResizeFilter": resize_filter_name
                })
                
            # Add bit depth arguments if needed
            if depth_enabled:
                bits = depth_info.get(Keys.depth.bits, 8)
                float_samples = depth_info.get(Keys.depth.float_samples, False)
                dither = depth_info.get(Keys.depth.dither, "")
                sample_type = VideoFormat.FLOAT if float_samples else VideoFormat.INTEGER
                
                format_id = query_format(
                    clip,
                    bits,
                    sample_type
                )
                
                resize_args["format"] = format_id
                
                resize_props.update({
                    "WobblyDepthEnabled": True,
                    "WobblyDepthBits": bits,
                    "WobblyDepthFloat": float_samples,
                    "WobblyDepthDither": dither
                })
                
            # Update all frame properties
            for n in frame_props:
                frame_props[n].update(resize_props) # type: ignore
                
            # Apply resize
            resize_filter = getattr(core.resize, resize_filter_name)
            clip = resize_filter(clip=clip, **resize_args)
                
        return clip, frame_props, frame_mapping