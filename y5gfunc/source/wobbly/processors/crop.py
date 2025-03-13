"""
Crop processor implementation.
"""

from typing import Tuple

import vapoursynth as vs

from .base import BaseProcessor
from ..types import ProjectData, FramePropertyMap, FrameMap, PresetDict, WobblyKeys
from ..core.context import safe_processing

core = vs.core


class CropProcessor(BaseProcessor):
    """Crop processor implementation"""
    
    def __init__(self, early: bool = True):
        """
        Initialize crop processor
        
        Args:
            early: Whether this is early crop (True) or final crop (False)
        """
        self.early = early
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
        Process crop operation
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        stage_name = "early_crop" if self.early else "final_crop"
        
        with safe_processing(stage_name):
            Keys = self.keys
            crop_info = project.get(Keys.project.crop, {})
            
            # Check if crop is enabled and matches the requested phase (early or final)
            crop_enabled = crop_info.get(Keys.crop.enabled, False)
            crop_is_early = crop_info.get(Keys.crop.early, False)
            
            if crop_enabled and crop_is_early == self.early:
                # Get crop values
                left = crop_info.get(Keys.crop.left, 0)
                top = crop_info.get(Keys.crop.top, 0) 
                right = crop_info.get(Keys.crop.right, 0)
                bottom = crop_info.get(Keys.crop.bottom, 0)
                
                # Create properties to record
                crop_props = {
                    "WobblyCropEarly": self.early,
                    "WobblyCropLeft": left,
                    "WobblyCropTop": top,
                    "WobblyCropRight": right,
                    "WobblyCropBottom": bottom
                }
                
                # Update all frame properties
                for n in frame_props:
                    frame_props[n].update(crop_props) # type: ignore
                
                # Apply crop
                clip = core.std.CropRel(
                    clip=clip,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom
                )
                
        return clip, frame_props, frame_mapping