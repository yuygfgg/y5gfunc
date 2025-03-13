"""
Trim processor implementation.
"""

from typing import Dict, List, Tuple, Any

import vapoursynth as vs

from .base import BaseProcessor
from ..types import ProjectData, FramePropertyMap, FrameMap, PresetDict, WobblyKeys
from ..errors import WobblyProcessError
from ..core.context import safe_processing

core = vs.core


class TrimProcessor(BaseProcessor):
    """Trim processor implementation"""
    
    def __init__(self):
        """Initialize trim processor"""
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
        Process trim operation
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        with safe_processing("trim_processing"):
            Keys = self.keys
            trim_list = project.get(Keys.project.trim, [])
            
            if not trim_list:
                return clip, frame_props, frame_mapping
                
            segments = []
            new_frame_props: Dict[int, Dict[str, Any]] = {}  # Store new frame property map
            new_frame_idx = 0
            
            for trim in trim_list:
                first, last = trim
                if first <= last and first < clip.num_frames and last < clip.num_frames:
                    # Create segment
                    segment = clip[first:last+1]
                    
                    # Update frame properties and mapping
                    for i in range(first, last+1):
                        # Update trim info
                        if i in frame_props:
                            props = frame_props[i].copy()
                            props.update({
                                "WobblyTrimStart": first,
                                "WobblyTrimEnd": last
                            })
                            new_frame_props[new_frame_idx] = props
                            # Update mapping
                            frame_mapping[new_frame_idx] = i
                            new_frame_idx += 1
                            
                    segments.append(segment)
                    
            if segments:
                clip = core.std.Splice(clips=segments)
                frame_props = new_frame_props  # Update frame property map
                
        return clip, frame_props, frame_mapping