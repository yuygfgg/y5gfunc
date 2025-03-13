"""
Custom list processor implementation.
"""

from typing import List, Tuple
import vapoursynth as vs

from .base import BaseProcessor
from ..types import (
    ProjectData, FramePropertyMap, FrameMap, PresetDict, 
    ProcessPosition, CustomListRanges, WobblyKeys
)
from ..errors import WobblyProcessError


class CustomListProcessor(BaseProcessor):
    """Custom list processor implementation"""
    
    def __init__(self, position: ProcessPosition):
        """
        Initialize custom list processor
        
        Args:
            position: Processing position
        """
        self.position = position
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
        Process custom lists
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        core = vs.core
        Keys = self.keys
        
        try:
            # Filter custom lists for the specified position
            custom_lists = [cl for cl in project.get(Keys.project.custom_lists, []) 
                            if cl.get(Keys.custom_lists.position) == self.position.value]
            
            if not custom_lists:
                return clip, frame_props, frame_mapping
                
            all_ranges: CustomListRanges = []  # all covered ranges
            
            for cl_info in custom_lists:
                cl_name = cl_info.get(Keys.custom_lists.name)
                cl_preset = cl_info.get(Keys.custom_lists.preset)
                cl_frames = cl_info.get(Keys.custom_lists.frames, [])
                
                # Check if we have preset and frame ranges
                if not cl_preset or not cl_frames:
                    continue
                    
                # Check if preset exists
                if cl_preset not in presets:
                    continue
                    
                # Ensure cl_frames is a list of lists
                ranges: List[Tuple[int, int]] = []
                for frame_range in cl_frames:
                    if isinstance(frame_range, list) and len(frame_range) == 2:
                        start, end = frame_range
                        
                        # Record all qualifying frames, and update frame properties
                        self._update_frame_properties(
                            frame_props, frame_mapping, start, end, cl_name, cl_preset
                        )
                        
                        ranges.append((start, end))
                        all_ranges.append((start, end, cl_name, cl_preset))
                
                # Sort the ranges
                ranges.sort()
                
                # Apply preset to clip
                if ranges:
                    # Create marked segments
                    marked_clips = []
                    last_end = 0
                    
                    for range_start, range_end in ranges:
                        # Ensure valid range
                        if not (0 <= range_start <= range_end < clip.num_frames):
                            continue
                            
                        if range_start > last_end:
                            marked_clips.append(clip[last_end:range_start])
                            
                        # Apply preset to current range
                        list_clip = presets[cl_preset](clip[range_start:range_end+1])
                        marked_clips.append(list_clip)
                        
                        last_end = range_end + 1
                        
                    if last_end < clip.num_frames:
                        marked_clips.append(clip[last_end:])
                        
                    if marked_clips:
                        clip = core.std.Splice(clips=marked_clips, mismatch=True)
            
            return clip, frame_props, frame_mapping
            
        except Exception as e:
            raise WobblyProcessError(
                f"Error applying custom lists at position {self.position.value}",
                stage="custom_list_processing", 
                details={"position": self.position.value},
                cause=e
            )
    
    def _update_frame_properties(
        self,
        frame_props: FramePropertyMap,
        frame_mapping: FrameMap,
        start: int,
        end: int,
        cl_name: str,
        cl_preset: str
    ) -> None:
        """
        Update properties for frames in the specified range
        
        Args:
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            start: Start frame
            end: End frame
            cl_name: Custom list name
            cl_preset: Preset name
        """
        for frame in range(start, end + 1):
            for n in frame_props:
                if frame_mapping[n] == frame:
                    frame_props[n].update({
                        "WobblyCustomList": cl_name,
                        "WobblyCustomListPreset": cl_preset,
                        "WobblyCustomListPosition": self.position.value
                    })