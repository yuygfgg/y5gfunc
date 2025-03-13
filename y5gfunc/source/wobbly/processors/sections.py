"""
Section processor implementation.
"""

from typing import Dict, List, Tuple, Any

import vapoursynth as vs

from .base import BaseProcessor
from ..types import ProjectData, FramePropertyMap, FrameMap, PresetDict, WobblyKeys
from ..core.context import safe_processing

core = vs.core


class SectionProcessor(BaseProcessor):
    """Section processor implementation"""
    
    def __init__(self):
        """Initialize section processor"""
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
        Process sections
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        with safe_processing("sections_processing"):
            Keys = self.keys
            sections_list = project.get(Keys.project.sections, [])
            
            if not sections_list:
                return clip, frame_props, frame_mapping
                
            # Sort sections by start frame
            sorted_sections = sorted(sections_list, key=lambda s: s.get(Keys.sections.start, 0))
            
            # Mark each frame with its section
            self._mark_section_frames(clip, sorted_sections, frame_props, frame_mapping, Keys)
                
            # Apply presets and splice
            sections = []
            new_frame_props: Dict[int, Dict[str, Any]] = {}
            new_frame_idx = 0
            
            for i, section_info in enumerate(sorted_sections):
                start = section_info.get(Keys.sections.start, 0)
                next_start = (sorted_sections[i+1].get(Keys.sections.start, clip.num_frames) 
                             if i+1 < len(sorted_sections) else clip.num_frames)
                              
                # Apply presets
                section_clip = clip[start:next_start]
                for preset_name in section_info.get(Keys.sections.presets, []):
                    if preset_name in presets:
                        section_clip = presets[preset_name](section_clip)
                        
                # Update frame mapping and properties
                for j in range(section_clip.num_frames):
                    src_idx = start + j
                    if src_idx < len(frame_mapping):
                        orig_frame = frame_mapping[src_idx]
                        # Copy original frame properties
                        if src_idx in frame_props:
                            new_frame_props[new_frame_idx] = frame_props[src_idx].copy() # type: ignore
                            # Update mapping
                            frame_mapping[new_frame_idx] = orig_frame
                            new_frame_idx += 1
                            
                sections.append(section_clip)
                
            # Merge all sections
            if sections:
                clip = core.std.Splice(clips=sections, mismatch=True)
                frame_props = new_frame_props  # type: ignore # Update frame properties
                
        return clip, frame_props, frame_mapping
    
    def _mark_section_frames(
        self, 
        clip: vs.VideoNode,
        sorted_sections: List[Dict[str, Any]],
        frame_props: FramePropertyMap,
        frame_mapping: FrameMap,
        Keys: WobblyKeys
    ) -> None:
        """Mark frames with section information"""
        for i, section_info in enumerate(sorted_sections):
            start = section_info.get(Keys.sections.start, 0)
            next_start = (sorted_sections[i+1].get(Keys.sections.start, clip.num_frames) 
                         if i+1 < len(sorted_sections) else clip.num_frames)
                          
            section_presets = section_info.get(Keys.sections.presets, [])
            presets_str = ",".join(section_presets)
            
            # Update frame properties
            for n in frame_props:
                orig_frame = frame_mapping[n]
                if start <= orig_frame < next_start:
                    frame_props[n].update({
                        "WobblySectionStart": start,
                        "WobblySectionEnd": next_start-1,
                        "WobblySectionPresets": presets_str
                    })