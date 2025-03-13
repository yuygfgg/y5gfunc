"""
Special frames processing (frozen frames, decimation).
"""

from typing import Dict, List, Set, Tuple, Any

import vapoursynth as vs

from .base import BaseProcessor
from ..types import ProjectData, FramePropertyMap, FrameMap, PresetDict, WobblyKeys
from ..errors import WobblyProcessError
from ..core.context import safe_processing

core = vs.core


class FrozenFramesProcessor(BaseProcessor):
    """Frozen frames processor implementation"""
    
    def __init__(self):
        """Initialize frozen frames processor"""
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
        Process frozen frames
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        with safe_processing("frozen_frames_processing"):
            Keys = self.keys
            frozen_frames_list = project.get(Keys.project.frozen_frames, [])
            
            if not (frozen_frames_list and hasattr(core.std, 'FreezeFrames')):
                return clip, frame_props, frame_mapping
                
            # Gather frame lists
            first_frames = []
            last_frames = []
            replacement_frames = []
            
            for ff_info in frozen_frames_list:
                if len(ff_info) == 3:
                    first, last, replacement = ff_info
                    if 0 <= first <= last < clip.num_frames and 0 <= replacement < clip.num_frames:
                        first_frames.append(first)
                        last_frames.append(last)
                        replacement_frames.append(replacement)
                        
                        # Record frozen frame info
                        for i in range(first, last+1):
                            if i in frame_props:
                                frame_props[i]["WobblyFrozenFrame"] = True
                                frame_props[i]["WobblyFrozenSource"] = replacement
                                
            if first_frames:
                clip = core.std.FreezeFrames(
                    clip=clip,
                    first=first_frames,
                    last=last_frames,
                    replacement=replacement_frames
                )
                
        return clip, frame_props, frame_mapping


class DecimationProcessor(BaseProcessor):
    """Decimation processor implementation"""
    
    def __init__(self):
        """Initialize decimation processor"""
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
        Process frame decimation
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        with safe_processing("decimation_processing"):
            Keys = self.keys
            decimated_frames_list = project.get(Keys.project.decimated_frames, [])
            
            if not decimated_frames_list:
                return clip, frame_props, frame_mapping
                
            # Filter valid frames
            frames_to_delete = [f for f in decimated_frames_list if 0 <= f < clip.num_frames]
            
            if not frames_to_delete:
                return clip, frame_props, frame_mapping
                
            # Create new frame property map
            new_frame_props: Dict[int, Dict[str, Any]] = {}
            new_idx = 0
            
            for n in range(clip.num_frames):
                orig_frame = frame_mapping.get(n, n)
                
                # If not a frame to delete
                if orig_frame not in frames_to_delete:
                    if n in frame_props:
                        new_frame_props[new_idx] = frame_props[n].copy() # type: ignore
                        new_idx += 1
                        
            # Delete frames
            clip = core.std.DeleteFrames(clip=clip, frames=frames_to_delete)
            frame_props = new_frame_props  # type: ignore # Update frame properties
                
        return clip, frame_props, frame_mapping


class SpecialFrameMarkProcessor(BaseProcessor):
    """Mark special frames processor implementation"""
    
    def __init__(self):
        """Initialize special frame marker"""
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
        Mark special frames
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        with safe_processing("special_frames_processing"):
            Keys = self.keys
            combed_frames = set(project.get(Keys.project.combed_frames, []))
            decimated_frames = set(project.get(Keys.project.decimated_frames, []))
            matches = self._get_matches_string(project, Keys)
            sections_list = project.get(Keys.project.sections, [])
            
            # Process Interlaced Fades
            interlaced_fades = project.get(Keys.project.interlaced_fades, [])
            fade_dict = self._process_interlaced_fades(interlaced_fades, Keys)
            
            # Identify orphan fields
            orphan_fields = self._identify_orphan_fields(matches, sections_list, decimated_frames, 
                                                        clip.num_frames, Keys)
            
            # Update special frame properties
            for n in frame_props:
                orig_frame = frame_mapping[n]
                props = frame_props[n]
                
                # Mark combed frames
                if orig_frame in combed_frames:
                    props["WobblyCombed"] = True
                    
                # Mark interlaced fades
                if orig_frame in fade_dict:
                    props["WobblyInterlacedFade"] = True
                    props["WobblyFieldDifference"] = fade_dict[orig_frame]
                    
                # Mark orphan fields
                if orig_frame in orphan_fields:
                    info = orphan_fields[orig_frame]
                    props["WobblyOrphan"] = True
                    props["WobblyOrphanType"] = info['type']
                    props["WobblyOrphanDecimated"] = info['decimated']
                    
                # Mark decimated frames
                if orig_frame in decimated_frames:
                    props["WobblyDecimated"] = True
                
        return clip, frame_props, frame_mapping
    
    def _get_matches_string(self, project: ProjectData, Keys: WobblyKeys) -> str:
        """Get the matches string from project data"""
        matches_list = project.get(Keys.project.matches)
        original_matches_list = project.get(Keys.project.original_matches)
        
        # Ensure we have match data for later
        matches = ""
        if matches_list:
            matches = "".join(matches_list)
        elif original_matches_list:
            matches = "".join(original_matches_list)
            
        return matches
    
    def _process_interlaced_fades(self, interlaced_fades: List[Dict[str, Any]], 
                                  Keys: WobblyKeys) -> Dict[int, float]:
        """Process interlaced fades data"""
        fade_dict: Dict[int, float] = {}
        
        if interlaced_fades:
            for fade in interlaced_fades:
                frame = fade.get(Keys.interlaced_fades.frame)
                field_diff = fade.get(Keys.interlaced_fades.field_difference, 0)
                if frame is not None:
                    fade_dict[frame] = field_diff
                    
        return fade_dict
    
    def _identify_orphan_fields(self, matches: str, 
                               sections_list: List[Dict[str, Any]],
                               decimated_frames: Set[int],
                               num_frames: int,
                               Keys: WobblyKeys) -> Dict[int, Dict[str, Any]]:
        """Identify orphan fields in the project"""
        orphan_fields: Dict[int, Dict[str, Any]] = {}
        
        if not matches or not sections_list:
            return orphan_fields
            
        # Sort sections by start frame
        sorted_sections = sorted(sections_list, key=lambda s: s.get(Keys.sections.start, 0))
        section_boundaries = [s.get(Keys.sections.start, 0) for s in sorted_sections]
        section_boundaries.append(num_frames)  # Add last frame as boundary
        
        # Identify orphan fields
        for i in range(len(section_boundaries) - 1):
            section_start = section_boundaries[i]
            section_end = section_boundaries[i+1] - 1
            
            # Check if section start has 'n' match
            if section_start < len(matches) and matches[section_start] == 'n':
                orphan_fields[section_start] = {
                    'type': 'n', 
                    'decimated': section_start in decimated_frames
                }
                
            # Check if section end has 'b' match
            if section_end < len(matches) and matches[section_end] == 'b':
                orphan_fields[section_end] = {
                    'type': 'b', 
                    'decimated': section_end in decimated_frames
                }
                
        return orphan_fields