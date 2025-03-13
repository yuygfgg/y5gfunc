"""
Processor base classes and protocol definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Tuple

import vapoursynth as vs

from ..types import FramePropertyMap, FrameMap, ProjectData, PresetDict


class ProcessorProtocol(Protocol):
    """Processor interface protocol"""
    
    def process(
        self, 
        clip: vs.VideoNode, 
        project: ProjectData,
        frame_props: FramePropertyMap,
        frame_mapping: FrameMap,
        presets: PresetDict
    ) -> Tuple[vs.VideoNode, FramePropertyMap, FrameMap]:
        """
        Process a video clip
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        ...
    
    
class BaseProcessor(ABC):
    """Base processor class"""
    
    @abstractmethod
    def process(
        self, 
        clip: vs.VideoNode, 
        project: ProjectData,
        frame_props: FramePropertyMap,
        frame_mapping: FrameMap,
        presets: PresetDict
    ) -> Tuple[vs.VideoNode, FramePropertyMap, FrameMap]:
        """
        Abstract method for processing a video clip
        
        Args:
            clip: Input video clip
            project: Project data
            frame_props: Frame property mapping
            frame_mapping: Frame number mapping
            presets: Preset function dictionary
            
        Returns:
            Processed clip, updated frame properties and frame mapping
        """
        pass


@dataclass
class ProcessorResult:
    """Processor result data class"""
    clip: vs.VideoNode
    frame_props: FramePropertyMap
    frame_mapping: FrameMap