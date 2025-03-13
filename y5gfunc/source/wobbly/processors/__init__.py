"""
Video processing components for Wobbly parser.
"""

from .base import BaseProcessor, ProcessorResult
from .custom_lists import CustomListProcessor
from .crop import CropProcessor
from .trim import TrimProcessor
from .sections import SectionProcessor
from .special_frames import (
    FrozenFramesProcessor, 
    DecimationProcessor, 
    SpecialFrameMarkProcessor
)
from .resize import ResizeProcessor

__all__ = [
    # Base classes
    "BaseProcessor",
    "ProcessorResult",
    
    # Processors
    "CustomListProcessor",
    "CropProcessor",
    "TrimProcessor",
    "SectionProcessor",
    "FrozenFramesProcessor",
    "DecimationProcessor",
    "SpecialFrameMarkProcessor",
    "ResizeProcessor"
]