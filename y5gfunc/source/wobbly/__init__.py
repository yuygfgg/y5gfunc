"""
WobblyParser - A VapourSynth parser for Wobbly project files (.wob)

This module provides functionality to directly load and process Wobbly projects
in VapourSynth scripts without needing to export .vpy files.
"""

__version__ = "0.2.0"

# Export main functionality
from .core import WobblySource, wobbly_source, load_and_process
from .types import (
    FrameProperties, 
    FieldMatchOrder, 
    ProcessPosition, 
    ProjectData,
    VideoResult
)
from .errors import WobblyError, WobblyParseError, WobblyProcessError


__all__ = [
    "WobblySource", 
    "wobbly_source",
    "load_and_process",
    "FrameProperties",
    "FieldMatchOrder",
    "ProcessPosition",
    "ProjectData",
    "VideoResult",
    "WobblyError",
    "WobblyParseError",
    "WobblyProcessError"
]