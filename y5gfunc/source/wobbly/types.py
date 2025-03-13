"""
Advanced type definitions for Wobbly parser.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Generic, Optional, TypedDict, Dict, List, Tuple, Set, Any, Union, Callable, 
    Protocol, TypeVar
)
from pathlib import Path
import vapoursynth as vs


# Generic type parameters
T = TypeVar('T')
R = TypeVar('R')
PathLike = Union[str, Path]


class FieldMatchOrder(Enum):
    """Field match order enumeration"""
    TFF = 1  # Top Field First
    BFF = 0  # Bottom Field First


class ProcessPosition(Enum):
    """Processing position enumeration"""
    POST_SOURCE = "post source"
    POST_FIELD_MATCH = "post field match"
    POST_DECIMATE = "post decimate"


class VideoFormat(Enum):
    """Video format types"""
    INTEGER = vs.INTEGER
    FLOAT = vs.FLOAT


class ResizeFilterType(Enum):
    """Resize filter type"""
    BILINEAR = "Bilinear"
    BICUBIC = "Bicubic"
    LANCZOS = "Lanczos"
    SPLINE16 = "Spline16"
    SPLINE36 = "Spline36"
    POINT = "Point"


class FrameProperties(TypedDict, total=False):
    """Frame property dictionary type"""
    WobblyProject: str
    WobblyVersion: str
    WobblySourceFilter: str
    WobblyCustomList: str
    WobblyCustomListPreset: str
    WobblyCustomListPosition: str
    WobblySectionStart: int
    WobblySectionEnd: int
    WobblySectionPresets: str
    WobblyMatch: str
    WobblyCombed: bool
    WobblyInterlacedFade: bool
    WobblyFieldDifference: float
    WobblyOrphan: bool
    WobblyOrphanType: str
    WobblyOrphanDecimated: bool
    WobblyFrozenFrame: bool
    WobblyFrozenSource: int
    WobblyDecimated: bool
    WobblyCropEarly: bool
    WobblyCropLeft: int
    WobblyCropTop: int
    WobblyCropRight: int
    WobblyCropBottom: int
    WobblyResizeEnabled: bool
    WobblyResizeWidth: int
    WobblyResizeHeight: int
    WobblyResizeFilter: str
    WobblyDepthEnabled: bool
    WobblyDepthBits: int
    WobblyDepthFloat: bool
    WobblyDepthDither: str
    WobblyTrimStart: int
    WobblyTrimEnd: int


@dataclass
class ProjectKey:
    """Project key constants"""
    wobbly_version: str = "wobbly version"
    project_format_version: str = "project format version"
    input_file: str = "input file"
    input_frame_rate: str = "input frame rate"
    input_resolution: str = "input resolution"
    trim: str = "trim"
    source_filter: str = "source filter"
    user_interface: str = "user interface"
    vfm_parameters: str = "vfm parameters"
    matches: str = "matches"
    original_matches: str = "original matches"
    sections: str = "sections"
    presets: str = "presets"
    frozen_frames: str = "frozen frames"
    combed_frames: str = "combed frames"
    interlaced_fades: str = "interlaced fades"
    decimated_frames: str = "decimated frames"
    custom_lists: str = "custom lists"
    resize: str = "resize"
    crop: str = "crop"
    depth: str = "depth"


@dataclass
class VFMParametersKey:
    """VFM parameters key constants"""
    order: str = "order"


@dataclass
class SectionsKey:
    """Sections key constants"""
    start: str = "start"
    presets: str = "presets"


@dataclass
class PresetsKey:
    """Presets key constants"""
    name: str = "name"
    contents: str = "contents"


@dataclass
class CustomListsKey:
    """Custom lists key constants"""
    name: str = "name"
    preset: str = "preset"
    position: str = "position"
    frames: str = "frames"


@dataclass
class ResizeKey:
    """Resize key constants"""
    width: str = "width"
    height: str = "height"
    filter: str = "filter"
    enabled: str = "enabled"


@dataclass
class CropKey:
    """Crop key constants"""
    early: str = "early"
    left: str = "left"
    top: str = "top"
    right: str = "right"
    bottom: str = "bottom"
    enabled: str = "enabled"


@dataclass
class DepthKey:
    """Depth key constants"""
    bits: str = "bits"
    float_samples: str = "float samples"
    dither: str = "dither"
    enabled: str = "enabled"


@dataclass
class InterlacedFadesKey:
    """Interlaced fades key constants"""
    frame: str = "frame"
    field_difference: str = "field difference"


@dataclass
class WobblyKeys:
    """Key constant collection for Wobbly project JSON structure"""
    # Using composition instead of inheritance
    project: ProjectKey = field(default_factory=ProjectKey)
    vfm_parameters: VFMParametersKey = field(default_factory=VFMParametersKey)  
    sections: SectionsKey = field(default_factory=SectionsKey)
    presets: PresetsKey = field(default_factory=PresetsKey)
    custom_lists: CustomListsKey = field(default_factory=CustomListsKey)
    resize: ResizeKey = field(default_factory=ResizeKey)
    crop: CropKey = field(default_factory=CropKey)
    depth: DepthKey = field(default_factory=DepthKey)
    interlaced_fades: InterlacedFadesKey = field(default_factory=InterlacedFadesKey)


@dataclass
class InterlacedFade:
    """Interlaced fade information"""
    frame: int
    field_difference: float = 0.0


@dataclass
class CropSettings:
    """Crop settings"""
    enabled: bool = False
    early: bool = False
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


@dataclass
class ResizeSettings:
    """Resize settings"""
    enabled: bool = False
    width: int = 0
    height: int = 0
    filter: ResizeFilterType = ResizeFilterType.BICUBIC


@dataclass
class DepthSettings:
    """Depth settings"""
    enabled: bool = False
    bits: int = 8
    float_samples: bool = False
    dither: str = ""


@dataclass
class SectionInfo:
    """Section information"""
    start: int
    presets: List[str] = field(default_factory=list)


@dataclass
class PresetInfo:
    """Preset information"""
    name: str
    contents: str


@dataclass
class CustomListInfo:
    """Custom list information"""
    name: str
    preset: str
    position: ProcessPosition
    frames: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class FrozenFrameInfo:
    """Frozen frame information"""
    first: int
    last: int
    replacement: int


@dataclass
class DecimationRange:
    """Decimation range"""
    start: int
    end: int
    dropped: int


@dataclass
class OrphanFieldInfo:
    """Orphan field information"""
    type: str
    decimated: bool


# Preset processing function type
PresetFunction = Callable[[vs.VideoNode], vs.VideoNode]

ProjectData = Dict[str, Any]
FrameMap = Dict[int, int]
FramePropertyMap = Dict[int, FrameProperties]
DecimatedFrameSet = Set[int]
MatchString = str
PresetDict = Dict[str, PresetFunction]
OrphanFieldDict = Dict[int, OrphanFieldInfo]
CycleDecimationDict = Dict[int, Set[int]]
DecimationRangeList = List[DecimationRange]
CustomListRanges = List[Tuple[int, int, str, str]]

class TimecodeVersion:
    V1 = "v1"
    V2 = "v2"
    
    @classmethod
    def is_valid(cls, version: str) -> bool:
        return version in (cls.V1, cls.V2)

class VideoResult(Protocol):
    """Video processing result protocol"""
    clip: vs.VideoNode
    frame_props: FramePropertyMap
    frame_mapping: FrameMap

@dataclass
class Result(Generic[T]):
    """Operation result container"""
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T]':
        """Create success result"""
        return cls(success=True, value=value)
    
    @classmethod
    def err(cls, error: str) -> 'Result[T]':
        """Create error result"""
        return cls(success=False, error=error)