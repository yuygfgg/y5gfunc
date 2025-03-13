"""
WobblySource class implementation.
"""

import os
from pathlib import Path
from typing import Optional

import vapoursynth as vs

from ..types import (
    WobblyKeys, PathLike, ProjectData, FrameMap, 
    FramePropertyMap, PresetDict, ProcessPosition, TimecodeVersion
)
from ..errors import WobblyError, WobblyParseError, WobblyProcessError, WobblyInputError
from ..processors import CropProcessor, TrimProcessor, CustomListProcessor, SectionProcessor, FrozenFramesProcessor, DecimationProcessor, SpecialFrameMarkProcessor, ResizeProcessor
from ..io import load_source_video, apply_frame_properties, load_project
from ..timecodes import TimecodeGeneratorFactory
from .context import safe_processing
from .presets import create_preset_functions


def load_and_process(
    wob_project_path: PathLike,
    timecode_output_path: Optional[PathLike] = None,
    timecode_version: str = TimecodeVersion.V2
) -> vs.VideoNode:
    """
    Load and process a Wobbly project file
    
    Args:
        wob_project_path: Path to Wobbly project file
        timecode_output_path: Optional path for timecode output
        timecode_version: Timecode version, defaults to "v2"
        
    Returns:
        Processed video node
    """
    source = WobblySource(wob_project_path, timecode_output_path, timecode_version)
    return source.process()


class WobblySource:
    """Wobbly source processing class"""
    
    def __init__(
        self,
        wob_project_path: PathLike,
        timecode_output_path: Optional[PathLike] = None,
        timecode_version: str = TimecodeVersion.V2
    ):
        """
        Initialize Wobbly source processor
        
        Args:
            wob_project_path: Path to Wobbly project file
            timecode_output_path: Optional path for timecode output
            timecode_version: Timecode version, defaults to "v2"
        """
        self.project_path = Path(wob_project_path)
        self.timecode_path = Path(timecode_output_path) if timecode_output_path else None
        self.timecode_version = timecode_version
        self.keys = WobblyKeys()
        
        # Initialize state
        self.project: Optional[ProjectData] = None
        self.input_file: Optional[str] = None
        self.source_filter: str = ""
        self.frame_props: FramePropertyMap = {}
        self.frame_mapping: FrameMap = {}
        self.presets: PresetDict = {}
        
    def load(self) -> 'WobblySource':
        """
        Load project file
        
        Returns:
            Self instance for method chaining
            
        Raises:
            WobblyParseError: When project file cannot be parsed
            WobblyInputError: When there are issues with input file
        """
        # Load project file
        result = load_project(self.project_path)
        if not result.success:
            raise WobblyParseError(result.error or "Unknown parsing error")
            
        self.project = result.value
        Keys = self.keys
        
        assert self.project
        
        # Get input file path
        self.input_file = self.project.get(Keys.project.input_file)
        self.source_filter = self.project.get(Keys.project.source_filter, "")
        
        if not self.input_file:
            raise WobblyInputError("No input file specified in the project")
            
        # Handle relative paths
        if not os.path.isabs(self.input_file):
            wob_dir = os.path.dirname(os.path.abspath(str(self.project_path)))
            self.input_file = os.path.join(wob_dir, self.input_file)
            
        return self
        
    def create_presets(self) -> 'WobblySource':
        """
        Create preset functions
        
        Returns:
            Self instance for method chaining
        """
        if not self.project:
            raise WobblyProcessError("Project not loaded", stage="create_presets")
            
        self.presets = create_preset_functions(self.project, self.keys)
        return self
    
    def init_properties(self, src: vs.VideoNode) -> None:
        """
        Initialize frame properties and mapping
        
        Args:
            src: Source video clip
        """
        # Initialize basic properties for each frame
        
        assert self.project
        
        for n in range(src.num_frames):
            self.frame_props[n] = {
                "WobblyProject": os.path.basename(str(self.project_path)),
                "WobblyVersion": self.project.get(self.keys.project.wobbly_version, ""),
                "WobblySourceFilter": self.source_filter,
                # Initialize empty values
                "WobblyCustomList": "",
                "WobblyCustomListPreset": "",
                "WobblyCustomListPosition": "",
                "WobblySectionStart": -1,
                "WobblySectionEnd": -1,
                "WobblySectionPresets": "",
                "WobblyMatch": ""
            }
            
        # Prepare data for processing
        for i in range(src.num_frames):
            self.frame_mapping[i] = i  # Initially the mapping is identical
    
    def generate_timecodes(self) -> None:
        """
        Generate timecodes file if requested
        """
        if not self.timecode_path or not self.project:
            return
            
        with safe_processing("timecode_generation"):
            # Use the timecode generator factory to create the appropriate generator
            generator = TimecodeGeneratorFactory.create(
                self.timecode_version, self.project
            )
            timecodes = generator.generate()
            
            with open(str(self.timecode_path), 'w', encoding='utf-8') as f:
                f.write(timecodes)
        
    def process(self) -> vs.VideoNode:
        """
        Process the Wobbly project
        
        Returns:
            Processed video node
            
        Raises:
            WobblyProcessError: When errors occur during processing
        """
        if not self.project or not self.input_file:
            self.load()
            
        if not self.presets:
            self.create_presets()
        
        assert self.input_file
        
        # Load source video
        src = load_source_video(self.input_file, self.source_filter)
        
        # Initialize properties
        self.init_properties(src)
            
        # Process video pipeline
        try:
            assert self.project
            # Apply early crop
            crop_processor = CropProcessor(early=True)
            src, self.frame_props, self.frame_mapping = crop_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply trimming
            trim_processor = TrimProcessor()
            src, self.frame_props, self.frame_mapping = trim_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply custom lists - PostSource
            custom_list_processor = CustomListProcessor(ProcessPosition.POST_SOURCE)
            src, self.frame_props, self.frame_mapping = custom_list_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Process match information
            src = self._process_match_information(src)
            
            # Apply custom lists - PostFieldMatch
            custom_list_processor = CustomListProcessor(ProcessPosition.POST_FIELD_MATCH)
            src, self.frame_props, self.frame_mapping = custom_list_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply sections and record section info
            section_processor = SectionProcessor()
            src, self.frame_props, self.frame_mapping = section_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Mark special frames
            special_frames_processor = SpecialFrameMarkProcessor()
            src, self.frame_props, self.frame_mapping = special_frames_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply frozen frames
            frozen_frames_processor = FrozenFramesProcessor()
            src, self.frame_props, self.frame_mapping = frozen_frames_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply frame rate conversion (delete frames)
            decimation_processor = DecimationProcessor()
            src, self.frame_props, self.frame_mapping = decimation_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply custom lists - PostDecimate
            custom_list_processor = CustomListProcessor(ProcessPosition.POST_DECIMATE)
            src, self.frame_props, self.frame_mapping = custom_list_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply final crop
            crop_processor = CropProcessor(early=False)
            src, self.frame_props, self.frame_mapping = crop_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Apply resize and bit depth
            resize_processor = ResizeProcessor()
            src, self.frame_props, self.frame_mapping = resize_processor.process(
                src, self.project, self.frame_props, self.frame_mapping, self.presets
            )
            
            # Generate timecodes if requested
            self.generate_timecodes()
            
            # Finally: Apply all frame properties
            src = apply_frame_properties(src, self.frame_props)
                
        except Exception as e:
            if isinstance(e, WobblyError):
                raise
            else:
                raise WobblyProcessError("Error processing Wobbly project", 
                                        stage="processing",
                                        cause=e)
                
        return src
    
    def _process_match_information(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Process match information and apply FieldHint
        
        Args:
            clip: Input video clip
            
        Returns:
            Processed clip
        """
        
        assert self.project
        
        with safe_processing("match_processing"):
            Keys = self.keys
            matches_list = self.project.get(Keys.project.matches)
            original_matches_list = self.project.get(Keys.project.original_matches)
            
            # Ensure we have match data for later
            matches = ""
            if matches_list:
                matches = "".join(matches_list)
            elif original_matches_list:
                matches = "".join(original_matches_list)
                
            # Record match for each frame
            if matches:
                for n in self.frame_props:
                    orig_frame = self.frame_mapping[n]
                    if orig_frame < len(matches):
                        self.frame_props[n]["WobblyMatch"] = matches[orig_frame]
                        
            # Apply FieldHint
            if hasattr(vs.core, 'fh') and matches:
                vfm_params = self.project.get(Keys.project.vfm_parameters, {})
                order = vfm_params.get(Keys.vfm_parameters.order, 1)
                clip = vs.core.fh.FieldHint(clip=clip, tff=order, matches=matches)
                
        return clip