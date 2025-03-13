"""
Preset handling for Wobbly parser.
"""


import vapoursynth as vs

from ..types import PresetDict, ProjectData, WobblyKeys
from .context import safe_processing


def create_preset_functions(project: ProjectData, keys: WobblyKeys) -> PresetDict:
    """
    Create preset functions from project data
    
    Args:
        project: Project data
        keys: Keys for accessing project data
        
    Returns:
        Dictionary mapping preset names to preset functions
    """
    presets: PresetDict = {}
    core = vs.core
    
    with safe_processing("preset_creation"):
        # Create preset functions
        for preset_info in project.get(keys.project.presets, []):
            preset_name = preset_info.get(keys.presets.name)
            preset_contents = preset_info.get(keys.presets.contents)
            
            if not preset_name or preset_contents is None:
                continue
                
            try:
                # Create executable preset function
                exec_globals = {'vs': vs, 'core': core, 'c': core}
                exec(f"def preset_{preset_name}(clip):\n" +
                     "\n".join("    " + line for line in preset_contents.split('\n')) +
                     "\n    return clip", exec_globals)
                     
                presets[preset_name] = exec_globals[f"preset_{preset_name}"]
            except Exception as e:
                print(f"Warning: Error creating preset '{preset_name}': {e}")
                
    return presets