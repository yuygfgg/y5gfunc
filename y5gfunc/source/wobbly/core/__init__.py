"""
Core functionality for Wobbly parser.
"""

from .source import WobblySource, load_and_process
from .context import safe_processing
from .presets import create_preset_functions

# Backward compatibility
def wobbly_source(*args, **kwargs):
    """Backward compatible function-based API"""
    return load_and_process(*args, **kwargs)

__all__ = [
    "WobblySource",
    "load_and_process",
    "wobbly_source",
    "safe_processing",
    "create_preset_functions"
]