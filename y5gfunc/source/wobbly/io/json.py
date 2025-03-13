"""
JSON handling with orjson optimization.
"""

from pathlib import Path
from typing import Dict, Any, Union

# Try to import orjson for better performance, fall back to standard json
try:
    import orjson
    
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file with orjson for better performance
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'rb') as f:
            return orjson.loads(f.read())
    
    def dump_json(data: Dict[str, Any]) -> bytes: # type: ignore
        """
        Dump data to JSON string with orjson
        
        Args:
            data: Data to serialize
            indent: Whether to indent the output (ignored in orjson)
            
        Returns:
            JSON bytes
        """
        return orjson.dumps(data)
        
except ImportError:
    import json
    
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file with standard json
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def dump_json(data: Dict[str, Any], indent: bool = False) -> str:
        """
        Dump data to JSON string with standard json
        
        Args:
            data: Data to serialize
            indent: Whether to indent the output
            
        Returns:
            JSON string
        """
        indent_val = 2 if indent else None
        return json.dumps(data, indent=indent_val)


from ..types import PathLike, ProjectData, Result


def load_project(project_path: PathLike) -> Result[ProjectData]:
    """
    Load and parse a Wobbly project file
    
    Args:
        project_path: Path to Wobbly project file
        
    Returns:
        Result containing project data if successful
    """
    try:
        project = load_json(project_path)
        return Result.ok(project)
    except Exception as e:
        return Result.err(f"Failed to read or parse Wobbly project file: {e}")