"""
JSON handling with orjson optimization.
"""

from pathlib import Path
from typing import Any, Union
import orjson
from ..types import PathLike, ProjectData, Result


def load_json(file_path: Union[str, Path]) -> dict[str, Any]:
    """
    Load JSON file with orjson for better performance

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def dump_json(data: dict[str, Any]) -> bytes:
    """
    Dump data to JSON string with orjson

    Args:
        data: Data to serialize

    Returns:
        JSON bytes
    """
    return orjson.dumps(data)


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
