"""
Utility functions for Wobbly parser.
"""

from typing import Dict, List, Set, Tuple, Optional, TypeVar, Generic
from dataclasses import dataclass

from .types import ProjectData, CycleDecimationDict, DecimationRange, DecimationRangeList

# Generic type parameter
T = TypeVar('T')


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


def get_decimation_info(project: ProjectData) -> Tuple[CycleDecimationDict, DecimationRangeList]:
    """
    Get decimation cycle information from the project
    
    Args:
        project: Wobbly project data
        
    Returns:
        Tuple of (decimated_by_cycle, ranges)
    """
    # Get decimated frames and project length
    decimated_frames: List[int] = project.get('decimated frames', [])
    
    # Calculate total frames from trim data
    num_frames = 0
    if 'trim' in project:
        for trim in project['trim']:
            if isinstance(trim, list) and len(trim) >= 2:
                num_frames += trim[1] - trim[0] + 1
                
    # Group decimated frames by cycle
    decimated_by_cycle: Dict[int, Set[int]] = {}
    for frame in decimated_frames:
        cycle = frame // 5
        if cycle not in decimated_by_cycle:
            decimated_by_cycle[cycle] = set()
        decimated_by_cycle[cycle].add(frame % 5)
        
    # Calculate decimation ranges
    ranges: List[DecimationRange] = []
    current_count = -1
    current_start = 0
    
    for cycle in range((num_frames + 4) // 5):
        count = len(decimated_by_cycle.get(cycle, set()))
        if count != current_count:
            if current_count != -1:
                ranges.append(DecimationRange(
                    start=current_start,
                    end=cycle * 5,
                    dropped=current_count
                ))
            current_count = count
            current_start = cycle * 5
            
    if current_count != -1:
        ranges.append(DecimationRange(
            start=current_start,
            end=num_frames,
            dropped=current_count
        ))
        
    return decimated_by_cycle, ranges


def frame_number_after_decimation(frame: int, decimated_by_cycle: CycleDecimationDict) -> int:
    """
    Calculate frame number after decimation
    
    Args:
        frame: Original frame number
        decimated_by_cycle: Dictionary mapping cycles to sets of decimated offsets
        
    Returns:
        Frame number after decimation
    """
    if frame < 0:
        return 0
        
    cycle = frame // 5
    offset = frame % 5
    
    # Count decimated frames before this one
    decimated_before = 0
    for c in range(cycle):
        decimated_before += len(decimated_by_cycle.get(c, set()))
        
    for o in range(offset):
        if o in decimated_by_cycle.get(cycle, set()):
            decimated_before += 1
            
    return frame - decimated_before