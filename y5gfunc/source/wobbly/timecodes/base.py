"""
Timecode generator base class and factory.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

from ..types import ProjectData
from ..utils import get_decimation_info


class TimecodeGenerator(ABC):
    """Timecode generator base class"""
    
    def __init__(self, project: ProjectData):
        """
        Initialize timecode generator
        
        Args:
            project: Project data
        """
        self.project = project
        self.decimated_by_cycle, self.ranges = get_decimation_info(project)
    
    @abstractmethod
    def generate(self) -> str:
        """
        Generate timecode string
        
        Returns:
            Timecode string
        """
        pass


class TimecodeGeneratorFactory:
    """Timecode generator factory"""
    
    _generators: Dict[str, Type[TimecodeGenerator]] = {}
    
    @classmethod
    def register(cls, version: str) -> Callable:
        """
        Decorator for registering timecode generator classes
        
        Args:
            version: Timecode version
            
        Returns:
            Decorator function
        """
        def decorator(generator_cls: Type[TimecodeGenerator]) -> Type[TimecodeGenerator]:
            cls._generators[version] = generator_cls
            return generator_cls
        return decorator
    
    @classmethod
    def create(cls, version: str, project: ProjectData) -> TimecodeGenerator:
        """
        Create timecode generator instance
        
        Args:
            version: Timecode version
            project: Project data
            
        Returns:
            Timecode generator instance
        """
                
        if version not in cls._generators:
            raise ValueError(f"Unsupported timecode version: {version}")
        
        generator_cls = cls._generators[version]
        return generator_cls(project)