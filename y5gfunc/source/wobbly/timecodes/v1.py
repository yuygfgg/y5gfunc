"""
V1 version timecode generator.
"""

from .base import TimecodeGenerator, TimecodeGeneratorFactory
from ..utils import frame_number_after_decimation
from ..types import TimecodeVersion


@TimecodeGeneratorFactory.register(TimecodeVersion.V1)
class TimecodesV1Generator(TimecodeGenerator):
    """V1 version timecode generator"""
    
    def generate(self) -> str:
        """
        Generate V1 format timecode
        
        Returns:
            V1 format timecode string
        """
        DEFAULT_FPS = 24000 / 1001
        
        tc = "# timecode format v1\n"
        tc += f"Assume {DEFAULT_FPS:.12f}\n"
        
        numerators = [30000, 24000, 18000, 12000, 6000]
        denominator = 1001
        
        for range_info in self.ranges:
            dropped = range_info.dropped
            
            if numerators[dropped] != 24000:
                start_frame = frame_number_after_decimation(
                    range_info.start, self.decimated_by_cycle
                )
                end_frame = frame_number_after_decimation(
                    range_info.end - 1, self.decimated_by_cycle
                )
                
                fps = numerators[dropped] / denominator
                tc += f"{start_frame},{end_frame},{fps:.12f}\n"
                
        return tc