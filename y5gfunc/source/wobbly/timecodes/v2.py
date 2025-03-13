"""
V2 version timecode generator.
"""

from .base import TimecodeGenerator, TimecodeGeneratorFactory
from ..utils import frame_number_after_decimation


@TimecodeGeneratorFactory.register("v2")
class TimecodesV2Generator(TimecodeGenerator):
    """V2 version timecode generator"""
    
    def generate(self) -> str:
        """
        Generate V2 format timecode
        
        Returns:
            V2 format timecode string
        """
        tc = "# timecode format v2\n"
        
        numerators = [30000, 24000, 18000, 12000, 6000]
        denominator = 1001
        
        # Calculate total output frames
        total_frames = 0
        for range_info in self.ranges:
            start = range_info.start
            end = range_info.end
            total_frames += (
                frame_number_after_decimation(end - 1, self.decimated_by_cycle) - 
                frame_number_after_decimation(start, self.decimated_by_cycle) + 1
            )
            
        current_frame = 0
        current_time_ms = 0.0
        
        for range_info in self.ranges:
            dropped = range_info.dropped
            fps = numerators[dropped] / denominator
            frame_duration_ms = 1000.0 / fps
            
            start_frame = frame_number_after_decimation(
                range_info.start, self.decimated_by_cycle
            )
            end_frame = frame_number_after_decimation(
                range_info.end - 1, self.decimated_by_cycle
            )
            
            for _ in range(start_frame, end_frame + 1):
                tc += f"{current_time_ms:.6f}\n"
                current_time_ms += frame_duration_ms
                current_frame += 1
                
        return tc