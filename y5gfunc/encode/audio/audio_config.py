from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

class ProcessMode(Enum):
    COPY = 'copy'
    COMPRESS = 'compress'
    LOSSY = 'lossy'
    DROP = 'drop'

@dataclass
class TrackConfig:
    mode: ProcessMode = ProcessMode.COPY
    format: str = 'flac'  # COMPRESS or LOSSY
    bitrate: Optional[str] = None  # LOSSY

def create_main_lossless_2ch() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.COMPRESS, format='flac')

def create_main_lossless_multi() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.COMPRESS, format='flac')

def create_main_lossy() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.COPY)

def create_main_special() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.COMPRESS, format='flac')

def create_comment_lossless_2ch() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.LOSSY, format='aac', bitrate='192k')

def create_comment_lossless_multi() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.LOSSY, format='aac', bitrate='320k')

def create_comment_lossy_low() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.COPY)

def create_comment_lossy_2ch() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.LOSSY, format='aac', bitrate='192k')

def create_comment_lossy_multi() -> TrackConfig:
    return TrackConfig(mode=ProcessMode.LOSSY, format='aac', bitrate='320k')

@dataclass
class AudioConfig:
    main_lossless_2ch: TrackConfig = field(default_factory=create_main_lossless_2ch)
    main_lossless_multi: TrackConfig = field(default_factory=create_main_lossless_multi)
    main_lossy: TrackConfig = field(default_factory=create_main_lossy)
    main_special: TrackConfig = field(default_factory=create_main_special)
    
    comment_lossless_2ch: TrackConfig = field(default_factory=create_comment_lossless_2ch)
    comment_lossless_multi: TrackConfig = field(default_factory=create_comment_lossless_multi)
    comment_lossy_low: TrackConfig = field(default_factory=create_comment_lossy_low)
    comment_lossy_2ch: TrackConfig = field(default_factory=create_comment_lossy_2ch)
    comment_lossy_multi: TrackConfig = field(default_factory=create_comment_lossy_multi)
    
    lossy_threshold: int = 512 # Kbps
    
    def get_track_config(
        self, 
        is_comment: bool,
        is_lossless: bool,
        channels: int,
        bitrate: Optional[int] = None,
        is_special: bool = False
    ) -> TrackConfig:

        if is_special:
            return self.main_special
            
        if is_comment:
            if is_lossless:
                return (self.comment_lossless_2ch if channels <= 2 
                        else self.comment_lossless_multi)
            else:  # lossy
                if bitrate and bitrate < self.lossy_threshold:
                    return self.comment_lossy_low
                return (self.comment_lossy_2ch if channels <= 2 
                        else self.comment_lossy_multi)
        else:  # main track
            if is_lossless:
                return (self.main_lossless_2ch if channels <= 2 
                        else self.main_lossless_multi)
            return self.main_lossy