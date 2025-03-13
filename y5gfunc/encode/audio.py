from dataclasses import dataclass, field
from typing import Union, Optional
from enum import Enum
from pathlib import Path
import json
import subprocess
from .utils import get_language_by_trackid


# TODO: fix positive delay for lossy tracks with copy codec
def encode_audio(
    input_file: Union[str, Path],
    output_file: Union[str, Path], 
    audio_track: int = 0,
    bitrate: Optional[str] = None,
    overwrite: bool = True,
    copy: bool = False,
    delay: float = 0.0,  # ms
) -> Path:
    import platform
    
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"encode_audio: Input file not found: {input_path}")

    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise RuntimeError(f"encode_audio: Output file already exists! {output_path}")
    
    if copy and bitrate:
        raise ValueError("encode_audio: Cannot apply bitrate using copy mode!")

    output_ext = output_path.suffix.lower()
    if output_ext == ".flac" and bitrate is not None:
        raise ValueError("encode_audio: Don't set bitrate for flac file!")

    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', f'a:{audio_track}',
        str(input_path)
    ]    

    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode != 0:
        raise RuntimeError(f"encode_audio: FFprobe failed: {probe_result.stderr}")

    audio_info = json.loads(probe_result.stdout)
    if not audio_info.get('streams'):
        raise RuntimeError(f"encode_audio: No audio track {audio_track} found in file")

    if output_ext in {".aac", ".mp3"} and bitrate is None and not copy:
        bitrate = "320k"

    ffmpeg_cmd = ['ffmpeg', '-i', str(input_path)]
    
    if delay != 0:
        delay_sec = delay / 1000
        if delay_sec > 0:
            ffmpeg_cmd.extend(['-af', f'adelay={int(delay)}'])
        else:
            ffmpeg_cmd.extend(['-ss', f'{-delay_sec}'])

    ffmpeg_cmd.extend(['-map', f'0:a:{audio_track}'])

    if copy: 
        ffmpeg_cmd.extend(['-c:a', 'copy'])
    else:
        if output_ext == ".flac":
            sample_fmt = audio_info['streams'][0]['sample_fmt']
            if "16" in sample_fmt:
                sample_fmt = "s16"
            else:
                sample_fmt = "s32"
            
            sample_rate = audio_info['streams'][0]['sample_rate']
            ffmpeg_cmd.extend([
                '-c:a', 'flac',
                '-sample_fmt', sample_fmt,
                '-ar', sample_rate,
                '-compression_level', '12'
            ])
        elif output_ext == ".aac":
            assert isinstance(bitrate, str)
            if platform.system() == 'Darwin':
                ffmpeg_cmd.extend([
                    '-c:a', 'aac_at',
                    '-global_quality:a', '14',
                    '-aac_at_mode', '2',
                    '-b:a', bitrate
                ])
            else:
                ffmpeg_cmd.extend([ # better qaac?
                    '-c:a', 'libfdk_aac',
                    '-vbr', '5',
                    '-cutoff', '20000',
                    '-b:a', bitrate
                ])
        elif output_ext == ".mp3":
            assert isinstance(bitrate, str)
            ffmpeg_cmd.extend([
                '-c:a', 'libmp3lame',
                '-q:a', '0',
                '-b:a', bitrate
            ])
        elif bitrate:
            ffmpeg_cmd.extend(['-b:a', bitrate])

    ffmpeg_cmd.append(str(output_path))
    
    process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"encode_audio: FFmpeg failed: {process.stderr}\n FFMPEG cmd: {ffmpeg_cmd}")

    return output_path

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
    
    def get_track_config(self, 
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

def extract_audio_tracks(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: AudioConfig = AudioConfig()
) -> list[dict[str, Union[str, Path, bool]]]:

    input_path = Path(input_path)
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_audio"
    else:
        output_dir = Path(output_dir)
    
    video_track_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v",
        str(input_path)
    ]
    
    video_result = subprocess.run(video_track_cmd, capture_output=True, text=True)
    if video_result.returncode != 0:
        raise RuntimeError(f"extract_audio_tracks: ffprobe failed when detecting video tracks: {video_result.stderr}")
    
    video_streams = json.loads(video_result.stdout).get("streams", [])
    video_track_count = len(video_streams)

    print(f"extract_audio_tracks: Detected {video_track_count} video tracks.")

    audio_track_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        str(input_path)
    ]
    
    audio_result = subprocess.run(audio_track_cmd, capture_output=True, text=True)
    if audio_result.returncode != 0:
        raise RuntimeError(f"extract_audio_tracks: ffprobe failed when detecting audio tracks: {audio_result.stderr}")
    
    streams = json.loads(audio_result.stdout).get("streams", [])
    
    if not streams:
        raise RuntimeError("extract_audio_tracks: No audio tracks found!")

    print(f"extract_audio_tracks: Found {len(streams)} audio tracks:")
    delays = {}
    try:
        tsmuxer_cmd = ["tsMuxeR", str(input_path)]
        tsmuxer_process = subprocess.Popen(tsmuxer_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tsmuxer_output = tsmuxer_process.communicate()[0].decode()
        
        current_track = None
        for line in tsmuxer_output.splitlines():
            if line.startswith("Track ID:"):
                current_track = line[9:].strip()
            elif line.startswith("Stream delay:") and current_track:
                delay_str = line[13:].strip()
                if delay_str.endswith('ms'):
                    delay_str = delay_str[:-2]
                try:
                    delays[int(current_track)] = float(delay_str)
                except ValueError:
                    pass
    except Exception as e:
        print(f"Warning: Failed to get delays from tsmuxer: {e}")
    
    print(delays)
    
    for stream in streams:
        default_str = " (Default)" if stream.get('disposition', {}).get('default') else ""
        comment_str = " (Commentary)" if stream.get('disposition', {}).get('comment') else ""
        if not stream.get('id'):
            stream['id'] = hex(int(stream.get('index')) + 1) # make up id for non-m2ts sources
        language = get_language_by_trackid(m2ts_path=input_path, ffprobe_id=stream.get('id'))
        if language == 'und':
            if stream.get('tags'):
                language = stream.get('tags', {}).get('language') # also try ffprobe
        stream['language'] = language
        print(
            f"Track {stream['index']}: {stream.get('codec_name', 'unknown')} "
            f"{stream.get('channels', '?')}ch "
            f"Language: {language} "
            f"Delay: {delays.get(int(stream.get('id'), 16), 0.0)}"
            f"{default_str}{comment_str}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_tracks = []

    for stream in streams:
        track_num = stream['index']
        codec_name = stream.get('codec_name', '').lower()
        codec_long_name = stream.get('codec_long_name', '').lower()
        profile = stream.get('profile', '').lower()
        language = stream['language']
        channels = int(stream.get('channels', 2))
        is_default = bool(stream.get('disposition', {}).get('default'))
        is_comment = bool(stream.get('disposition', {}).get('comment'))

        LOSSLESS_CODECS = {'truehd', 'flac', 'alac', 'mlp', 'pcm'}
        LOSSLESS_PROFILES = {'dts-hd ma'}

        is_lossless = (
            any(codec in codec_long_name for codec in LOSSLESS_CODECS) or
            any(codec in codec_name for codec in LOSSLESS_CODECS) or
            any(profile in LOSSLESS_PROFILES for profile in [profile])
        )

        is_core_track = False
        if codec_name == 'dts':
            base_id = stream.get('id', '').rsplit('.', 1)[0]
            for other_stream in streams:
                if (other_stream['index'] != track_num and
                    other_stream.get('id', '').startswith(base_id) and
                    other_stream.get('profile', '').lower() == 'dts-hd ma'):
                    is_core_track = True
                    break
        elif codec_name == 'ac3':
            base_id = stream.get('id', '').rsplit('.', 1)[0]
            for other_stream in streams:
                if (other_stream['index'] != track_num and
                    other_stream.get('codec_name', '').lower() == 'truehd' and
                    other_stream.get('id', '').startswith(base_id)):
                    is_core_track = True
                    break

        if is_core_track:
            print(f"extract_audio_tracks: Skipping core track {track_num} ({codec_name})")
            continue

        source_bitrate = None
        if 'bit_rate' in stream:
            try:
                source_bitrate = int(stream['bit_rate']) // 1000  # kbps
            except (ValueError, TypeError):
                pass

        track_config = config.get_track_config(
            is_comment=is_comment,
            is_lossless=is_lossless,
            channels=channels,
            bitrate=source_bitrate
        )

        if track_config.mode == ProcessMode.DROP:
            print(f"extract_audio_tracks: Dropping track {track_num}")
        if track_config.mode == ProcessMode.COPY:
            output_ext = codec_name
            should_copy = True
            encode_bitrate = None
        elif track_config.mode == ProcessMode.COMPRESS:
            output_ext = track_config.format
            should_copy = False
            encode_bitrate = None
        else:  # LOSSY
            output_ext = track_config.format
            should_copy = False
            encode_bitrate = track_config.bitrate

        prefix = "comment_" if is_comment else "track_"
        output_path = output_dir / f"{prefix}{track_num}_{language}.{output_ext}"

        print(f"\nextract_audio_tracks: Processing {'comment ' if is_comment else ''}audio track {track_num}")
        print(f"Codec: {codec_name}, Channels: {channels}, Language: {language}")
        if source_bitrate:
            print(f"Original bitrate: {source_bitrate}kbps")

        delay = delays.get(int(stream.get('id', '-0x1'), 16), 0.0)
        if delay != 0.0:
            print(f"extract_audio_tracks: Applying delay of {delay}ms for track {track_num}")

        try:
            output_path = encode_audio(
                input_file=input_path,
                output_file=output_path,
                audio_track=track_num - video_track_count,
                overwrite=True,
                copy=should_copy,
                delay=delay,
                bitrate=encode_bitrate
            )

            extracted_tracks.append({
                "path": output_path,
                "language": language,
                "default": is_default,
                "comment": is_comment
            })

            print(f"extract_audio_tracks: Successfully extracted to: {output_path}")

        except Exception as e:
            raise RuntimeError(f"extract_audio_tracks: Failed to extract track {track_num}: {e}")

    return extracted_tracks