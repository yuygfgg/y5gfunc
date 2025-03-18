from .audio_config import AudioConfig, ProcessMode
from typing import Union, Optional
from pathlib import Path
import json
import subprocess
import platform
import tempfile
import shutil
from .utils import check_audio_stream_lossless
from ..utils import get_language_by_trackid
from ...utils import resolve_path

def encode_audio(
    input_file: Union[str, Path],
    output_file: Union[str, Path], 
    audio_track: int = 0,
    bitrate: Optional[str] = None,
    overwrite: bool = True,
    copy: bool = False,
    delay: float = 0.0,  # ms
) -> Path:
    
    input_path = resolve_path(input_file)
    output_path = resolve_path(output_file)

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

    input_stream = audio_info['streams'][0]
    
    input_is_lossless = check_audio_stream_lossless(input_stream)
    
    output_is_lossless = output_ext.lower() in ['.flac', '.wav', '.alac']
    
    if output_ext in {".aac", ".mp3"} and bitrate is None and not copy:
        bitrate = "320k"

    if delay > 0 and copy and not input_is_lossless and not output_is_lossless:
        temp_dir = Path(tempfile.mkdtemp())
        extracted_audio = temp_dir / f"extracted{output_ext}"
        silence_file = temp_dir / f"silence{output_ext}"
        concat_list = temp_dir / "concat.txt"
        
        extract_cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-map', f'0:a:{audio_track}',
            '-c:a', 'copy',
            str(extracted_audio)
        ]
        
        extract_process = subprocess.run(extract_cmd, capture_output=True, text=True)
        if extract_process.returncode != 0:
            shutil.rmtree(temp_dir)
            raise RuntimeError(f"encode_audio: Failed to extract audio: {extract_process.stderr}\nCommand: {extract_cmd}")
        
        extract_probe_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(extracted_audio)
        ]
        
        extract_probe_result = subprocess.run(extract_probe_cmd, capture_output=True, text=True)
        if extract_probe_result.returncode != 0:
            shutil.rmtree(temp_dir)
            raise RuntimeError(f"encode_audio: FFprobe failed on extracted audio: {extract_probe_result.stderr}")
        
        extracted_info = json.loads(extract_probe_result.stdout)
        if not extracted_info.get('streams'):
            shutil.rmtree(temp_dir)
            raise RuntimeError("encode_audio: No streams found in extracted audio")
        
        stream = extracted_info['streams'][0]
        codec = stream['codec_name']
        sample_rate = stream['sample_rate']
        sample_fmt = stream.get('sample_fmt')
        
        if 'channel_layout' in stream:
            channel_layout = stream['channel_layout'] 
        else:
            # Fallback based on channel count
            channels = stream['channels']
            if channels == 1:
                channel_layout = 'mono'
            elif channels == 2:
                channel_layout = 'stereo'
            elif channels == 6:
                channel_layout = '5.1'
            elif channels == 8:
                channel_layout = '7.1'
            else:
                channel_layout = f"{channels}c"
        
        if 'bit_rate' in stream:
            audio_bitrate = stream['bit_rate']
        else:
            audio_bitrate = '320k' if output_ext in {".aac", ".mp3"} else '192k'
        
        delay_sec = delay / 1000
        silence_cmd = [
            'ffmpeg', 
            '-f', 'lavfi', 
            '-i', f'anullsrc=channel_layout={channel_layout}:sample_rate={sample_rate}',
            '-t', f'{delay_sec}',
            '-c:a', codec,
            '-b:a', audio_bitrate
        ]
        if sample_fmt:
            silence_cmd.extend([
                '-sample_fmt', sample_fmt
            ])
        
        silence_cmd.append(str(silence_file))
        
        silence_process = subprocess.run(silence_cmd, capture_output=True, text=True)
        if silence_process.returncode != 0:
            shutil.rmtree(temp_dir)
            raise RuntimeError(f"encode_audio: Failed to generate silence: {silence_process.stderr}\nCommand: {silence_cmd}")
        
        with open(concat_list, 'w') as f:
            f.write(f"file '{silence_file}'\n")
            f.write(f"file '{extracted_audio}'\n")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            str(output_path)
        ]
        
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        shutil.rmtree(temp_dir)
        
        if process.returncode != 0:
            raise RuntimeError(f"encode_audio: FFmpeg concat failed: {process.stderr}\nCommand: {ffmpeg_cmd}")
        
        return output_path
    
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

def extract_audio_tracks(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: AudioConfig = AudioConfig()
) -> list[dict[str, Union[str, Path, bool]]]:

    input_path = resolve_path(input_path)
    assert input_path.exists()
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_audio"
    
    output_dir = resolve_path(output_dir)
    
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
        language = stream['language']
        channels = int(stream.get('channels', 2))
        is_default = bool(stream.get('disposition', {}).get('default'))
        is_comment = bool(stream.get('disposition', {}).get('comment'))
        
        is_stream_lossless = check_audio_stream_lossless(stream)

        is_core_track = False
        if codec_name == 'dts':
            base_id = stream.get('id', '').rsplit('.', 1)[0]
            for other_stream in streams:
                if all([
                    other_stream['index'] != track_num,
                    other_stream.get('id', '').startswith(base_id),
                    other_stream.get('profile', '').lower() == 'dts-hd ma'
                    ]):
                    is_core_track = True
                    break
        elif codec_name == 'ac3':
            base_id = stream.get('id', '').rsplit('.', 1)[0]
            for other_stream in streams:
                if all([
                    other_stream['index'] != track_num,
                    other_stream.get('codec_name', '').lower() == 'truehd',
                    other_stream.get('id', '').startswith(base_id)
                    ]):
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
            is_lossless=is_stream_lossless,
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