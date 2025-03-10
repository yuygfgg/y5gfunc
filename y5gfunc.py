from dataclasses import dataclass, field
from enum import Enum
import functools
import subprocess
from typing import Literal, Union, Any, Callable, Optional, IO, Sequence
from collections import deque
from subprocess import Popen
from types import FrameType
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import vsutil
from pathlib import Path
import time
import sys
import json
if sys.version_info >= (3, 11):
    from typing import LiteralString
else:
    LiteralString = str

from vsrgtools import removegrain


_output_index = 0
used_indices = set()

def reset_output_index(index: int = 0) -> None:
    global _output_index
    _output_index = index
    global used_indices
    used_indices = set()

def output(*args, debug: bool = True) -> None:
    import inspect
    from vspreview import set_output
    
    def _get_variable_name(frame: FrameType, clip: vs.VideoNode) -> str:
        for var_name, var_val in frame.f_locals.items():
            if var_val is clip:
                return var_name
        return "Unknown Variable"

    def _add_text(clip: vs.VideoNode, text: str, debug: bool = debug) -> vs.VideoNode:
        return core.akarin.Text(clip, text) if debug else clip
    
    if debug and __name__ == '__vapoursynth__':
        raise ValueError("Don't set debug=True when encoding!")
    
    frame: FrameType = inspect.currentframe().f_back # type: ignore
    
    global _output_index
    global used_indices
    clips_to_process = []

    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, tuple) and len(item) == 2:
                    clip, index = item
                    if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                        raise TypeError("Tuple must be (VideoNode, int)")
                    if index < 0:
                        raise ValueError("Output index must be non-negative")
                    clips_to_process.append((clip, index))
                elif isinstance(item, vs.VideoNode):
                    clips_to_process.append((item, None))
                else:
                    raise TypeError(f"Invalid element in list/tuple: {type(item)}")
        elif isinstance(arg, vs.VideoNode):
            clips_to_process.append((arg, None))
        elif isinstance(arg, tuple) and len(arg) == 2:
            clip, index = arg
            if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                raise TypeError("Tuple must be (VideoNode, int)")
            if index < 0:
                raise ValueError("Output index must be non-negative")
            clips_to_process.append((clip, index))
        else:
            raise TypeError(f"Invalid argument type: {type(arg)}")

    for clip, index in clips_to_process:
        if index is not None:
            if index in used_indices:
                raise ValueError(f"Output index {index} is already in use")
            variable_name = _get_variable_name(frame, clip)
            clip = _add_text(clip, f"{variable_name}")
            set_output(clip, index, f'{index}: {variable_name}')
            used_indices.add(index)
            
    for clip, index in clips_to_process:
        if index is None:
            while _output_index in used_indices:
                _output_index += 1
            variable_name = _get_variable_name(frame, clip)
            clip = _add_text(clip, f"{variable_name}")            
            set_output(clip, _output_index, f'{_output_index}: {variable_name}')
            used_indices.add(_output_index)

# TODO: add a function to get encoder params

def encode_video(
    clip: Union[vs.VideoNode, list[Union[vs.VideoNode, tuple[vs.VideoNode, int]]]],
    encoder: Union[list[Popen], Popen, IO, None] = None,
    multi: bool = False
) -> None:
    """
    Encode one or multiple VapourSynth video nodes using external encoders or output directly to stdout.
    
    Args:
        clip: A VapourSynth video node or a list of video nodes/tuples to encode.
        encoder: External encoder process(es) created with subprocess.Popen, a file-like object,
                or None to output to stdout.
        multi: If True, handle multiple input clips and multiple encoders. If False, handle a single clip.
    
    Examples:
        # Output to an external encoder
        encoder = subprocess.Popen(['x264', '--demuxer', 'y4m', '-', '-o', 'output.mp4'], stdin=subprocess.PIPE)
        encode_video(clip, encoder)
        
        # Output directly to stdout (like vspipe)
        encode_video(clip, None)
        # or
        encode_video(clip, sys.stdout)
        
        # Output to a file
        with open('output.y4m', 'wb') as f:
            encode_video(clip, f)
        
        # Example with multiple encoders
        encoders = [
            subprocess.Popen(['x264', '--demuxer', 'y4m', '-', '-o', 'output1.mp4'], stdin=subprocess.PIPE),
            subprocess.Popen(['x264', '--demuxer', 'y4m', '-', '-o', 'output2.mp4'], stdin=subprocess.PIPE)
        ]
        encode_video([clip1, clip2], encoders, multi=True)
    """
    
    # copied from https://skyeysnow.com/forum.php?mod=viewthread&tid=38690
    def _MIMO(clips: Sequence[vs.VideoNode], files: Sequence[IO]) -> None:
        ''' Multiple-Input-Multiple-Output
        '''

        def _y4m_header(clip: vs.VideoNode) -> str:
            y4mformat = ""
            if clip.format.color_family == vs.GRAY:
                y4mformat = 'mono'
                if clip.format.bits_per_sample > 8:
                    y4mformat = y4mformat + str(clip.format.bits_per_sample)
            else: # YUV
                if clip.format.subsampling_w == 1 and clip.format.subsampling_h == 1:
                    y4mformat = '420'
                elif clip.format.subsampling_w == 1 and clip.format.subsampling_h == 0:
                    y4mformat = '422'
                elif clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0:
                    y4mformat = '444'
                elif clip.format.subsampling_w == 2 and clip.format.subsampling_h == 2:
                    y4mformat = '410'
                elif clip.format.subsampling_w == 2 and clip.format.subsampling_h == 0:
                    y4mformat = '411'
                elif clip.format.subsampling_w == 0 and clip.format.subsampling_h == 1:
                    y4mformat = '440'
                
                if clip.format.bits_per_sample > 8:
                    y4mformat = y4mformat + 'p' + str(clip.format.bits_per_sample)

            y4mformat = 'C' + y4mformat + ' '
            data = 'YUV4MPEG2 {y4mformat}W{width} H{height} F{fps_num}:{fps_den} Ip A0:0 XLENGTH={length}\n'.format(
                y4mformat=y4mformat,
                width=clip.width,
                height=clip.height,
                fps_num=clip.fps_num,
                fps_den=clip.fps_den,
                length=len(clip)
            )

            return data
        
        # Checks
        num_clips = len(clips)
        num_files = len(files)
        assert num_clips > 0 and num_clips == num_files
        for clip in clips:
            assert clip.format

        is_y4m = [clip.format.color_family in (vs.YUV, vs.GRAY) for clip in clips]

        buffer = [False] * num_files
        for n in range(num_files):
            fileobj = files[n]
            if (fileobj is sys.stdout or fileobj is sys.stderr) and hasattr(fileobj, "buffer"):
                buffer[n] = True

        # Interleave
        max_len = max(len(clip) for clip in clips)
        clips_aligned: list[vs.VideoNode] = []
        for clip in clips:
            if len(clip) < max_len:
                clip_aligned = clip + vs.core.std.BlankClip(clip, length=max_len - len(clip))
            else:
                clip_aligned = clip
            clips_aligned.append(vs.core.std.Interleave([clip_aligned] * num_clips))   
        clips_varfmt = vs.core.std.BlankClip(length=max_len * num_clips, varformat=True, varsize=True)
        def _interleave(n: int, f: vs.VideoFrame) -> vs.VideoNode:
            return clips_aligned[n % num_clips]
        interleaved = vs.core.std.FrameEval(clips_varfmt, _interleave, clips_aligned, clips)

        # Y4M header
        for n in range(num_clips):
            if is_y4m[n]:
                clip = clips[n]
                fileobj = files[n].buffer if buffer[n] else files[n] # type: ignore
                data = _y4m_header(clip)
                fileobj.write(data.encode("ascii"))

        # Output
        for idx, frame in enumerate(interleaved.frames(close=True)):
            n = idx % num_clips
            clip = clips[n]
            fileobj = files[n].buffer if buffer[n] else files[n] # type: ignore
            finished = idx // num_clips
            if finished < len(clip):
                if is_y4m[n]:
                    fileobj.write(b"FRAME\n")
                for planeno, plane in enumerate(frame): # type: ignore
                    if frame.get_stride(planeno) != plane.shape[1] * clip.format.bytes_per_sample: # type: ignore
                        fileobj.write(bytes(plane))
                    else:
                        fileobj.write(plane)
                if hasattr(fileobj, "flush"):
                    fileobj.flush()
    
    if not multi:
        output_clip = None
        
        if isinstance(clip, vs.VideoNode):
            output_clip = clip
        elif isinstance(clip, list):
            for item in clip:
                if isinstance(item, tuple):
                    clip, index = item
                    if isinstance(clip, vs.VideoNode) and isinstance(index, int):
                        if index == 0:
                            output_clip = clip
                    else:
                        raise TypeError("encode_video: Tuple must be (VideoNode, int)")
            if not output_clip:
                output_clip = clip[0] if isinstance(clip[0], vs.VideoNode) else None
            if not output_clip:
                raise ValueError("encode_video: Couldn't parse clip!")
        
        assert isinstance(output_clip, vs.VideoNode)
        assert output_clip.format.color_family == vs.YUV, "encode_video: All clips must be YUV color family"
        assert output_clip.fps != 0, "encode_video: all clips must be CFR"
        
        # Allow direct output to stdout when encoder is None
        if encoder is None:
            _MIMO([output_clip], [sys.stdout])
        elif hasattr(encoder, 'write') and callable(encoder.write):  # type: ignore # type: ignore # File-like object
            _MIMO([output_clip], [encoder]) # type: ignore
        else:  # Subprocess.Popen object
            _MIMO([output_clip], [encoder.stdin])  # type: ignore
            encoder.communicate()  # type: ignore
            encoder.wait()  # type: ignore
    else:
        assert isinstance(encoder, list), "encode_video: encoder must be a list when multi=True"
        assert isinstance(clip, list), "encode_video: clip must be a list when multi=True"
        assert len(encoder) == len(clip), "encode_video: encoder and clip must have the same length"
        assert all(isinstance(item, vs.VideoNode) for item in clip), "encode_video: all items in clip must be VideoNodes"
        assert all(clip.format.color_family == vs.YUV for clip in clip), "encode_video: all clips must be YUV color family" # type: ignore
        assert all(clip.fps != 0 for clip in clip), "encode_video: all clips must be CFR" # type: ignore
        
        output_clips = []
        stdins = []
        for i, clip in enumerate(clip): # type: ignore
            output_clips.append(clip)
            if hasattr(encoder[i], 'write') and callable(encoder[i].write):  # type: ignore # File-like object
                stdins.append(encoder[i])
            else:  # Subprocess.Popen object
                stdins.append(encoder[i].stdin)
        
        _MIMO(output_clips, stdins)  # type: ignore
        
        # Only call communicate and wait for Popen objects
        for i, enc in enumerate(encoder):
            if not hasattr(enc, 'write') or not callable(enc.write):  # type: ignore # Not a file-like object
                enc.communicate()
                enc.wait()

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

def get_language_by_trackid(m2ts_path: Path, ffprobe_id) -> str:
    if str(ffprobe_id).startswith('0x'):
        track_id = int(ffprobe_id, 16)
    else:
        track_id = int(ffprobe_id) + 1
    
    try:
        tsmuxer_output = subprocess.check_output(['tsMuxeR', str(m2ts_path)], text=True)
    except subprocess.CalledProcessError:
        return "und"
    
    current_track = None
    track_langs = {}
    
    for line in tsmuxer_output.splitlines():
        line = line.strip()
        
        if line.startswith('Track ID:'):
            current_track = int(line.split(':')[1].strip())
        elif line.startswith('Stream lang:') and current_track is not None:
            lang = line.split(':')[1].strip()
            track_langs[current_track] = lang
            current_track = None
            
    return track_langs.get(track_id, 'und')

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


def get_bd_chapter(
    m2ts_or_mpls_path: Union[str, Path],
    chapter_save_path: Union[str, Path],
    target_clip: Optional[str] = None,
    all: bool = False  # True: return all mpls marks; False: return chapter
) -> Path:
    import struct
    
    m2ts_or_mpls_path = Path(m2ts_or_mpls_path)
    chapter_save_path = Path(chapter_save_path)

    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        whole_seconds = int(seconds_remainder)
        milliseconds = int((seconds_remainder - whole_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    def _process_mpls(mpls_path: Path, target_clip: Optional[str] = None) -> Optional[list[float]]:
        try:
            with mpls_path.open('rb') as f:
                if f.read(4) != b'MPLS':
                    raise ValueError(f"get_bd_chapter: Invalid MPLS format in file: {mpls_path}")
                
                f.seek(0)
                header: dict[str, Any] = {}
                header["TypeIndicator"] = f.read(4)
                header["VersionNumber"] = f.read(4)
                header["PlayListStartAddress"], = struct.unpack(">I", f.read(4))
                header["PlayListMarkStartAddress"], = struct.unpack(">I", f.read(4))
                header["ExtensionDataStartAddress"], = struct.unpack(">I", f.read(4))

                f.seek(header["PlayListStartAddress"])
                playlist_length, = struct.unpack(">I", f.read(4))
                f.read(2)  # reserved
                num_items, = struct.unpack(">H", f.read(2))
                num_subpaths, = struct.unpack(">H", f.read(2))

                play_items = []
                target_item_index = None
                for i in range(num_items):
                    item_length, = struct.unpack(">H", f.read(2))
                    item_start = f.tell()
                    
                    clip_name = f.read(5).decode('utf-8', errors='ignore')
                    codec_id = f.read(4).decode('utf-8', errors='ignore')  # noqa: F841
                    
                    f.read(3)  # reserved
                    stc_id = f.read(1)  # noqa: F841
                    in_time, = struct.unpack(">I", f.read(4))
                    out_time, = struct.unpack(">I", f.read(4))
                    
                    if target_clip and clip_name == target_clip:
                        target_item_index = i
                    
                    play_items.append({
                        'clip_name': clip_name,
                        'in_time': in_time,
                        'out_time': out_time
                    })
                    
                    f.seek(item_start + item_length)

                if target_clip and target_item_index is None:
                    return None

                f.seek(header["PlayListMarkStartAddress"])
                marks_length, = struct.unpack(">I", f.read(4))
                num_marks, = struct.unpack(">H", f.read(2))

                chapters_by_item = {}
                for _ in range(num_marks):
                    f.read(1)  # reserved
                    mark_type, = struct.unpack(">B", f.read(1))
                    ref_play_item_id, = struct.unpack(">H", f.read(2))
                    mark_timestamp, = struct.unpack(">I", f.read(4))
                    entry_es_pid, = struct.unpack(">H", f.read(2))
                    duration, = struct.unpack(">I", f.read(4))

                    if mark_type == 1:
                        if ref_play_item_id not in chapters_by_item:
                            chapters_by_item[ref_play_item_id] = []
                        chapters_by_item[ref_play_item_id].append(mark_timestamp)

                result = []
                if target_clip:
                    if target_item_index in chapters_by_item:
                        marks = chapters_by_item[target_item_index]
                        offset = min(marks)
                        if play_items[target_item_index]['in_time'] < offset: # type: ignore
                            offset = play_items[target_item_index]['in_time'] # type: ignore
                        
                        for timestamp in marks:
                            relative_time = (timestamp - offset) / 45000.0
                            if relative_time >= 0:
                                result.append(relative_time)
                else:
                    for item_id, marks in chapters_by_item.items():
                        offset = min(marks)
                        if play_items[item_id]['in_time'] < offset:
                            offset = play_items[item_id]['in_time']
                        
                        for timestamp in marks:
                            relative_time = (timestamp - offset) / 45000.0
                            if relative_time >= 0:
                                result.append(relative_time)

                return sorted(result)

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"get_bd_chapter: Error processing MPLS file: {str(e)}")

    if not m2ts_or_mpls_path.exists():
        raise FileNotFoundError(f"get_bd_chapter: Path does not exist: {m2ts_or_mpls_path}")

    is_mpls = m2ts_or_mpls_path.suffix.lower() == '.mpls'
    
    if is_mpls:
        if not target_clip and not all:
            raise ValueError("get_bd_chapter: target_clip must be provided with MPLS input if all is False!")
        chapters = _process_mpls(m2ts_or_mpls_path, target_clip) if not all else _process_mpls(m2ts_or_mpls_path)
    else:
        bdmv_root = next((p.parent for p in m2ts_or_mpls_path.parents if p.name.upper() == "BDMV"), None)
        if not bdmv_root:
            raise FileNotFoundError("get_bd_chapter: Could not find BDMV directory in path hierarchy")

        target_clip = m2ts_or_mpls_path.stem
        mpls_dir = bdmv_root / "BDMV" / "PLAYLIST"

        if not mpls_dir.exists():
            raise FileNotFoundError(f"PLAYLIST directory not found: {mpls_dir}")

        chapters = None
        for mpls_file in mpls_dir.glob("*.mpls"):
            try:
                chapters = _process_mpls(mpls_file, target_clip=target_clip)
                if chapters:
                    if all:
                        chapters = _process_mpls(mpls_file)
                    break
            except (ValueError, RuntimeError):
                continue

    if not chapters:
        raise ValueError("get_bd_chapter: No chapters found in the Blu-ray disc")

    try:
        with chapter_save_path.open('w', encoding='utf-8') as f:
            for i, time in enumerate(chapters, 1):
                chapter_num = f"{i:02d}"
                timestamp = _format_timestamp(time)
                f.write(f"CHAPTER{chapter_num}={timestamp}\n")
                f.write(f"CHAPTER{chapter_num}NAME=Chapter {i}\n")
    except IOError as e:
        raise IOError(f"get_bd_chapter: Failed to write chapter file: {str(e)}")

    return chapter_save_path

def get_mkv_chapter(mkv_path: Union[str, Path], output_path: Union[str, Path]) -> Path:
    
    mkv_path = Path(mkv_path)
    output_path = Path(output_path)

    result = subprocess.run(
        ["mkvextract", "chapters", str(mkv_path), "-s"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"get_mkv_chapter: Error extracting chapters from '{mkv_path}': {result.stderr.strip()}\n")
        

    chapter_data = result.stdout
    print(chapter_data)

    output_path.write_text(chapter_data, encoding="utf-8")
    return output_path

def subset_fonts(
    ass_path: Union[list[Union[str, Path]], str, Path], 
    fonts_path: Union[str, Path], 
    output_directory: Union[str, Path]
) -> Path:
    if isinstance(ass_path, (str, Path)):
        ass_path = [ass_path]
    
    ass_paths = [Path(path) for path in ass_path]
    fonts_path = Path(fonts_path)
    output_directory = Path(output_directory)

    subtitle_command = ["assfonts"]
    for path in ass_paths:
        subtitle_command += ["-i", str(path)]
    
    subtitle_command += ["-r", "-c", "-f", str(fonts_path), "-o", str(output_directory)]

    process = subprocess.run(subtitle_command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"subset_fonts: assfonts failed: {process.stderr}")

    return output_directory

def extract_pgs_subtitles(
    m2ts_path: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None
) -> list[dict[str, Union[str, Path, bool]]]:
    
    import tempfile
    import shutil
    
    m2ts_path = Path(m2ts_path)
    
    if output_dir is None:
        output_dir = m2ts_path.parent / f"{m2ts_path.stem}_subs"
    else:
        output_dir = Path(output_dir)
    
    print(f"extract_pgs_subtitles: Analyzing {m2ts_path}...")
    
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(m2ts_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"extract_pgs_subtitles: ffprobe failed: {result.stderr}")
    
    probe_data = json.loads(result.stdout)
    
    pgs_streams = []
    for stream in probe_data['streams']:
        if not stream.get('id'):
            stream['id'] = hex(int(stream.get('index')) + 1)
        if stream['codec_name'] == 'hdmv_pgs_subtitle':
            stream_id = stream['id']
            language = get_language_by_trackid(m2ts_path, stream_id)
            
            pgs_streams.append({
                'track_id': int(stream_id, 16),
                'language': language or 'und',
                'default': bool(stream['disposition']['default']),
                'type': 'PGS'
            })
    
    if not pgs_streams:
        raise RuntimeError("extract_pgs_subtitles: No PGS subtitles found!")
    
    print(f"Found {len(pgs_streams)} PGS subtitle streams:")
    for stream in pgs_streams:
        default_str = " (Default)" if stream['default'] else ""
        print(f"Track {stream['track_id']}: Language {stream['language']}{default_str}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        print(f"extract_pgs_subtitles: Using temporary directory: {temp_dir}")
        
        meta_content = ["MUXOPT --no-pcr-on-video-pid --new-audio-pes --demux\n"]
        for stream in pgs_streams:
            meta_line = f"S_HDMV/PGS, \"{m2ts_path}\", track={stream['track_id']}\n"
            meta_content.append(meta_line)
            print(f"extract_pgs_subtitles: Adding to meta: {meta_line.strip()}")
            
        meta_file = temp_dir / "meta.txt"
        meta_file.write_text("".join(meta_content))
        print(f"extract_pgs_subtitles: Created meta file at: {meta_file}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"extract_pgs_subtitles: Output directory: {output_dir}")
        
        print("\nextract_pgs_subtitles: Running tsMuxeR...")
        cmd = ["tsmuxer", str(meta_file), str(temp_dir)]
        print(f"extract_pgs_subtitles: Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline() # type: ignore
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"extract_pgs_subtitles: tsMuxeR: {output.strip()}")
        
        stdout, stderr = process.communicate()
        if stdout:
            print(f"extract_pgs_subtitles: tsMuxeR additional output: {stdout}")
        if stderr:
            print(f"extract_pgs_subtitles: tsMuxeR error output: {stderr}")
            
        if process.returncode != 0:
            raise RuntimeError(f"extract_pgs_subtitles: tsMuxeR failed with return code {process.returncode}")
        
        print("\nextract_pgs_subtitles: Extracting subtitles...")
        
        subtitles = []
        for stream in pgs_streams:
            track_num = stream['track_id']
            try:
                sup_file = next(temp_dir.glob(f"*track_{track_num}.sup"))
                
                final_path = output_dir / f"track_{track_num}_{stream['language']}.sup"
                shutil.move(str(sup_file), str(final_path))
                
                default_str = " (Default)" if stream['default'] else ""
                print(f"extract_pgs_subtitles: Extracted subtitle track {track_num} to {final_path}{default_str}")
                
                subtitles.append({
                    "path": final_path,
                    "language": stream['language'],
                    "default": stream['default']
                })
            except StopIteration:
                raise RuntimeError(f"extract_pgs_subtitles: Could not find extracted subtitle for track {track_num}")
    
    print("\nextract_pgs_subtitles: Extraction completed!")
    return subtitles

def mux_mkv(
    output_path: Union[str, Path],
    videos: Optional[Union[list[dict[str, Union[str, Path, bool]]], dict[str, Union[str, Path, bool]]]] = None,
    audios: Optional[Union[list[dict[str, Union[str, Path, bool]]], dict[str, Union[str, Path, bool]]]] = None,
    subtitles: Optional[Union[list[dict[str, Union[str, Path, bool]]], dict[str, Union[str, Path, bool]]]] = None,
    fonts_dir: Optional[Union[str, Path]] = None,
    chapters: Optional[Union[str, Path]] = None
) -> Path:
    '''
    {"path": str | Path, "language": str, "track_name": str, "default": bool, "comment": bool, "timecode": str | Path}
    '''
    output_path = Path(output_path)
    if fonts_dir:
        fonts_dir = Path(fonts_dir)
    if chapters:
        chapters = Path(chapters)

    assert any(x is not None for x in (fonts_dir, videos, audios, subtitles, chapters)), "mux_mkv: At least one input must be provided."

    def _normalize_inputs(inputs):
        if isinstance(inputs, dict):
            return [inputs]
        return inputs or []

    videos = _normalize_inputs(videos)
    audios = _normalize_inputs(audios)
    subtitles = _normalize_inputs(subtitles)

    for track_list in (videos, audios, subtitles):
        for track in track_list:
            track["path"] = Path(track["path"]) # type: ignore

    all_files = [track["path"] for track in videos + audios + subtitles] + ([chapters] if chapters else [])
    for file in all_files:
        if not file.exists(): # type: ignore
            raise FileNotFoundError(f"mux_mkv: Required file not found: {file}")

    mkvmerge_cmd = ["mkvmerge", "-o", str(output_path)]

    def _process_tracks(tracks) -> None:
        first_default_set = False
        for i, track in enumerate(tracks):
            if "language" in track:
                mkvmerge_cmd.extend(["--language", f"0:{track['language']}"])
            if "track_name" in track:
                mkvmerge_cmd.extend(["--track-name", f"0:{track['track_name']}"])
            if "comment" in track:
                mkvmerge_cmd.extend(["--commentary-flag", "0:yes"])
            if "timecode" in track:
                mkvmerge_cmd.extend(["--timestamps", f"0:{str(track['timecode'])}"])
            
            if track.get("default") is True:
                mkvmerge_cmd.extend(["--default-track", "0:yes"])
                first_default_set = True
            elif track.get("default") is False:
                mkvmerge_cmd.extend(["--default-track", "0:no"])
            elif not first_default_set and i == 0:
                mkvmerge_cmd.extend(["--default-track", "0:yes"])
                first_default_set = True
            else:
                mkvmerge_cmd.extend(["--default-track", "0:no"])

            mkvmerge_cmd.append(str(track["path"]))

    _process_tracks(videos)
    _process_tracks(audios)
    _process_tracks(subtitles)

    if chapters:
        mkvmerge_cmd.extend(["--chapters", str(chapters)])

    result = subprocess.run(mkvmerge_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mux_mkv: Error executing mkvmerge:\n{result.stdout}")

    if fonts_dir and fonts_dir.exists():
        for font_ext in ["ttf", "otf"]:
            for font_file in fonts_dir.glob(f"*.{font_ext}"):
                font_cmd = [
                    "mkvpropedit", str(output_path),
                    "--attachment-mime-type", f"font/{font_ext}",
                    "--add-attachment", str(font_file)
                ]
                font_result = subprocess.run(font_cmd, capture_output=True, text=True)
                if font_result.returncode != 0:
                    raise RuntimeError(f"mux_mkv: Error adding font {font_file}:\n{font_result.stderr}")
    
    return output_path

# modified from https://github.com/OrangeChannel/acsuite/blob/e40f50354a2fc26f2a29bf3a2fe76b96b2983624/acsuite/__init__.py#L252
def get_frame_timestamp(
    frame_num: int,
    clip: vs.VideoNode,
    precision: Literal['second', 'millisecond', 'microsecond' ,'nanosecond'] = 'millisecond',
    timecodes_v2_file: Optional[str] = None
)-> str:
    import fractions
    
    assert frame_num >= 0
    assert timecodes_v2_file is None or Path(timecodes_v2_file).exists()
    
    if frame_num == 0:
        s = 0.0
    elif clip.fps != fractions.Fraction(0, 1):
        t = round(float(10 ** 9 * frame_num * clip.fps ** -1))
        s = t / 10 ** 9
    else:
        if timecodes_v2_file is not None:
            timecodes = [float(x) / 1000 for x in open(timecodes_v2_file, "r").read().splitlines()[1:]]
            s = timecodes[frame_num]
        else:
            s = clip_to_timecodes(clip)[frame_num]

    m = s // 60
    s %= 60
    h = m // 60
    m %= 60

    if precision == 'second':
        return f"{h:02.0f}:{m:02.0f}:{round(s):02}"
    elif precision == 'millisecond':
        return f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
    elif precision == 'microsecond':
        return f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
    elif precision == 'nanosecond':
        return f"{h:02.0f}:{m:02.0f}:{s:012.9f}"

# TODO: use fps for CFR clips
# modified from https://github.com/OrangeChannel/acsuite/blob/e40f50354a2fc26f2a29bf3a2fe76b96b2983624/acsuite/__init__.py#L305
@functools.lru_cache
def clip_to_timecodes(clip: vs.VideoNode, path: Optional[str] = None) -> deque[float]:
    import collections
    import fractions

    timecodes = collections.deque([0.0], maxlen=clip.num_frames + 1)
    curr_time = fractions.Fraction()
    init_percentage = 0

    with open(path, "w", encoding="utf-8") if path else None as file: # type: ignore
        if file:
            file.write("# timecode format v2\n")

        for i, frame in enumerate(clip.frames()):
            num: int = frame.props["_DurationNum"] # type: ignore
            den: int = frame.props["_DurationDen"] # type: ignore
            curr_time += fractions.Fraction(num, den)
            timecode = float(curr_time)
            timecodes.append(timecode)

            if file:
                file.write(f"{timecode:.6f}\n")

            percentage_done = round(100 * len(timecodes) / clip.num_frames)
            if percentage_done % 10 == 0 and percentage_done != init_percentage:
                print(f"Finding timecodes for variable-framerate clip: {percentage_done}% done")
                init_percentage = percentage_done

    return timecodes

def create_minmax_expr(
    clip: vs.VideoNode,
    process_expr: str,
    threshold_expr: str,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    coordinates: list[int] = [1, 1, 1, 1, 1, 1, 1, 1],
    boundary: int = 1
) -> vs.VideoNode:
    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]
    def _build_neighbor_expr(coordinates: list[int]) -> str:
        NEIGHBOR_OFFSETS = [
            (-1, -1), (0, -1), (1, -1),  # 1, 2, 3
            (-1,  0),          (1,  0),  # 4  ,  5
            (-1,  1), (0,  1), (1,  1),  # 6, 7, 8
        ]
        return " ".join(
            f"x[{dx},{dy}]" 
            for flag, (dx, dy) in zip(coordinates, NEIGHBOR_OFFSETS) 
            if flag
        )
        
    if len(coordinates) != 8:
        raise ValueError("coordinates must contain exactly 8 elements.")

    neighbor_expr = _build_neighbor_expr(coordinates)
    expr = f"x[0,0] {' ' + neighbor_expr if neighbor_expr else ''} sort{sum(coordinates) + 1} {process_expr}"
    
    if threshold is not None:
        expr += threshold_expr.format(threshold)

    expressions = [
        expr if (i in planes) else "x" 
        for i in range(clip.format.num_planes)
    ]

    return core.akarin.Expr(clips=[clip], expr=expressions, boundary=boundary)

def minimum(
    clip: vs.VideoNode,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    coordinates: list[int] =   [1, 1, 1, 
                                1,    1, 
                                1, 1, 1],
    boundary: int = 1,
    force_std=False
) -> vs.VideoNode:

    if force_std:
        return core.std.Minimum(clip, planes, threshold, coordinates) # type: ignore
    else:
        return create_minmax_expr(clip, "min! drop{} min@".format(sum(coordinates)), " x[0,0] {} - swap max", planes, threshold, coordinates, boundary)

def maximum(
    clip: vs.VideoNode,
    planes: Optional[Union[list[int], int]] = None,
    threshold: Optional[float] = None,
    coordinates: list[int] =   [1, 1, 1, 
                                1,    1, 
                                1, 1, 1],
    boundary: int = 1,
    force_std=False
) -> vs.VideoNode:
    if force_std:
        return core.std.Maximum(clip, planes, threshold, coordinates) # type: ignore
    else:
        return create_minmax_expr(clip, "drop{}".format(sum(coordinates)), " x[0,0] {} + swap min", planes, threshold, coordinates, boundary)

# TODO: add exprs for other modes
def convolution(
    clip: vs.VideoNode,
    matrix: list[int],
    bias: float = 0.0,
    divisor: float = 0.0,
    planes: Optional[Union[list[int], int]] = None,
    saturate: bool = True,
    mode: str = "s",
    force_std: bool = False
) -> vs.VideoNode:

    if planes is None:
        planes = list(range(clip.format.num_planes))
    if isinstance(planes, int):
        planes = [planes]
    
    if mode != "s" or (len(matrix) != 9 and len(matrix) != 25) or force_std:
        return core.std.Convolution(clip, matrix, bias, divisor, planes, saturate, mode)
    
    if len(matrix) == 9:
        if abs(divisor) < 1e-9:
            actual_divisor = sum(matrix) if abs(sum(matrix)) > 1e-9 else 1.0
        else:
            actual_divisor = divisor
            
        coeffs = [f"{c:.6f}" for c in matrix]
        
        expr_parts = []
        
        if len(matrix) == 9:
            offsets =  [(-1, -1), (0, -1), (1, -1), 
                        (-1, 0), (0, 0), (1, 0), 
                        (-1, 1), (0, 1), (1, 1)]

        for i, (dx, dy) in enumerate(offsets):
            expr_parts.append(f"x[{dx},{dy}] {coeffs[i]} *")
            if i > 0:
                expr_parts.append("+")
        
        expr_parts.append(f" {actual_divisor:.6f} / {bias:.6f} + ")
        
        if saturate:
            if clip.format.sample_type == vs.INTEGER:
                peak = (1 << clip.format.bits_per_sample) - 1
                expr_parts.append(f"0 {peak} clip")
            else:
                expr_parts.append("0 1.0 clip")
        else:
            expr_parts.append("abs")
            if clip.format.sample_type == vs.INTEGER:
                peak = (1 << clip.format.bits_per_sample) - 1
                expr_parts.append(f"{peak} min")
            else:
                expr_parts.append("1.0 min")
        
        expr = " ".join(expr_parts)
        expressions = [expr if i in planes else "x" for i in range(clip.format.num_planes)]
        
        return core.akarin.Expr(clip, expressions, boundary=1)
    
    return core.std.Convolution(clip, matrix, bias, divisor, planes, saturate, mode)

# TODO: auto matrix handle
def load_source(
    file_path: Union[Path, str],
    track: int = 0,
    matrix_s: str = "709",
    matrix_in_s: str = "709",
    timecodes_v2_path: Optional[Union[Path, str]] = None
) -> vs.VideoNode:
    
    # refer to https://github.com/yuygfgg/vswobbly/blob/main/WobblyParser.py for more clear code and timecode v1 support
    def _wobbly_source(
        wob_project_path: Union[str, Path], 
        timecodes_v2_path: Optional[Union[str, Path]] = None
    ) -> vs.VideoNode:
        
        import os
        try:
            import orjson
        except Exception:
            pass
        
        class WobblyKeys:
            wobbly_version = "wobbly version"
            project_format_version = "project format version"
            input_file = "input file"
            input_frame_rate = "input frame rate"
            input_resolution = "input resolution"
            trim = "trim"
            source_filter = "source filter"
            user_interface = "user interface"
            vfm_parameters = "vfm parameters"
            matches = "matches"
            original_matches = "original matches"
            sections = "sections"
            presets = "presets"
            frozen_frames = "frozen frames"
            combed_frames = "combed frames"
            interlaced_fades = "interlaced fades"
            decimated_frames = "decimated frames"
            custom_lists = "custom lists"
            resize = "resize"
            crop = "crop"
            depth = "depth"
            
            class VFMParameters:
                order = "order"
            
            class Sections:
                start = "start"
                presets = "presets"
            
            class Presets:
                name = "name"
                contents = "contents"
            
            class CustomLists:
                name = "name"
                preset = "preset"
                position = "position"
                frames = "frames"
            
            class Resize:
                width = "width"
                height = "height"
                filter = "filter"
                enabled = "enabled"
            
            class Crop:
                early = "early"
                left = "left"
                top = "top"
                right = "right"
                bottom = "bottom"
                enabled = "enabled"
            
            class Depth:
                bits = "bits"
                float_samples = "float samples"
                dither = "dither"
                enabled = "enabled"
                
            class InterlacedFades:
                frame = "frame"
                field_difference = "field difference"
        
        def _apply_custom_lists(
            src: vs.VideoNode, 
            project: dict[str, Any], 
            presets: dict[str, Callable[[vs.VideoNode], vs.VideoNode]], 
            position: str, 
            WobblyKeys: Any, 
            frame_props: dict[int, dict], 
            frame_mapping: dict[int, int]
        ) -> tuple[vs.VideoNode, dict[int, dict], dict[int, int]]:
            
            custom_lists = [cl for cl in project.get(WobblyKeys.custom_lists, []) if cl.get(WobblyKeys.CustomLists.position) == position]
            
            if not custom_lists:
                return src, frame_props, frame_mapping
            
            all_ranges: list[tuple[int, int, str, str]] = []
            
            for cl_info in custom_lists:
                cl_name = cl_info.get(WobblyKeys.CustomLists.name)
                cl_preset = cl_info.get(WobblyKeys.CustomLists.preset)
                cl_frames = cl_info.get(WobblyKeys.CustomLists.frames, [])
                
                if not cl_preset or not cl_frames:
                    continue
                
                if cl_preset not in presets:
                    continue
                
                try:
                    ranges: list[tuple[int, int]] = []
                    for frame_range in cl_frames:
                        if isinstance(frame_range, list) and len(frame_range) == 2:
                            start, end = frame_range
                            
                            for frame in range(start, end+1):
                                for n in frame_props:
                                    if frame_mapping[n] == frame:
                                        frame_props[n].update({
                                            "WobblyCustomList": cl_name,
                                            "WobblyCustomListPreset": cl_preset,
                                            "WobblyCustomListPosition": position
                                        })
                            
                            ranges.append((start, end))
                            all_ranges.append((start, end, cl_name, cl_preset))
                    
                    ranges.sort()
                    
                    if ranges:
                        marked_clips = []
                        last_end = 0
                        
                        for range_start, range_end in ranges:
                            if not (0 <= range_start <= range_end < src.num_frames):
                                continue
                            
                            if range_start > last_end:
                                marked_clips.append(src[last_end:range_start])
                            
                            list_clip = presets[cl_preset](src[range_start:range_end+1])
                            marked_clips.append(list_clip)
                            
                            last_end = range_end + 1
                        
                        if last_end < src.num_frames:
                            marked_clips.append(src[last_end:])
                        
                        if marked_clips:
                            src = core.std.Splice(clips=marked_clips, mismatch=True)
                except Exception as e:
                    print(f"Warning: Error applying custom list '{cl_name}': {e}")
            
            return src, frame_props, frame_mapping


        def _get_decimation_info(project: dict[str, Any]) -> tuple[dict[int, set[int]], list[dict[str, int]]]:
            decimated_frames: list[int] = project.get('decimated frames', [])
            
            num_frames = 0
            if 'trim' in project:
                for trim in project['trim']:
                    if isinstance(trim, list) and len(trim) >= 2:
                        num_frames += trim[1] - trim[0] + 1
            
            decimated_by_cycle: dict[int, set[int]] = {}
            for frame in decimated_frames:
                cycle = frame // 5
                if cycle not in decimated_by_cycle:
                    decimated_by_cycle[cycle] = set()
                decimated_by_cycle[cycle].add(frame % 5)
            
            ranges: list[dict[str, int]] = []
            current_count = -1
            current_start = 0
            
            for cycle in range((num_frames + 4) // 5):
                count = len(decimated_by_cycle.get(cycle, set()))
                if count != current_count:
                    if current_count != -1:
                        ranges.append({
                            'start': current_start,
                            'end': cycle * 5,
                            'dropped': current_count
                        })
                    current_count = count
                    current_start = cycle * 5
            
            if current_count != -1:
                ranges.append({
                    'start': current_start,
                    'end': num_frames,
                    'dropped': current_count
                })
            
            return decimated_by_cycle, ranges


        def _frame_number_after_decimation(frame: int, decimated_by_cycle: dict[int, set[int]]) -> int:
            if frame < 0:
                return 0
            
            cycle = frame // 5
            offset = frame % 5
            
            decimated_before = 0
            for c in range(cycle):
                decimated_before += len(decimated_by_cycle.get(c, set()))
            
            for o in range(offset):
                if o in decimated_by_cycle.get(cycle, set()):
                    decimated_before += 1
            
            return frame - decimated_before

        def _generate_timecodes_v2(project: dict[str, Any]) -> str:
            decimated_by_cycle, ranges = _get_decimation_info(project)
            
            tc = "# timecode format v2\n"
            
            numerators = [30000, 24000, 18000, 12000, 6000]
            denominator = 1001
            
            total_frames = 0
            for range_info in ranges:
                start = range_info['start']
                end = range_info['end']
                total_frames += _frame_number_after_decimation(end - 1, decimated_by_cycle) - _frame_number_after_decimation(start, decimated_by_cycle) + 1
            
            current_frame = 0
            current_time_ms = 0.0
            
            for range_info in ranges:
                dropped = range_info['dropped']
                fps = numerators[dropped] / denominator
                frame_duration_ms = 1000.0 / fps
                
                start_frame = _frame_number_after_decimation(range_info['start'], decimated_by_cycle)
                end_frame = _frame_number_after_decimation(range_info['end'] - 1, decimated_by_cycle)
                
                for _ in range(start_frame, end_frame + 1):
                    tc += f"{current_time_ms:.6f}\n"
                    current_time_ms += frame_duration_ms
                    current_frame += 1
            
            return tc
        
        try:
            with open(wob_project_path, 'r', encoding='utf-8') as f:
                try:
                    project = orjson.loads(f.read())
                except Exception:
                    project = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read or parse Wobbly project file: {e}")
        
        input_file = project.get(WobblyKeys.input_file)
        source_filter = project.get(WobblyKeys.source_filter, "")
        
        if not input_file:
            raise ValueError("No input file specified in the project")
        
        if not os.path.isabs(input_file):
            wob_dir = os.path.dirname(os.path.abspath(str(wob_project_path)))
            input_file = os.path.join(wob_dir, input_file)
        
        if not os.path.exists(input_file):
            raise ValueError(f"Input file does not exist: {input_file}")

        frame_props: dict[int, dict] = {}
        
        try:
            if source_filter == "bs.VideoSource":
                src = core.bs.VideoSource(input_file, rff=True, showprogress=False)
            else:
                filter_parts = source_filter.split('.')
                plugin = getattr(core, filter_parts[0])
                src = getattr(plugin, filter_parts[1])(input_file)
        except Exception as e:
            raise ValueError(f"Failed to load video: {e}")
        
        for n in range(src.num_frames):
            frame_props[n] = {
                "WobblyProject": os.path.basename(str(wob_project_path)),
                "WobblyVersion": project.get(WobblyKeys.wobbly_version, ""),
                "WobblySourceFilter": source_filter,
                "WobblyCustomList": "",
                "WobblyCustomListPreset": "",
                "WobblyCustomListPosition": "",
                "WobblySectionStart": -1,
                "WobblySectionEnd": -1,
                "WobblySectionPresets": "",
                "WobblyMatch": ""
            }

        presets: dict[str, Callable[[vs.VideoNode], vs.VideoNode]] = {}
        for preset_info in project.get(WobblyKeys.presets, []):
            preset_name = preset_info.get(WobblyKeys.Presets.name)
            preset_contents = preset_info.get(WobblyKeys.Presets.contents)
            
            if not preset_name or preset_contents is None:
                continue
            
            try:
                exec_globals = {'vs': vs, 'core': core, 'c': core}
                exec(f"def preset_{preset_name}(clip):\n" + 
                    "\n".join("    " + line for line in preset_contents.split('\n')) + 
                    "\n    return clip", exec_globals)
                
                presets[preset_name] = exec_globals[f"preset_{preset_name}"]
            except Exception as e:
                print(f"Warning: Error creating preset '{preset_name}': {e}")
        
        frame_mapping: dict[int, int] = {}
        for i in range(src.num_frames):
            frame_mapping[i] = i
        
        try:
            crop_info = project.get(WobblyKeys.crop, {})
            if crop_info.get(WobblyKeys.Crop.enabled, False) and crop_info.get(WobblyKeys.Crop.early, False):
                crop_props = {
                    "WobblyCropEarly": True,
                    "WobblyCropLeft": crop_info.get(WobblyKeys.Crop.left, 0),
                    "WobblyCropTop": crop_info.get(WobblyKeys.Crop.top, 0),
                    "WobblyCropRight": crop_info.get(WobblyKeys.Crop.right, 0),
                    "WobblyCropBottom": crop_info.get(WobblyKeys.Crop.bottom, 0)
                }
                
                for n in frame_props:
                    frame_props[n].update(crop_props)
                    
                src = core.std.CropRel(
                    clip=src,
                    left=crop_info.get(WobblyKeys.Crop.left, 0),
                    top=crop_info.get(WobblyKeys.Crop.top, 0),
                    right=crop_info.get(WobblyKeys.Crop.right, 0),
                    bottom=crop_info.get(WobblyKeys.Crop.bottom, 0)
                )
            
            trim_list = project.get(WobblyKeys.trim, [])
            if trim_list:
                clips = []
                new_frame_props: dict[int, dict] = {}
                new_frame_idx = 0
                
                for trim in trim_list:
                    first, last = trim
                    if first <= last and first < src.num_frames and last < src.num_frames:
                        segment = src[first:last+1]
                        
                        for i in range(first, last+1):
                            if i in frame_props:
                                props = frame_props[i].copy()
                                props.update({
                                    "WobblyTrimStart": first,
                                    "WobblyTrimEnd": last
                                })
                                new_frame_props[new_frame_idx] = props
                                frame_mapping[new_frame_idx] = i
                                new_frame_idx += 1
                        
                        clips.append(segment)
                
                if clips:
                    src = core.std.Splice(clips=clips)
                    frame_props = new_frame_props
            
            src, frame_props, frame_mapping = _apply_custom_lists(
                src, project, presets, "post source", WobblyKeys, frame_props, frame_mapping
            )
            
            matches_list = project.get(WobblyKeys.matches)
            original_matches_list = project.get(WobblyKeys.original_matches)
            
            matches = ""
            if matches_list:
                matches = "".join(matches_list)
            elif original_matches_list:
                matches = "".join(original_matches_list)
            
            if matches:
                for n in frame_props:
                    orig_frame = frame_mapping[n]
                    if orig_frame < len(matches):
                        frame_props[n]["WobblyMatch"] = matches[orig_frame]
            
            if hasattr(core, 'fh'):
                if matches_list:
                    vfm_params = project.get(WobblyKeys.vfm_parameters, {})
                    order = vfm_params.get(WobblyKeys.VFMParameters.order, 1)
                    src = core.fh.FieldHint(clip=src, tff=order, matches=matches)
                elif original_matches_list:
                    vfm_params = project.get(WobblyKeys.vfm_parameters, {})
                    order = vfm_params.get(WobblyKeys.VFMParameters.order, 1)
                    src = core.fh.FieldHint(clip=src, tff=order, matches=matches)
            
            src, frame_props, frame_mapping = _apply_custom_lists(
                src, project, presets, "post field match", WobblyKeys, frame_props, frame_mapping
            )
            
            sections_list = project.get(WobblyKeys.sections, [])
            
            if sections_list:
                sorted_sections = sorted(sections_list, key=lambda s: s.get(WobblyKeys.Sections.start, 0))
                
                for i, section_info in enumerate(sorted_sections):
                    start = section_info.get(WobblyKeys.Sections.start, 0)
                    next_start = sorted_sections[i+1].get(WobblyKeys.Sections.start, src.num_frames) if i+1 < len(sorted_sections) else src.num_frames
                    
                    section_presets = section_info.get(WobblyKeys.Sections.presets, [])
                    presets_str = ",".join(section_presets)
                    
                    for n in frame_props:
                        orig_frame = frame_mapping[n]
                        if start <= orig_frame < next_start:
                            frame_props[n].update({
                                "WobblySectionStart": start,
                                "WobblySectionEnd": next_start-1,
                                "WobblySectionPresets": presets_str
                            })
                
                sections = []
                new_frame_props: dict[int, dict] = {}
                new_frame_idx = 0
                
                for i, section_info in enumerate(sorted_sections):
                    start = section_info.get(WobblyKeys.Sections.start, 0)
                    next_start = sorted_sections[i+1].get(WobblyKeys.Sections.start, src.num_frames) if i+1 < len(sorted_sections) else src.num_frames
                    
                    section_clip = src[start:next_start]
                    for preset_name in section_info.get(WobblyKeys.Sections.presets, []):
                        if preset_name in presets:
                            section_clip = presets[preset_name](section_clip)
                    
                    for j in range(section_clip.num_frames):
                        src_idx = start + j
                        if src_idx < len(frame_mapping):
                            orig_frame = frame_mapping[src_idx]
                            if src_idx in frame_props:
                                new_frame_props[new_frame_idx] = frame_props[src_idx].copy()
                                frame_mapping[new_frame_idx] = orig_frame
                                new_frame_idx += 1
                    
                    sections.append(section_clip)

                if sections:
                    src = core.std.Splice(clips=sections, mismatch=True)
                    frame_props = new_frame_props
            
            combed_frames = set(project.get(WobblyKeys.combed_frames, []))
            decimated_frames = set(project.get(WobblyKeys.decimated_frames, []))
            
            interlaced_fades = project.get(WobblyKeys.interlaced_fades, [])
            fade_dict: dict[int, float] = {}
            
            if interlaced_fades:
                for fade in interlaced_fades:
                    frame = fade.get(WobblyKeys.InterlacedFades.frame)
                    field_diff = fade.get(WobblyKeys.InterlacedFades.field_difference, 0)
                    if frame is not None:
                        fade_dict[frame] = field_diff
            
            orphan_fields: dict[int, dict[str, Any]] = {}
            
            if matches and sections_list:
                sorted_sections = sorted(sections_list, key=lambda s: s.get(WobblyKeys.Sections.start, 0))
                section_boundaries = [s.get(WobblyKeys.Sections.start, 0) for s in sorted_sections]
                section_boundaries.append(src.num_frames)
                
                for i in range(len(section_boundaries) - 1):
                    section_start = section_boundaries[i]
                    section_end = section_boundaries[i+1] - 1
                    
                    if section_start < len(matches) and matches[section_start] == 'n':
                        orphan_fields[section_start] = {'type': 'n', 'decimated': section_start in decimated_frames}
                    
                    if section_end < len(matches) and matches[section_end] == 'b':
                        orphan_fields[section_end] = {'type': 'b', 'decimated': section_end in decimated_frames}
            
            for n in frame_props:
                orig_frame = frame_mapping[n]
                props = frame_props[n]
                
                if orig_frame in combed_frames:
                    props["WobblyCombed"] = True
                
                if orig_frame in fade_dict:
                    props["WobblyInterlacedFade"] = True
                    props["WobblyFieldDifference"] = fade_dict[orig_frame]
                
                if orig_frame in orphan_fields:
                    info = orphan_fields[orig_frame]
                    props["WobblyOrphan"] = True
                    props["WobblyOrphanType"] = info['type']
                    props["WobblyOrphanDecimated"] = info['decimated']
                
                if orig_frame in decimated_frames:
                    props["WobblyDecimated"] = True
            
            frozen_frames_list = project.get(WobblyKeys.frozen_frames, [])
            if frozen_frames_list and hasattr(core.std, 'FreezeFrames'):
                first_frames = []
                last_frames = []
                replacement_frames = []
                
                for ff_info in frozen_frames_list:
                    if len(ff_info) == 3:
                        first, last, replacement = ff_info
                        if 0 <= first <= last < src.num_frames and 0 <= replacement < src.num_frames:
                            first_frames.append(first)
                            last_frames.append(last)
                            replacement_frames.append(replacement)
                            
                            for i in range(first, last+1):
                                if i in frame_props:
                                    frame_props[i]["WobblyFrozenFrame"] = True
                                    frame_props[i]["WobblyFrozenSource"] = replacement
                
                if first_frames:
                    src = core.std.FreezeFrames(
                        clip=src,
                        first=first_frames,
                        last=last_frames,
                        replacement=replacement_frames
                    )
            
            decimated_frames_list = project.get(WobblyKeys.decimated_frames, [])
            if decimated_frames_list:
                frames_to_delete = [f for f in decimated_frames_list if 0 <= f < src.num_frames]
                
                if frames_to_delete:
                    new_frame_props: dict[int, dict] = {}
                    new_idx = 0
                    
                    for n in range(src.num_frames):
                        orig_frame = frame_mapping.get(n, n)
                        
                        if orig_frame not in frames_to_delete:
                            if n in frame_props:
                                new_frame_props[new_idx] = frame_props[n].copy()
                                new_idx += 1
                    
                    src = core.std.DeleteFrames(clip=src, frames=frames_to_delete)
                    frame_props = new_frame_props
            
            src, frame_props, frame_mapping = _apply_custom_lists(
                src, project, presets, "post decimate", WobblyKeys, frame_props, frame_mapping
            )
            
            if crop_info.get(WobblyKeys.Crop.enabled, False) and not crop_info.get(WobblyKeys.Crop.early, False):
                crop_props = {
                    "WobblyCropEarly": False,
                    "WobblyCropLeft": crop_info.get(WobblyKeys.Crop.left, 0),
                    "WobblyCropTop": crop_info.get(WobblyKeys.Crop.top, 0),
                    "WobblyCropRight": crop_info.get(WobblyKeys.Crop.right, 0),
                    "WobblyCropBottom": crop_info.get(WobblyKeys.Crop.bottom, 0)
                }
                
                for n in frame_props:
                    frame_props[n].update(crop_props)
                    
                src = core.std.CropRel(
                    clip=src,
                    left=crop_info.get(WobblyKeys.Crop.left, 0),
                    top=crop_info.get(WobblyKeys.Crop.top, 0),
                    right=crop_info.get(WobblyKeys.Crop.right, 0),
                    bottom=crop_info.get(WobblyKeys.Crop.bottom, 0)
                )
            
            resize_info = project.get(WobblyKeys.resize, {})
            depth_info = project.get(WobblyKeys.depth, {})
            
            resize_enabled = resize_info.get(WobblyKeys.Resize.enabled, False)
            depth_enabled = depth_info.get(WobblyKeys.Depth.enabled, False)
            
            if resize_enabled or depth_enabled:
                resize_props: dict[str, Any] = {}
                
                resize_filter_name = resize_info.get(WobblyKeys.Resize.filter, "Bicubic")
                if resize_filter_name:
                    resize_filter_name = resize_filter_name[0].upper() + resize_filter_name[1:]
                else:
                    resize_filter_name = "Bicubic"
                
                if not hasattr(core.resize, resize_filter_name):
                    resize_filter_name = "Bicubic"
                
                resize_args: dict[str, Any] = {}
                if resize_enabled:
                    resize_width = resize_info.get(WobblyKeys.Resize.width, src.width)
                    resize_height = resize_info.get(WobblyKeys.Resize.height, src.height)
                    resize_args["width"] = resize_width
                    resize_args["height"] = resize_height
                    
                    resize_props.update({
                        "WobblyResizeEnabled": True,
                        "WobblyResizeWidth": resize_width,
                        "WobblyResizeHeight": resize_height,
                        "WobblyResizeFilter": resize_filter_name
                    })
                
                if depth_enabled:
                    bits = depth_info.get(WobblyKeys.Depth.bits, 8)
                    float_samples = depth_info.get(WobblyKeys.Depth.float_samples, False)
                    dither = depth_info.get(WobblyKeys.Depth.dither, "")
                    sample_type = vs.FLOAT if float_samples else vs.INTEGER
                    
                    format_id = core.query_video_format(
                        src.format.color_family,
                        sample_type,
                        bits,
                        src.format.subsampling_w,
                        src.format.subsampling_h
                    ).id
                    
                    resize_args["format"] = format_id
                    
                    resize_props.update({
                        "WobblyDepthEnabled": True,
                        "WobblyDepthBits": bits,
                        "WobblyDepthFloat": float_samples,
                        "WobblyDepthDither": dither
                    })
                
                for n in frame_props:
                    frame_props[n].update(resize_props)
                
                resize_filter = getattr(core.resize, resize_filter_name)
                src = resize_filter(clip=src, **resize_args)
            
            if timecodes_v2_path:
                timecodes = _generate_timecodes_v2(project)
                with open(str(timecodes_v2_path), 'w', encoding='utf-8') as f:
                    f.write(timecodes)
            
            def _apply_frame_props(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                if n in frame_props:
                    fout = f.copy()
                    
                    for key, value in frame_props[n].items():
                        if value is not None:
                            fout.props[key] = value
                    
                    return fout
                return f
            
            src = core.std.ModifyFrame(src, src, _apply_frame_props)
        except Exception as e:
            raise RuntimeError(f"Error processing Wobbly project: {e}")
        
        return src


    def _bestsource(
        file_path: Union[Path, str],
        track: int = 0,
        timecodes_v2_path: Optional[Union[Path, str]] = None,
        variableformat: int = -1,
        rff: bool = False
    ) -> vs.VideoNode:
    
        if timecodes_v2_path:
            return core.bs.VideoSource(str(file_path), track, variableformat, timecodes=str(timecodes_v2_path), rff=rff)
        else:
            return core.bs.VideoSource(str(file_path), track, variableformat, rff=rff)

    file_path = Path(file_path)
    
    assert file_path.exists()
    
    if file_path.suffix.lower() == ".wob":
        assert track == 0
        clip = _wobbly_source(file_path, timecodes_v2_path)
    else:
        # modified from https://guides.vcb-s.com/basic-guide-10/#%E6%A3%80%E6%B5%8B%E6%98%AF%E5%90%A6%E4%B8%BA%E5%85%A8%E7%A8%8B-soft-pulldownpure-film
        a = _bestsource(file_path, rff=False)
        b = _bestsource(file_path, rff=True)
        rff = False if abs(b.num_frames * 0.8 - a.num_frames) < 1 else True
        
        clip = _bestsource(file_path, track, timecodes_v2_path, rff=rff)
    
    return clip.resize.Spline36(matrix_s=matrix_s, matrix_in_s=matrix_in_s)
    
# TODO: add mvf.bm3d style presets
# modified from rksfunc.BM3DWrapper()
def Fast_BM3DWrapper(
    clip: vs.VideoNode,
    bm3d=core.bm3dcpu,
    chroma: bool = True,
    
    sigma_Y: Union[float, int] = 1.2,
    radius_Y: int = 1,
    delta_sigma_Y: Union[float, int] = 0.6,
    
    sigma_chroma: Union[float, int] = 2.4,
    radius_chroma: int = 0,
    delta_sigma_chroma: Union[float, int] = 1.2
) -> vs.VideoNode:

    '''
    Note: delta_sigma_xxx is added to sigma_xxx in step basic.
    '''

    assert clip.format.id == vs.YUV420P16
    
    # modified from yvsfunc
    def _rgb2opp(clip: vs.VideoNode) -> vs.VideoNode:
        coef = [1/3, 1/3, 1/3, 0, 1/2, -1/2, 0, 0, 1/4, 1/4, -1/2, 0]
        opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.YUV, coef=coef)
        opp = core.std.SetFrameProps(opp, _Matrix=vs.MATRIX_UNSPECIFIED, BM3D_OPP=1)
        return opp

    # modified from yvsfunc
    def _opp2rgb(clip: vs.VideoNode) -> vs.VideoNode:
        coef = [1, 1, 2/3, 0, 1, -1, 2/3, 0, 1, 0, -4/3, 0]
        rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.RGB, coef=coef)
        rgb = core.std.SetFrameProps(rgb, _Matrix=vs.MATRIX_RGB)
        rgb = core.std.RemoveFrameProps(rgb, 'BM3D_OPP')
        return rgb

    half_width = clip.width // 2  # half width
    half_height = clip.height // 2  # half height
    srcY_float, srcU_float, srcV_float = vsutil.split(vsutil.depth(clip, 32))

    vbasic_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=srcY_float,
        sigma=sigma_Y + delta_sigma_Y,
        radius=radius_Y
    )

    vfinal_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=vbasic_y,
        sigma=sigma_Y,
        radius=radius_Y
    )
    
    vyhalf = vfinal_y.resize2.Spline36(half_width, half_height, src_left=-0.5)
    srchalf_444 = vsutil.join([vyhalf, srcU_float, srcV_float])
    srchalf_opp = _rgb2opp(mvf.ToRGB(input=srchalf_444, depth=32, matrix="709", sample=1))

    vbasic_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=srchalf_opp,
        sigma=sigma_chroma + delta_sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0
    )

    vfinal_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=vbasic_half,
        sigma=sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0
    )

    vfinal_half = _opp2rgb(vfinal_half).resize2.Spline36(format=vs.YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = vsutil.split(vfinal_half)
    vfinal = vsutil.join([vfinal_y, vfinal_u, vfinal_v])
    return vsutil.depth(vfinal, 16)

# modified from rksfunc.SynDeband()
def SynDeband(
    clip: vs.VideoNode, 
    r1: int = 14, 
    y1: int = 72, 
    uv1: int = 48, 
    r2: int = 30,
    y2: int = 48, 
    uv2: int = 32, 
    mstr: int = 6000, 
    inflate: int = 2,
    include_mask: bool = False, 
    kill: Optional[vs.VideoNode] = None, 
    bmask: Optional[vs.VideoNode] = None,
    limit: bool = True,
    limit_thry: float = 0.12,
    limit_thrc: float = 0.1,
    limit_elast: float = 20,
) -> Union[vs.VideoNode, tuple[vs.VideoNode, vs.VideoNode]]:
    
    assert clip.format.id == vs.YUV420P16
    
    # copied from kagefunc.retinex_edgemask()
    def _retinex_edgemask(src: vs.VideoNode) -> vs.VideoNode:

        # modified from kagefunc.kirsch()
        def _kirsch(src: vs.VideoNode) -> vs.VideoNode:
            kirsch1 = convolution(src, matrix=[ 5,  5,  5, -3,  0, -3, -3, -3, -3], saturate=False)
            kirsch2 = convolution(src, matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], saturate=False)
            kirsch3 = convolution(src, matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], saturate=False)
            kirsch4 = convolution(src, matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], saturate=False)
            return core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4], 'x y max z max a max')
            
        luma = vsutil.get_y(src)
        max_value = 1 if src.format.sample_type == vs.FLOAT else (1 << vsutil.get_depth(src)) - 1
        ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
        tcanny = minimum(ret.tcanny.TCanny(mode=1, sigma=1), coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
        return core.akarin.Expr([_kirsch(luma), tcanny], f'x y + {max_value} min')
        
    if kill is None:
        kill = vsutil.iterate(clip, functools.partial(removegrain, mode=[20, 11]), 2)
    elif not kill:
        kill = clip
    
    assert isinstance(kill, vs.VideoNode)
    grain = core.std.MakeDiff(clip, kill)
    f3kdb_params = {
        'grainy': 0,
        'grainc': 0,
        'sample_mode': 2,
        'blur_first': True,
        'dither_algo': 2,
    }
    f3k1 = kill.neo_f3kdb.Deband(r1, y1, uv1, uv1, **f3kdb_params)
    f3k2 = f3k1.neo_f3kdb.Deband(r2, y2, uv2, uv2, **f3kdb_params)
    if limit:
        f3k2 = mvf.LimitFilter(f3k2, kill, thr=limit_thry, thrc=limit_thrc, elast=limit_elast)
    if bmask is None:
        bmask = _retinex_edgemask(kill).std.Binarize(mstr)
        bmask = vsutil.iterate(bmask, core.std.Inflate, inflate)
    deband = core.std.MaskedMerge(f3k2, kill, bmask)
    deband = core.std.MergeDiff(deband, grain)
    if include_mask:
        return deband, bmask
    else:
        return deband

# modified from LoliHouse: https://share.dmhy.org/topics/view/478666_LoliHouse_LoliHouse_1st_Anniversary_Announcement_and_Gift.html
def DBMask(clip: vs.VideoNode) -> vs.VideoNode:
    nr8: vs.VideoNode = vsutil.depth(clip, 8, dither_type='none')
    nrmasks = core.tcanny.TCanny(nr8, sigma=0.8, op=2, mode=1, planes=[0, 1, 2]).akarin.Expr(["x 7 < 0 65535 ?",""], vs.YUV420P16)
    nrmaskb = core.tcanny.TCanny(nr8, sigma=1.3, t_h=6.5, op=2, planes=0)
    nrmaskg = core.tcanny.TCanny(nr8, sigma=1.1, t_h=5.0, op=2, planes=0)
    nrmask = core.akarin.Expr([nrmaskg, nrmaskb, nrmasks, nr8],["a 20 < 65535 a 48 < x 256 * a 96 < y 256 * z ? ? ?",""], vs.YUV420P16)
    nrmask = minimum(vsutil.iterate(nrmask, functools.partial(maximum, planes=[0]), 2), planes=[0])
    nrmask = removegrain(nrmask, [20, 0])
    return nrmask

# modified from vardefunc.cambi_mask
def cambi_mask(
    clip: vs.VideoNode,
    scale: int = 1,
    merge_previous: bool = True,
    blur_func: Callable[[vs.VideoNode], vs.VideoNode] = lambda clip: core.std.BoxBlur(clip, planes=0, hradius=2, hpasses=3),
    **cambi_args: Any
) -> vs.VideoNode:
    
    assert 0 <= scale < 5
    assert callable(blur_func)
    
    if vsutil.get_depth(clip) > 10:
        clip = vsutil.depth(clip, 10, dither_type="none")

    scores = core.akarin.Cambi(clip, scores=True, **cambi_args)
    if merge_previous:
        cscores = [
            blur_func(scores.std.PropToClip(f'CAMBI_SCALE{i}').std.Deflate().std.Deflate())
            for i in range(scale + 1)
        ]
        expr_parts = [f"src{i} {scale + 1} /" for i in range(scale + 1)]
        expr = " ".join(expr_parts) + " " + " ".join(["+"] * (scale))
        deband_mask = core.akarin.Expr([core.resize2.Bilinear(c, scores.width, scores.height) for c in cscores], expr)
    else:
        deband_mask = blur_func(scores.std.PropToClip(f'CAMBI_SCALE{scale}').std.Deflate().std.Deflate())

    return deband_mask.std.CopyFrameProps(scores)

def Descale(
    src: vs.VideoNode,
    width: int,
    height: int,
    kernel: str,
    custom_kernel: Optional[Callable] = None,
    taps: int = 3,
    b: Union[int, float] = 0.0,
    c: Union[int, float] = 0.5,
    blur: Union[int, float] = 1.0,
    post_conv : Optional[list[Union[float, int]]] = None,
    src_left: Union[int, float] = 0.0,
    src_top: Union[int, float] = 0.0,
    src_width: Optional[Union[int, float]] = None,
    src_height: Optional[Union[int, float]] = None,
    border_handling: int = 0,
    ignore_mask: Optional[vs.VideoNode] = None,
    force: bool = False,
    force_h: bool = False,
    force_v: bool = False,
    opt: int = 0
) -> vs.VideoNode:
    
    def _get_resize_name(kernal_name: str) -> str:
        if kernal_name == 'Decustom':
            return 'ScaleCustom'
        if kernal_name.startswith('De'):
            return kernal_name[2:].capitalize()
        return kernal_name
    
    def _get_descaler_name(kernal_name: str) -> str:
        if kernal_name == 'ScaleCustom':
            return 'Decustom'
        if kernal_name.startswith('De'):
            return kernal_name
        return 'De' + kernal_name[0].lower() + kernal_name[1:]
    
    assert width > 0 and height > 0
    assert opt in [0, 1, 2]
    assert isinstance(src, vs.VideoNode) and src.format.id == vs.GRAYS
    
    kernel = kernel.capitalize()
    
    if src_width is None:
        src_width = width
    if src_height is None:
        src_height = height
    
    if width > src.width or height > src.height:
        kernel = _get_resize_name(kernel)
    else:
        kernel = _get_descaler_name(kernel)
    
    descaler = getattr(core.descale, kernel)
    assert callable(descaler)
    extra_params: dict[str, dict[str, Union[float, int, Callable]]] = {}
    if _get_descaler_name(kernel) == "Debicubic":
        extra_params = {
            'dparams': {'b': b, 'c': c},
        }
    elif _get_descaler_name(kernel) == "Delanczos":
        extra_params = {
            'dparams': {'taps': taps},
        }
    elif _get_descaler_name(kernel) == "Decustom":
        assert callable(custom_kernel)
        extra_params = {
            'dparams': {'custom_kernel': custom_kernel},
        }
    descaled = descaler(
        src=src,
        width=width,
        height=height,
        blur=blur,
        post_conv=post_conv,
        src_left=src_left,
        src_top=src_top,
        src_width=src_width,
        src_height=src_height,
        border_handling=border_handling,
        ignore_mask=ignore_mask,
        force=force,
        force_h=force_h,
        force_v=force_v,
        opt=opt,
        **extra_params.get('dparams', {})
    )
    
    assert isinstance(descaled, vs.VideoNode)
    
    return descaled

# TODO: use vs-jetpack Rescalers, handle asymmetrical descales
# inspired by https://skyeysnow.com/forum.php?mod=viewthread&tid=58390
def rescale(
    clip: vs.VideoNode,
    descale_kernel: Union[str, list[str]] = "Debicubic",
    src_height: Union[Union[float, int], list[Union[float, int]]] = 720,
    bw: Optional[Union[int, list[int]]] = None,
    bh: Optional[Union[int, list[int]]]  = None,
    show_upscaled: bool = False,
    show_fft: bool = False,
    detail_mask_threshold: float = 0.05,
    use_detail_mask: bool = True,
    show_detail_mask: bool = False,
    show_common_mask: bool = False,
    nnedi3_args: dict = {'field': 1, 'nsize': 4, 'nns': 4, 'qual': 2},
    taps: Union[int, list[int]] = 4,
    b: Union[Union[float, int], list[Union[float, int]]] = 0.33,
    c: Union[Union[float, int], list[Union[float, int]]] = 0.33,
    threshold_max: float = 0.007,
    threshold_min: float = -1,
    show_osd: bool = True,
    ex_thr: Union[float, int] = 0.015, 
    norm_order: int = 1, 
    crop_size: int = 5,
    exclude_common_mask: bool = True,
    scene_stable: bool = False,
    scene_descale_threshold_ratio: float = 0.5,
    scenecut_threshold: Union[float, int] = 0.1,
    opencl = True
) -> Union[
    vs.VideoNode,
    tuple[vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode]
]:
    
    '''
    To rescale from multiple native resolution, use this func for every possible src_height, then choose the largest MaxDelta one.
    
    e.g. 
    rescaled1, detail_mask1, osd1 = rescale(clip=srcorg, src_height=ranger(714.5, 715, 0.025)+[713, 714, 716, 717], bw=1920, bh=1080, descale_kernel="Debicubic", b=1/3, c=1/3, show_detail_mask=True)
    rescaled2, detail_mask2, osd2 = rescale(clip=srcorg, src_height=ranger(955, 957,0.1)+[953, 954, 958], bw=1920, bh=1080, descale_kernel="Debicubic", b=1/3, c=1/3, show_detail_mask=True)

    select_expr = "src0.MaxDelta src0.Descaled * src1.MaxDelta src1.Descaled * argmax2"

    osd = core.akarin.Select([osd1, osd2], [rescaled1, rescaled2], select_expr)
    src = core.akarin.Select([rescaled1, rescaled2], [rescaled1, rescaled2], select_expr)
    detail_mask = core.akarin.Select([detail_mask1, detail_mask2], [rescaled1, rescaled2], select_expr)
    '''
    
    from itertools import product
    from getfnative import descale_cropping_args
    from muvsfunc import SSIM_downsample
    
    KERNEL_MAP = {
        "Debicubic": 1,
        "Delanczos": 2,
        "Debilinear": 3,
        "Despline16": 5,
        "Despline36": 6,
        "Despline64": 7
    }
    
    descale_kernel = [descale_kernel] if isinstance(descale_kernel, str) else descale_kernel
    src_height = [src_height] if isinstance(src_height, (int, float)) else src_height
    if bw is None:
        bw = clip.width
    if bh is None:
        bh = clip.height
    bw = [bw] if isinstance(bw, int) else bw
    bh = [bh] if isinstance(bh, int) else bh
    taps = [taps] if isinstance(taps, int) else taps
    b = [b] if isinstance(b, (float, int)) else b
    c = [c] if isinstance(c, (float, int)) else c
    
    clip = vsutil.depth(clip, 32)
    
    def scene_descale(
        n: int,
        f: list[vs.VideoFrame],
        cache: list[int],
        prefetch: vs.VideoNode,
        length: int,
        scene_descale_threshold_ratio: float = scene_descale_threshold_ratio
    ) -> vs.VideoFrame:
    
        fout = f[0].copy()
        if n == 0 or n == prefetch.num_frames:
            fout.props['_SceneChangePrev'] = 1

        if cache[n] == -1: # not cached
            i = n
            scene_start = n
            while i >= 0:
                frame = prefetch.get_frame(i)
                if frame.props['_SceneChangePrev'] == 1: # scene srart
                    scene_start = i
                    break
                i -= 1
            i = scene_start
            scene_length = 0

            min_index_buffer = [0] * length
            num_descaled = 0

            while (i < prefetch.num_frames):
                frame = prefetch.get_frame(i)
                min_index_buffer[frame.props['MinIndex']] += 1 # type: ignore
                if frame.props['Descaled']:
                    num_descaled += 1
                scene_length += 1
                i += 1
                if frame.props['_SceneChangeNext'] == 1: # scene end
                    break

            scene_min_index = max(enumerate(min_index_buffer), key=lambda x: x[1])[0] if num_descaled >= scene_descale_threshold_ratio * scene_length else length
            
            i = scene_start
            for i in ranger(scene_start, scene_start+scene_length, step=1): # write scene prop
                cache[i] = scene_min_index

        fout.props['SceneMinIndex'] = cache[n]
        return fout

    def _get_resize_name(descale_name: str) -> str:
        if descale_name.startswith('De'):
            return descale_name[2:].capitalize()
        return descale_name
    
    # modified from kegefunc._generate_descale_mask()
    def _generate_detail_mask(source: vs.VideoNode, upscaled: vs.VideoNode, detail_mask_threshold: float = detail_mask_threshold) -> vs.VideoNode:
        mask = core.akarin.Expr([source, upscaled], 'src0 src1 - abs').std.Binarize(threshold=detail_mask_threshold)
        mask = vsutil.iterate(mask, maximum, 3)
        mask = vsutil.iterate(mask, core.std.Inflate, 3)
        return mask

    def _mergeuv(clipy: vs.VideoNode, clipuv: vs.VideoNode) -> vs.VideoNode:
        return core.std.ShufflePlanes([clipy, clipuv], [0, 1, 2], vs.YUV)
    
    def _generate_common_mask(detail_mask_clips: list[vs.VideoNode]) -> vs.VideoNode:
        load_expr = [f'src{i} * ' for i in range(len(detail_mask_clips))]
        merge_expr = ' '.join(load_expr)
        merge_expr = merge_expr[:4] + merge_expr[6:]
        return core.akarin.Expr(clips=detail_mask_clips, expr=merge_expr)
    
    def _select_per_frame(
        reference: vs.VideoNode,
        upscaled_clips: list[vs.VideoNode],
        candidate_clips: list[vs.VideoNode],
        params_list: list[dict],
        common_mask_clip: vs.VideoNode,
        threshold_max: float = threshold_max,
        threshold_min: float = threshold_min,
        ex_thr: float = ex_thr,
        norm_order: int = norm_order,
        crop_size: int = crop_size
    ) -> vs.VideoNode:
        
        def _crop(clip: vs.VideoNode, crop_size: int = crop_size) -> vs.VideoNode:
            return clip.std.CropRel(*([crop_size] * 4)) if crop_size > 0 else clip
        
        if len(upscaled_clips) != len(candidate_clips) or len(upscaled_clips) != len(params_list):
            raise ValueError("upscaled_clips, rescaled_clips, and params_list must have the same length.")

        calc_diff_expr = f"src0 src1 - abs dup {ex_thr} > swap {norm_order} pow 0 ? src2 - 0 1 clip"
        
        diffs = [core.akarin.Expr([_crop(reference), _crop(upscaled_clip), _crop(common_mask_clip)], calc_diff_expr).std.PlaneStats() for upscaled_clip in upscaled_clips]

        for diff in diffs:
            diff = diff.akarin.PropExpr(lambda: {'PlaneStatsAverage': f'x.PlaneStatsAverage {1 / norm_order} pow'})
        
        load_PlaneStatsAverage_exprs = [f'src{i}.PlaneStatsAverage' for i in range(len(diffs))]
        diff_expr = ' '.join(load_PlaneStatsAverage_exprs)
    
        min_index_expr = diff_expr + f' argmin{len(diffs)}'
        min_diff_expr = diff_expr + f' sort{len(diffs)} min_diff! drop{len(diffs)-1} min_diff@'
        
        max_index_expr = diff_expr + f' argmax{len(diffs)}'
        max_diff_expr = diff_expr + f' sort{len(diffs)} drop{len(diffs)-1}'
        
        max_delta_expr = max_diff_expr + " " + min_diff_expr + " / "

        def props() -> dict[str, str]:
            d = {
                'MinIndex': min_index_expr,
                'MinDiff': min_diff_expr,
                'MaxIndex': max_index_expr,
                'MaxDiff': max_diff_expr,
                "MaxDelta": max_delta_expr
            }
            for i in range(len(diffs)):
                d[f'Diff{i}'] = load_PlaneStatsAverage_exprs[i]
                params = params_list[i]
                d[f'KernelId{i}'] = KERNEL_MAP.get(params["Kernel"], 0) # type: ignore
                d[f'Bw{i}'] = params.get("BaseWidth", 0), # type: ignore
                d[f'Bh{i}'] = params.get("BaseHeight", 0)
                d[f'SrcHeight{i}'] = params['SrcHeight']
                d[f'B{i}'] = params.get('B', 0)
                d[f'C{i}'] = params.get('C', 0)
                d[f'Taps{i}'] = params.get('Taps', 0)
            return d

        prop_src = core.akarin.PropExpr(diffs, props)
        
        # Kernel is different because it's a string.
        for i in range(len(diffs)):
            prop_src = core.akarin.Text(prop_src, params_list[i]["Kernel"], prop=f"Kernel{i}")

        minDiff_clip = core.akarin.Select(clip_src=candidate_clips, prop_src=[prop_src], expr='x.MinIndex')

        final_clip = core.akarin.Select(clip_src=[reference, minDiff_clip], prop_src=[prop_src], expr=f'x.MinDiff {threshold_max} > x.MinDiff {threshold_min} <= or 0 1 ?')

        final_clip = final_clip.std.CopyFrameProps(prop_src)
        final_clip = core.akarin.PropExpr([final_clip], lambda: {'Descaled': f'x.MinDiff {threshold_max} <= x.MinDiff {threshold_min} > and'})

        return final_clip

    def _fft(clip: vs.VideoNode, grid: bool = True) -> vs.VideoNode:
        return core.fftspectrum.FFTSpectrum(clip=vsutil.depth(clip,8), grid=grid)
    
    if hasattr(core, "nnedi3cl") and opencl:
        nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)
        def nn2x(nn2x) -> vs.VideoNode:
            return nnedi3(nnedi3(nn2x, dh=True), dw=True)
    else:
        nnedi3 = functools.partial(core.nnedi3.nnedi3, **nnedi3_args)
        def nn2x(nn2x) -> vs.VideoNode:
            return nnedi3(nnedi3(nn2x, dh=True).std.Transpose(), dh=True).std.Transpose()
    
    upscaled_clips: list[vs.VideoNode] = []
    rescaled_clips: list[vs.VideoNode] = []
    detail_masks: list[vs.VideoNode] = []
    params_list: list[dict] = []
    
    src_luma = vsutil.get_y(clip)
    
    for kernel_name, sh, base_w, base_h, _taps, _b, _c in product(descale_kernel, src_height, bw, bh, taps, b, c):
        extra_params: dict[str, dict[str, Union[float, int]]] = {}
        if kernel_name == "Debicubic":
            extra_params = {
                'dparams': {'b': _b, 'c': _c},
                'rparams': {'filter_param_a': _b, 'filter_param_b': _c}
            }
        elif kernel_name == "Delanczos":
            extra_params = {
                'dparams': {'taps': _taps},
                'rparams': {'filter_param_a': _taps}
            }
        else:
            extra_params = {}

        dargs = descale_cropping_args(clip=clip, src_height=sh, base_height=base_h, base_width=base_w)

        descaled = getattr(core.descale, kernel_name)(
            src_luma,
            **dargs,
            **extra_params.get('dparams', {})
        )

        upscaled = getattr(core.resize2, _get_resize_name(kernel_name))(
            descaled,
            width=clip.width,
            height=clip.height,
            src_left=dargs['src_left'],
            src_top=dargs['src_top'],
            src_width=dargs['src_width'],
            src_height=dargs['src_height'],
            **extra_params.get('rparams', {})
        )

        n2x = nn2x(descaled)
        
        rescaled = SSIM_downsample(
            clip=n2x,
            w=clip.width,
            h=clip.height,
            sigmoid=False,
            src_left=dargs['src_left'] * 2 - 0.5,
            src_top=dargs['src_top'] * 2 - 0.5,
            src_width=dargs['src_width'] * 2,
            src_height=dargs['src_height'] * 2
        )

        upscaled_clips.append(upscaled)
        rescaled_clips.append(rescaled)
        
        if use_detail_mask or show_detail_mask or exclude_common_mask:
            detail_mask = _generate_detail_mask(src_luma, upscaled, detail_mask_threshold)
            detail_masks.append(detail_mask)

        params_list.append({
            'Kernel': kernel_name,
            'SrcHeight': sh,
            'BaseWidth': base_w,
            'BaseHeight': base_h,
            'Taps': _taps,
            'B': _b,
            'C': _c
        })
    
    common_mask_clip = _generate_common_mask(detail_mask_clips=detail_masks) if exclude_common_mask else core.std.BlankClip(clip=detail_masks[0], color=0)  
    if not scene_stable:
        rescaled = _select_per_frame(reference=src_luma, upscaled_clips=upscaled_clips, candidate_clips=rescaled_clips, params_list=params_list, common_mask_clip=common_mask_clip)
        detail_mask = core.akarin.Select(clip_src=detail_masks, prop_src=rescaled, expr="src0.MinIndex")
        detail_mask = core.akarin.Select(clip_src=[core.std.BlankClip(clip=detail_mask), detail_mask], prop_src=rescaled, expr="src0.Descaled")
        upscaled = core.akarin.Select(clip_src=upscaled_clips, prop_src=rescaled, expr="src0.MinIndex")
    else:
        # detail mask: matched one when descaling, otherwise blank clip
        # upscaled clip: matched one when descaling, otherwise frame level decision
        # rescaled clip: mostly-choosed index in a scene when descaling, otherwise src_luma
        # 'Descaled', 'SceneMinIndex': scene-level information
        # other props: frame-level information
        per_frame = _select_per_frame(reference=src_luma, upscaled_clips=upscaled_clips, candidate_clips=upscaled_clips, params_list=params_list, common_mask_clip=common_mask_clip)
        upscaled_per_frame = core.akarin.Select(clip_src=upscaled_clips, prop_src=per_frame, expr="src0.MinIndex")
        scene = core.misc.SCDetect(clip, scenecut_threshold)
        prefetch = core.std.BlankClip(clip)
        prefetch = core.akarin.PropExpr([scene, per_frame], lambda: {'_SceneChangeNext': 'x._SceneChangeNext', '_SceneChangePrev': 'x._SceneChangePrev', 'MinIndex': 'y.MinIndex', 'Descaled': 'y.Descaled'})
        cache = [-1] * clip.num_frames
        length = len(upscaled_clips)
        per_scene = core.std.ModifyFrame(per_frame, [per_frame, per_frame], functools.partial(scene_descale, prefetch=prefetch, cache=cache, length=length, scene_descale_threshold_ratio=scene_descale_threshold_ratio))
        rescaled = core.akarin.Select(rescaled_clips + [src_luma], [per_scene], 'src0.SceneMinIndex')
        rescaled = core.std.CopyFrameProps(per_scene, per_scene)
        rescaled = core.akarin.PropExpr([rescaled], lambda: {'Descaled': f'x.SceneMinIndex {len(upscaled_clips)} = not'})
        detail_mask = core.akarin.Select(clip_src=detail_masks + [core.std.BlankClip(clip=detail_masks[0])], prop_src=rescaled, expr="src0.SceneMinIndex")
        upscaled = core.akarin.Select(clip_src=upscaled_clips + [upscaled_per_frame], prop_src=rescaled, expr="src0.SceneMinIndex")

    if use_detail_mask:
        rescaled = core.std.MaskedMerge(rescaled, src_luma, detail_mask)

    final = _mergeuv(rescaled, clip) if clip.format.color_family == vs.YUV else rescaled
    
    if not scene_stable:
        format_string = (
            "\nMinIndex: {MinIndex}\n"
            "MinDiff: {MinDiff}\n"
            "Descaled: {Descaled}\n"
            f"Threshold_Max: {threshold_max}\n"
            f"Threshold_Min: {threshold_min}\n\n"
        )
    else:
        format_string = (
            "\nMinIndex: {MinIndex}\n"
            "MinDiff: {MinDiff}\n"
            "Descaled: {Descaled}\n"
            "SceneMinIndex: {SceneMinIndex}\n"
            f"Threshold_Max: {threshold_max}\n"
            f"Threshold_Min: {threshold_min}\n"
        )

    format_string += (
        "|    i   |          Diff          |   Kernel   | SrcHeight |    Bw     |     Bh    | B      | C      | Taps   |\n"
        "|--------|------------------------|------------|-----------|-----------|-----------|--------|--------|--------|\n"
    )

    for i in range(len(upscaled_clips)):
        format_string += (
            f"| {i:04}   | {{Diff{i}}}  | {{Kernel{i}}}  | {{SrcHeight{i}}}   | "
            f"{{Bw{i}}} | {{Bh{i}}} |"
            f"{{B{i}}}   | {{C{i}}}   | {{Taps{i}}}   |\n"
        )
    osd_clip = core.akarin.Text(final, format_string)

    if show_fft:
        src_fft = _fft(clip)
        rescaled_fft = _fft(final)
    
    # FUCK YOU PYLINT
    if show_upscaled: 
        if show_common_mask:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, common_mask_clip, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, upscaled, detail_mask, common_mask_clip, osd_clip
                elif show_fft:
                    return final, upscaled, common_mask_clip, src_fft, rescaled_fft, osd_clip
                else:
                    return final, upscaled, common_mask_clip, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, common_mask_clip, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, upscaled, detail_mask, common_mask_clip
                elif show_fft:
                    return final, upscaled, common_mask_clip, src_fft, rescaled_fft
                else:
                    return final, upscaled, common_mask_clip
        else:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, upscaled, detail_mask, osd_clip
                elif show_fft:
                    return final, upscaled, src_fft, rescaled_fft, osd_clip
                else:
                    return final, upscaled, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, upscaled, detail_mask
                elif show_fft:
                    return final, upscaled, src_fft, rescaled_fft
                else:
                    return final, upscaled
    else:
        if show_common_mask:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, detail_mask, common_mask_clip, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, detail_mask, common_mask_clip, osd_clip
                elif show_fft:
                    return final, common_mask_clip, src_fft, rescaled_fft, osd_clip
                else:
                    return final, common_mask_clip, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, detail_mask, common_mask_clip, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, detail_mask, common_mask_clip
                elif show_fft:
                    return final, common_mask_clip, src_fft, rescaled_fft
                else:
                    return final, common_mask_clip
        else:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, detail_mask, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, detail_mask, osd_clip
                elif show_fft:
                    return final, src_fft, rescaled_fft, osd_clip
                else:
                    return final, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, detail_mask, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, detail_mask
                elif show_fft:
                    return final, src_fft, rescaled_fft
                else:
                    return final



def ranger(start, end, step):
    if step == 0:
        raise ValueError("ranger: step must not be 0!")
    return [round(start + i * step, 10) for i in range(int((end - start) / step))]

def PickFrames(clip: vs.VideoNode, indices: list[int]) -> vs.VideoNode:
    try: 
        ret = core.akarin.PickFrames(clip, indices=indices) # type: ignore
    except AttributeError:
        try:
            ret = core.pickframes.PickFrames(clip, indices=indices)
        except AttributeError:
            # modified from https://github.com/AkarinVS/vapoursynth-plugin/issues/26#issuecomment-1951230729
            new = clip.std.BlankClip(length=len(indices))
            ret = new.std.FrameEval(lambda n: clip[indices[n]], None, clip) # type: ignore
    
    return ret

def screen_shot(clip: vs.VideoNode, frames: Union[list[int], int], path: str, file_name: str, overwrite: bool = True):

    if isinstance(frames, int):
        frames = [frames]
        
    clip = clip.resize2.Spline36(format=vs.RGB24)
    clip = PickFrames(clip=clip, indices=frames)
    
    output_path = Path(path).resolve()
    
    for i, _ in enumerate(clip.frames()):
        tmp = clip.std.Trim(first=i, last=i).fpng.Write(filename=(output_path / (file_name%frames[i])).with_suffix('.png'), overwrite=overwrite, compression=2) # type: ignore
        for f in tmp.frames():
            pass

# modified from https://github.com/DJATOM/VapourSynth-atomchtools/blob/34e16238291954206b3f7d5b704324dd6885b224/atomchtools.py#L370
def TIVTC_VFR(
    source: vs.VideoNode,
    clip2: Optional[vs.VideoNode] = None,
    tfmIn: Union[Path, str] = "matches.txt",
    tdecIn: Union[Path, str] = "metrics.txt",
    mkvOut: Union[Path, str] = "timecodes.txt",
    tfm_args: dict = dict(),
    tdecimate_args: dict = dict(),
    overwrite: bool = False
) -> vs.VideoNode:
    
    '''
    Convenient wrapper on tivtc to perform automatic vfr decimation with one function.
    '''
    
    def _resolve_folder_path(path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

    analyze = True

    assert isinstance(tfmIn, (str, Path))
    assert isinstance(tdecIn, (str, Path))
    assert isinstance(mkvOut, (str, Path))
    
    tfmIn = Path(tfmIn).resolve()
    tdecIn = Path(tdecIn).resolve()
    mkvOut = Path(mkvOut).resolve()

    if tfmIn.exists() and tdecIn.exists():
        analyze = False

    if clip2 and not overwrite:
        tfm_args.update(dict(clip2=clip2))

    if analyze:
        _resolve_folder_path(tfmIn)
        _resolve_folder_path(tdecIn)
        _resolve_folder_path(mkvOut)
        tfm_pass1_args = tfm_args.copy()
        tdecimate_pass1_args = tdecimate_args.copy()
        tfm_pass1_args.update(dict(output=str(tfmIn)))
        tdecimate_pass1_args.update(dict(output=str(tdecIn), mode=4))
        tmpnode = core.tivtc.TFM(source, **tfm_pass1_args)
        tmpnode = core.tivtc.TDecimate(tmpnode, **tdecimate_pass1_args)

        for i, _ in enumerate(tmpnode.frames()):
            print(f"Analyzing frame #{i}...", end='\r')

        del tmpnode
        time.sleep(0.5) # let it write logs

    tfm_args.update(dict(input=str(tfmIn)))
    tdecimate_args.update(dict(input=str(tdecIn), tfmIn=str(tfmIn), mkvOut=str(mkvOut), mode=5, hybrid=2, vfrDec=1))

    output = core.tivtc.TFM(source, **tfm_args)
    output = core.tivtc.TDecimate(output,  **tdecimate_args)

    return output

# inspired by mvf.postfix2infix
def postfix2infix(expr: str) -> LiteralString:
    import re
    # Preprocessing
    expr = expr.strip()
    expr = re.sub(r'\[\s*(\w+)\s*,\s*(\w+)\s*\]', r'[\1,\2]', expr) # [x, y] => [x,y]
    tokens = re.split(r'\s+', expr)
    
    stack = []
    output_lines = []

    # Regex patterns for numbers
    number_pattern = re.compile(
        r'^('
        r'0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?'
        r'|'
        r'0[0-7]*'
        r'|'
        r'[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)'
        r')$'
    )

    i = 0
    while i < len(tokens):
        def pop(n=1):
            try:
                if n == 1:
                    return stack.pop()
                r = stack[-n:]
                del stack[-n:]
                return r
            except IndexError:
                raise IndexError(f"postfix2infix: Stack Underflow at token at {i}th token {token}.")

        def push(item):
            stack.append(item)
        
        
        token = tokens[i]
        
        # Single letter
        if token.isalpha() and len(token) == 1:
            push(token)
            i += 1
            continue

        # Numbers
        if number_pattern.match(token):
            push(token)
            i += 1
            continue

        # Source clips (srcN)
        if re.match(r'^src\d+$', token):
            push(token)
            i += 1
            continue

        # Frame property
        if re.match(r'^[a-zA-Z]\w*\.[a-zA-Z]\w*$', token):
            push(token)
            i += 1
            continue

        # Dynamic pixel access
        if token.endswith('[]'):
            clip_identifier = token[:-2]
            absY = pop()
            absX = pop()
            push(f"{clip_identifier}.dyn({absX}, {absY})")
            i += 1
            continue

        # Static relative pixel access
        m = re.match(r'^([a-zA-Z]\w*)\[\-?(\d+)\,\-?(\d+)\](\:\w)?$', token)
        if m:
            clip_identifier = m.group(1)
            statX = int(m.group(2))
            statY = int(m.group(3))
            boundary_suffix = m.group(4)
            if boundary_suffix not in [None, ":c", ":m"]:
                raise ValueError(f"postfix2infix: unknown boundary_suffix {boundary_suffix} at {i}th token {token}")
            boundary_type = "_c" if not boundary_suffix or boundary_suffix == ":c" else "_m"
            push(f"{clip_identifier}.stat{boundary_type}({statX}, {statY})")
            i += 1
            continue

        # Variable operations
        var_store_match = re.match(r'^([a-zA-Z]\w*)\!$', token)
        var_load_match = re.match(r'^([a-zA-Z]\w*)\@$', token)
        if var_store_match:
            var_name = var_store_match.group(1)
            val = pop()
            output_lines.append(f"{var_name} = {val}")
            i += 1
            continue
        elif var_load_match:
            var_name = var_load_match.group(1)
            push(var_name)
            i += 1
            continue

        # Drop operations
        drop_match = re.match(r'^drop(\d*)$', token)
        if drop_match:
            num = int(drop_match.group(1)) if drop_match.group(1) else 1
            pop(num)
            i += 1
            continue

        # Sort operations
        sort_match = re.match(r'^sort(\d+)$', token)
        if sort_match:
            num = int(sort_match.group(1))
            items = pop(num)
            sorted_items_expr = f"sort({', '.join(items)})"
            for idx in range(len(items)):
                push(f"{sorted_items_expr}[{idx}]")
            i += 1
            continue

        # Duplicate operations
        dup_match = re.match(r'^dup(\d*)$', token)
        if dup_match:
            n = int(dup_match.group(1)) if dup_match.group(1) else 0
            if len(stack) <= n:
                raise ValueError(f"postfix2infix: {i}th token {token} needs at least {n} values, while only {len(stack)} in stack.")
            else:
                push(stack[-1 - n])
            i += 1
            continue

        # Swap operations
        swap_match = re.match(r'^swap(\d*)$', token)
        if swap_match:
            n = int(swap_match.group(1)) if swap_match.group(1) else 1
            if len(stack) <= n:
                raise ValueError(f"postfix2infix: {i}th token {token} needs at least {n} values, while only {len(stack)} in stack.")
            else:
                stack[-1], stack[-1 - n] = stack[-1 - n], stack[-1]
            i += 1
            continue

        # Special constants
        if token in ('N', 'X', 'Y', 'width', 'height'):
            constants = {
                'N': 'current_frame_number',
                'X': 'current_x',
                'Y': 'current_y',
                'width': 'current_width',
                'height': 'current_height'
            }
            push(constants[token])
            i += 1
            continue

        # Unary operators
        if token in ('sin', 'cos', 'round', 'trunc', 'floor', 'bitnot', 'abs', 'sqrt', 'not'):
            a = pop()
            if token == 'not':
                push(f"(!({a}))")
            else:
                push(f"{token}({a})")
            i += 1
            continue

        # Binary operators
        if token in ('%', '**', 'pow', 'bitand', 'bitor', 'bitxor'):
            b = pop()
            a = pop()
            if token == '%':
                push(f"({a} % {b})")
            elif token in ('**', 'pow'):
                push(f"pow({a}, {b})")
            elif token == 'bitand':
                push(f"({a} & {b})")
            elif token == 'bitor':
                push(f"({a} | {b})")
            elif token == 'bitxor':
                push(f"({a} ^ {b})")
            i += 1
            continue

        # Basic arithmetic, comparison and logical operators
        if token in ('+', '-', '*', '/', 'max', 'min', '>', '<', '>=', '<=', '=', 'and', 'or', 'xor'):
            b = pop()
            a = pop()
            if token in ('max', 'min'):
                push(f"{token}({a}, {b})")
            elif token == 'and':
                push(f"({a} && {b})")
            elif token == 'or':
                push(f"({a} || {b})")
            elif token == 'xor':
                # (a || b) && !(a && b)
                # (a && !b) || (!a && b)
                push(f"(({a} && !{b}) || (!{a} && {b}))")
            elif token == "=":
                push(f"{a} == {b}")
            else:
                push(f"({a} {token} {b})")
            i += 1
            continue

        # Ternary operator
        if token == '?':
            false_val = pop()
            true_val = pop()
            cond = pop()
            push(f"({cond} ? {true_val} : {false_val})")
            i += 1
            continue
        if token == 'clip' or token == 'clamp':
            max = pop()
            min = pop()
            value = pop()
            push(f"(clamp({value}, {min}, {max}))")
            i += 1
            continue

        # Unknown tokens
        output_lines.append(f"# [Unknown token]: {token}  (Push as-is)")
        push(token)
        i += 1

    # Handle remaining stack items
    if len(stack) == 1:
        output_lines.append(f"RESULT = {stack[0]}")
        ret = '\n'.join(output_lines)
        print(ret)
    else:
        for idx, item in enumerate(stack):
            output_lines.append(f"# stack[{idx}]: {item}")
        ret = '\n'.join(output_lines)
        raise ValueError(f"postfix2infix: Invalid expression: the stack contains not exactly one value after evaluation. \n {ret}")
    return ret

def encode_check(
    encoded: vs.VideoNode,
    source: Optional[vs.VideoNode] = None,
    mode: Literal["BOTH", "SSIM", "CAMBI"] = "BOTH",
    threshold_cambi: float = 5,
    threshold_ssim: float = 0.9,
    return_type: Literal["encoded", "error", "both"] = "encoded"
) -> Union[
    vs.VideoNode,
    tuple[vs.VideoNode, vs.VideoNode]
]:
    
    from muvsfunc import SSIM
    
    assert 0 <= threshold_cambi <= 24
    assert 0 <= threshold_ssim <= 1
    assert mode in ["BOTH", "SSIM", "CAMBI"]
    assert return_type in ['encoded', 'error', 'both']

    if mode == "BOTH":
        enable_ssim = enable_cambi = True
    elif mode == "SSIM":
        enable_ssim = True
        enable_cambi = False
    else:
        enable_ssim = False
        enable_cambi = True
    
    if enable_ssim:
        assert source
        assert encoded.format.id == source.format.id
        ssim = SSIM(encoded, source)
            
    if enable_cambi:
        cambi = cambi_mask(encoded)
    
    error_frames = []
    def _chk(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        def print_red_bold(text) -> None:
            print("\033[1;31m" + text + "\033[0m")
            
        fout = f[0].copy()
        
        ssim_err = cambi_err = False
        
        if enable_ssim: 
            fout.props['PlaneSSIM'] = ssim_val = f[2].props['PlaneSSIM']
            fout.props['ssim_err'] = ssim_err = (1 if threshold_ssim > f[2].props['PlaneSSIM'] else 0) # type: ignore
        
        if enable_cambi: 
            fout.props['CAMBI'] = cambi_val = f[1].props['CAMBI'] 
            fout.props['cambi_err'] = cambi_err = (1 if threshold_cambi < f[1].props['CAMBI'] else 0) # type: ignore
        
        if cambi_err and enable_cambi:
            print_red_bold (f"frame {n}: Banding detected! CAMBI: {cambi_val}"
                            f"    Note: banding threshold is {threshold_cambi}")
        if ssim_err and enable_ssim:
            print_red_bold (f"frame {n}: Distortion detected! SSIM: {ssim_val}"
                            f"    Note: distortion threshold is {threshold_ssim}")
        if not (cambi_err or ssim_err):
            print(f"Frame {n}: OK!")
        else:
            error_frames.append(n)

        return fout

    if enable_ssim and enable_cambi:
        output = core.std.ModifyFrame(encoded, [encoded, cambi, ssim], _chk)
    elif enable_cambi:
        output = core.std.ModifyFrame(encoded, [encoded, cambi, cambi], _chk)
    else:
        output = core.std.ModifyFrame(encoded, [encoded, ssim, ssim], _chk)
    
    if return_type == "encoded": 
        return output
    
    for _ in output.frames():
        pass
    
    err = PickFrames(encoded, error_frames)

    if return_type == "both":
        return output, err
    else:
        return err

# inspired by https://skyeysnow.com/forum.php?mod=redirect&goto=findpost&ptid=13824&pid=333218
def is_stripe(
    clip: vs.VideoNode,
    threshold: Union[float, int] = 2,
    freq_range: Union[int, float] = 0.25,
    scenecut_threshold: Union[float, int] = 0.1
) -> vs.VideoNode:

    def scene_fft(n: int, f: list[vs.VideoFrame], cache: list[float], prefetch: vs.VideoNode) -> vs.VideoFrame:
        fout = f[0].copy()
        if n == 0 or n == prefetch.num_frames:
            fout.props['_SceneChangePrev'] = 1
        
        if cache[n] == -1.0: # not cached
            i = n
            scene_start = n
            while i >= 0:
                frame = prefetch.get_frame(i)
                if frame.props['_SceneChangePrev'] == 1: # scene srart
                    scene_start = i
                    break
                i -= 1
            i = scene_start
            scene_length = 0
            hor_accum = 1e-9
            ver_accum = 0
            while (i < prefetch.num_frames):
                frame = prefetch.get_frame(i)
                hor_accum += frame.props['hor'] # type: ignore
                ver_accum += frame.props['ver'] # type: ignore
                scene_length += 1
                i += 1
                if frame.props['_SceneChangeNext'] == 1: # scene end
                    break
            
            ratio = ver_accum / hor_accum
            
            i = scene_start
            for i in ranger(scene_start, scene_start+scene_length, step=1): # write scene prop
                cache[i] = ratio
        
        fout.props['ratio'] = cache[n]
        return fout
    
    assert clip.format.bits_per_sample == 8
    assert 0 < freq_range < 0.5
    
    freq_drop_range = 1 - freq_range
    freq_drop_lr = int(clip.width * freq_drop_range)
    freq_drop_bt = int(clip.height * freq_drop_range)
    
    fft = core.fftspectrum.FFTSpectrum(clip)

    left = core.std.Crop(fft, right=freq_drop_lr)
    right = core.std.Crop(fft, left=freq_drop_lr)
    hor = core.std.StackHorizontal([left, right]).std.PlaneStats()

    top = core.std.Crop(fft, bottom=freq_drop_bt)
    bottom = core.std.Crop(fft, top=freq_drop_bt)
    ver = core.std.StackHorizontal([top, bottom]).std.PlaneStats()

    scene = core.misc.SCDetect(clip, threshold=scenecut_threshold)
    
    prefetch = core.std.BlankClip(clip)
    prefetch = core.akarin.PropExpr([hor, ver, scene], lambda: {'hor': 'x.PlaneStatsAverage', 'ver': 'y.PlaneStatsAverage', '_SceneChangeNext': 'z._SceneChangeNext', '_SceneChangePrev': 'z._SceneChangePrev'})

    cache = [-1.0] * scene.num_frames

    ret = core.std.ModifyFrame(scene, [scene, scene], functools.partial(scene_fft, prefetch=prefetch, cache=cache))
    ret = core.akarin.PropExpr([ret], lambda: {'_Stripe': f'x.ratio {threshold} >'}) # x.ratio > threshold: Stripe
    
    return ret

def get_oped_mask(
    clip: vs.VideoNode,
    ncop: vs.VideoNode,
    nced: vs.VideoNode,
    op_start: int,
    ed_start: int,
    threshold: int = 7
) -> tuple[vs.VideoNode, vs.VideoNode]:

    from fvsfunc import rfs
    
    assert clip.format == ncop.format == nced.format
    assert clip.format.color_family == vs.YUV
    
    op_end = op_start + ncop.num_frames
    ed_end = ed_start + nced.num_frames
    
    assert 0 <= op_start <= op_end < ed_start <= ed_end < clip.num_frames
    
    if op_start != 0:
        ncop = core.std.Trim(clip, first=0, last=op_start-1) + ncop + core.std.Trim(clip, first=op_end+1, last=clip.num_frames-1)
    else:
        ncop = ncop + core.std.Trim(clip, first=op_end, last=clip.num_frames-1)
    
    if ed_end != clip.num_frames - 1:
        nced = core.std.Trim(clip, first=0, last=ed_start-1) + nced + core.std.Trim(clip, first=ed_end+1, last=clip.num_frames-1)
    else:
        nced = core.std.Trim(clip, first=0, last=ed_start-1) + nced
    
    nc = rfs(clip, ncop, f"[{op_start} {op_end}]")
    nc = rfs(nc, nced, f"[{ed_start} {ed_end}]")
    
    thr = threshold / 255 * ((1 << clip.format.bits_per_sample) - 1) if clip.format.sample_type == vs.INTEGER else threshold / 255
    max = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1

    diff = core.akarin.Expr([nc, clip], f"x y - abs {thr} < 0 {max} ?")
    diff = vsutil.get_y(diff)
    
    diff = vsutil.iterate(diff, maximum, 5)
    diff = vsutil.iterate(diff, minimum, 6)
    
    return nc, diff
    

# copied from https://github.com/Artoriuz/glsl-chroma-from-luma-prediction/blob/main/CfL_Prediction.glsl
cfl_shader = R'''
//!PARAM chroma_offset_x
//!TYPE float
0.0

//!PARAM chroma_offset_y
//!TYPE float
0.0

//!HOOK CHROMA
//!BIND LUMA
//!BIND CHROMA
//!SAVE LUMA_LR
//!WIDTH CHROMA.w
//!HEIGHT LUMA.h
//!WHEN CHROMA.w LUMA.w <
//!DESC Chroma From Luma Prediction (Hermite 1st step, Downscaling Luma)

float comp_wd(vec2 v) {
    float x = min(length(v), 1.0);
    return smoothstep(0.0, 1.0, 1.0 - x);
}

vec4 hook() {
    vec2 luma_pos = LUMA_pos;
    luma_pos.x += chroma_offset_x / LUMA_size.x;
    float start  = ceil((luma_pos.x - (1.0 / CHROMA_size.x)) * LUMA_size.x - 0.5);
    float end = floor((luma_pos.x + (1.0 / CHROMA_size.x)) * LUMA_size.x - 0.5);

    float wt = 0.0;
    float luma_sum = 0.0;
    vec2 pos = luma_pos;

    for (float dx = start.x; dx <= end.x; dx++) {
        pos.x = LUMA_pt.x * (dx + 0.5);
        vec2 dist = (pos - luma_pos) * CHROMA_size;
        float wd = comp_wd(dist);
        float luma_pix = LUMA_tex(pos).x;
        luma_sum += wd * luma_pix;
        wt += wd;
    }

    vec4 output_pix = vec4(luma_sum /= wt, 0.0, 0.0, 1.0);
    return clamp(output_pix, 0.0, 1.0);
}

//!HOOK CHROMA
//!BIND LUMA_LR
//!BIND CHROMA
//!BIND LUMA
//!SAVE LUMA_LR
//!WIDTH CHROMA.w
//!HEIGHT CHROMA.h
//!WHEN CHROMA.w LUMA.w <
//!DESC Chroma From Luma Prediction (Hermite 2nd step, Downscaling Luma)

float comp_wd(vec2 v) {
    float x = min(length(v), 1.0);
    return smoothstep(0.0, 1.0, 1.0 - x);
}

vec4 hook() {
    vec2 luma_pos = LUMA_LR_pos;
    luma_pos.y += chroma_offset_y / LUMA_LR_size.y;
    float start  = ceil((luma_pos.y - (1.0 / CHROMA_size.y)) * LUMA_LR_size.y - 0.5);
    float end = floor((luma_pos.y + (1.0 / CHROMA_size.y)) * LUMA_LR_size.y - 0.5);

    float wt = 0.0;
    float luma_sum = 0.0;
    vec2 pos = luma_pos;

    for (float dy = start; dy <= end; dy++) {
        pos.y = LUMA_LR_pt.y * (dy + 0.5);
        vec2 dist = (pos - luma_pos) * CHROMA_size;
        float wd = comp_wd(dist);
        float luma_pix = LUMA_LR_tex(pos).x;
        luma_sum += wd * luma_pix;
        wt += wd;
    }

    vec4 output_pix = vec4(luma_sum /= wt, 0.0, 0.0, 1.0);
    return clamp(output_pix, 0.0, 1.0);
}

//!HOOK CHROMA
//!BIND HOOKED
//!BIND LUMA
//!BIND LUMA_LR
//!WHEN CHROMA.w LUMA.w <
//!WIDTH LUMA.w
//!HEIGHT LUMA.h
//!OFFSET ALIGN
//!DESC Chroma From Luma Prediction (Upscaling Chroma)

#define USE_12_TAP_REGRESSION 1
#define USE_8_TAP_REGRESSIONS 1
#define DEBUG 0

float comp_wd(vec2 v) {
    float d = min(length(v), 2.0);
    float d2 = d * d;
    float d3 = d2 * d;

    if (d < 1.0) {
        return 1.25 * d3 - 2.25 * d2 + 1.0;
    } else {
        return -0.75 * d3 + 3.75 * d2 - 6.0 * d + 3.0;
    }
}

vec4 hook() {
    float ar_strength = 0.8;
    vec2 mix_coeff = vec2(0.8);
    vec2 corr_exponent = vec2(4.0);

    vec4 output_pix = vec4(0.0, 0.0, 0.0, 1.0);
    float luma_zero = LUMA_texOff(0.0).x;

    vec2 pp = HOOKED_pos * HOOKED_size - vec2(0.5);
    vec2 fp = floor(pp);
    pp -= fp;

#ifdef HOOKED_gather
    vec2 quad_idx[4] = {{0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}, {2.0, 2.0}};

    vec4 luma_quads[4];
    vec4 chroma_quads[4][2];

    for (int i = 0; i < 4; i++) {
        luma_quads[i] = LUMA_LR_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 0);
        chroma_quads[i][0] = HOOKED_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 0);
        chroma_quads[i][1] = HOOKED_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 1);
    }

    vec2 chroma_pixels[16];
    chroma_pixels[0]  = vec2(chroma_quads[0][0].w, chroma_quads[0][1].w);
    chroma_pixels[1]  = vec2(chroma_quads[0][0].z, chroma_quads[0][1].z);
    chroma_pixels[2]  = vec2(chroma_quads[1][0].w, chroma_quads[1][1].w);
    chroma_pixels[3]  = vec2(chroma_quads[1][0].z, chroma_quads[1][1].z);
    chroma_pixels[4]  = vec2(chroma_quads[0][0].x, chroma_quads[0][1].x);
    chroma_pixels[5]  = vec2(chroma_quads[0][0].y, chroma_quads[0][1].y);
    chroma_pixels[6]  = vec2(chroma_quads[1][0].x, chroma_quads[1][1].x);
    chroma_pixels[7]  = vec2(chroma_quads[1][0].y, chroma_quads[1][1].y);
    chroma_pixels[8]  = vec2(chroma_quads[2][0].w, chroma_quads[2][1].w);
    chroma_pixels[9]  = vec2(chroma_quads[2][0].z, chroma_quads[2][1].z);
    chroma_pixels[10] = vec2(chroma_quads[3][0].w, chroma_quads[3][1].w);
    chroma_pixels[11] = vec2(chroma_quads[3][0].z, chroma_quads[3][1].z);
    chroma_pixels[12] = vec2(chroma_quads[2][0].x, chroma_quads[2][1].x);
    chroma_pixels[13] = vec2(chroma_quads[2][0].y, chroma_quads[2][1].y);
    chroma_pixels[14] = vec2(chroma_quads[3][0].x, chroma_quads[3][1].x);
    chroma_pixels[15] = vec2(chroma_quads[3][0].y, chroma_quads[3][1].y);

    float luma_pixels[16];
    luma_pixels[0]  = luma_quads[0].w;
    luma_pixels[1]  = luma_quads[0].z;
    luma_pixels[2]  = luma_quads[1].w;
    luma_pixels[3]  = luma_quads[1].z;
    luma_pixels[4]  = luma_quads[0].x;
    luma_pixels[5]  = luma_quads[0].y;
    luma_pixels[6]  = luma_quads[1].x;
    luma_pixels[7]  = luma_quads[1].y;
    luma_pixels[8]  = luma_quads[2].w;
    luma_pixels[9]  = luma_quads[2].z;
    luma_pixels[10] = luma_quads[3].w;
    luma_pixels[11] = luma_quads[3].z;
    luma_pixels[12] = luma_quads[2].x;
    luma_pixels[13] = luma_quads[2].y;
    luma_pixels[14] = luma_quads[3].x;
    luma_pixels[15] = luma_quads[3].y;
#else
    vec2 pix_idx[16] = {{-0.5,-0.5}, {0.5,-0.5}, {1.5,-0.5}, {2.5,-0.5},
                        {-0.5, 0.5}, {0.5, 0.5}, {1.5, 0.5}, {2.5, 0.5},
                        {-0.5, 1.5}, {0.5, 1.5}, {1.5, 1.5}, {2.5, 1.5},
                        {-0.5, 2.5}, {0.5, 2.5}, {1.5, 2.5}, {2.5, 2.5}};

    float luma_pixels[16];
    vec2 chroma_pixels[16];

    for (int i = 0; i < 16; i++) {
        luma_pixels[i] = LUMA_LR_tex(vec2((fp + pix_idx[i]) * HOOKED_pt)).x;
        chroma_pixels[i] = HOOKED_tex(vec2((fp + pix_idx[i]) * HOOKED_pt)).xy;
    }
#endif

#if (DEBUG == 1)
    vec2 chroma_spatial = vec2(0.5);
    mix_coeff = vec2(1.0);
#else
    float wd[16];
    float wt = 0.0;
    vec2 ct = vec2(0.0);

    vec2 chroma_min = min(min(min(chroma_pixels[5], chroma_pixels[6]), chroma_pixels[9]), chroma_pixels[10]);
    vec2 chroma_max = max(max(max(chroma_pixels[5], chroma_pixels[6]), chroma_pixels[9]), chroma_pixels[10]);

    const int dx[16] = {-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2};
    const int dy[16] = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

    for (int i = 0; i < 16; i++) {
        wd[i] = comp_wd(vec2(dx[i], dy[i]) - pp);
        wt += wd[i];
        ct += wd[i] * chroma_pixels[i];
    }

    vec2 chroma_spatial = ct / wt;
    chroma_spatial = clamp(mix(chroma_spatial, clamp(chroma_spatial, chroma_min, chroma_max), ar_strength), 0.0, 1.0);
#endif

#if (USE_12_TAP_REGRESSION == 1 || USE_8_TAP_REGRESSIONS == 1)
    const int i12[12] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14};
    const int i4y[4] = {1, 2, 13, 14};
    const int i4x[4] = {4, 7, 8, 11};
    const int i4[4] = {5, 6, 9, 10};

    float luma_sum_4 = 0.0;
    float luma_sum_4y = 0.0;
    float luma_sum_4x = 0.0;
    vec2 chroma_sum_4 = vec2(0.0);
    vec2 chroma_sum_4y = vec2(0.0);
    vec2 chroma_sum_4x = vec2(0.0);

    for (int i = 0; i < 4; i++) {
        luma_sum_4 += luma_pixels[i4[i]];
        luma_sum_4y += luma_pixels[i4y[i]];
        luma_sum_4x += luma_pixels[i4x[i]];
        chroma_sum_4 += chroma_pixels[i4[i]];
        chroma_sum_4y += chroma_pixels[i4y[i]];
        chroma_sum_4x += chroma_pixels[i4x[i]];
    }

    float luma_avg_12 = (luma_sum_4 + luma_sum_4y + luma_sum_4x) / 12.0;
    float luma_var_12 = 0.0;
    vec2 chroma_avg_12 = (chroma_sum_4 + chroma_sum_4y + chroma_sum_4x) / 12.0;
    vec2 chroma_var_12 = vec2(0.0);
    vec2 luma_chroma_cov_12 = vec2(0.0);

    for (int i = 0; i < 12; i++) {
        luma_var_12 += pow(luma_pixels[i12[i]] - luma_avg_12, 2.0);
        chroma_var_12 += pow(chroma_pixels[i12[i]] - chroma_avg_12, vec2(2.0));
        luma_chroma_cov_12 += (luma_pixels[i12[i]] - luma_avg_12) * (chroma_pixels[i12[i]] - chroma_avg_12);
    }

    vec2 corr = clamp(abs(luma_chroma_cov_12 / max(sqrt(luma_var_12 * chroma_var_12), 1e-6)), 0.0, 1.0);
    mix_coeff = pow(corr, corr_exponent) * mix_coeff;
#endif

#if (USE_12_TAP_REGRESSION == 1)
    vec2 alpha_12 = luma_chroma_cov_12 / max(luma_var_12, 1e-6);
    vec2 beta_12 = chroma_avg_12 - alpha_12 * luma_avg_12;
    vec2 chroma_pred_12 = clamp(alpha_12 * luma_zero + beta_12, 0.0, 1.0);
#endif

#if (USE_8_TAP_REGRESSIONS == 1)
    const int i8y[8] = {1, 2, 5, 6, 9, 10, 13, 14};
    const int i8x[8] = {4, 5, 6, 7, 8, 9, 10, 11};

    float luma_avg_8y = (luma_sum_4 + luma_sum_4y) / 8.0;
    float luma_avg_8x = (luma_sum_4 + luma_sum_4x) / 8.0;
    float luma_var_8y = 0.0;
    float luma_var_8x = 0.0;
    vec2 chroma_avg_8y = (chroma_sum_4 + chroma_sum_4y) / 8.0;
    vec2 chroma_avg_8x = (chroma_sum_4 + chroma_sum_4x) / 8.0;
    vec2 luma_chroma_cov_8y = vec2(0.0);
    vec2 luma_chroma_cov_8x = vec2(0.0);

    for (int i = 0; i < 8; i++) {
        luma_var_8y += pow(luma_pixels[i8y[i]] - luma_avg_8y, 2.0);
        luma_var_8x += pow(luma_pixels[i8x[i]] - luma_avg_8x, 2.0);
        luma_chroma_cov_8y += (luma_pixels[i8y[i]] - luma_avg_8y) * (chroma_pixels[i8y[i]] - chroma_avg_8y);
        luma_chroma_cov_8x += (luma_pixels[i8x[i]] - luma_avg_8x) * (chroma_pixels[i8x[i]] - chroma_avg_8x);
    }

    vec2 alpha_8y = luma_chroma_cov_8y / max(luma_var_8y, 1e-6);
    vec2 alpha_8x = luma_chroma_cov_8x / max(luma_var_8x, 1e-6);
    vec2 beta_8y = chroma_avg_8y - alpha_8y * luma_avg_8y;
    vec2 beta_8x = chroma_avg_8x - alpha_8x * luma_avg_8x;
    vec2 chroma_pred_8y = clamp(alpha_8y * luma_zero + beta_8y, 0.0, 1.0);
    vec2 chroma_pred_8x = clamp(alpha_8x * luma_zero + beta_8x, 0.0, 1.0);
    vec2 chroma_pred_8 = mix(chroma_pred_8y, chroma_pred_8x, 0.5);
#endif

#if (USE_12_TAP_REGRESSION == 1 && USE_8_TAP_REGRESSIONS == 1)
    output_pix.xy = mix(chroma_spatial, mix(chroma_pred_12, chroma_pred_8, 0.5), mix_coeff);
#elif (USE_12_TAP_REGRESSION == 1 && USE_8_TAP_REGRESSIONS == 0)
    output_pix.xy = mix(chroma_spatial, chroma_pred_12, mix_coeff);
#elif (USE_12_TAP_REGRESSION == 0 && USE_8_TAP_REGRESSIONS == 1)
    output_pix.xy = mix(chroma_spatial, chroma_pred_8, mix_coeff);
#else
    output_pix.xy = chroma_spatial;
#endif

    output_pix.xy = clamp(output_pix.xy, 0.0, 1.0);
    return output_pix;
}

//!PARAM distance_coeff
//!TYPE float
//!MINIMUM 0.0
2.0

//!PARAM intensity_coeff
//!TYPE float
//!MINIMUM 0.0
128.0

//!HOOK CHROMA
//!BIND CHROMA
//!BIND LUMA
//!DESC Chroma From Luma Prediction (Smoothing Chroma)

float comp_w(vec2 spatial_distance, float intensity_distance) {
    return max(100.0 * exp(-distance_coeff * pow(length(spatial_distance), 2.0) - intensity_coeff * pow(intensity_distance, 2.0)), 1e-32);
}

vec4 hook() {
    vec4 output_pix = vec4(0.0, 0.0, 0.0, 1.0);
    float luma_zero = LUMA_texOff(0).x;
    float wt = 0.0;
    vec2 ct = vec2(0.0);

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            vec2 chroma_pixels = CHROMA_texOff(vec2(i, j)).xy;
            float luma_pixels = LUMA_texOff(vec2(i, j)).x;
            float w = comp_w(vec2(i, j), luma_zero - luma_pixels);
            wt += w;
            ct += w * chroma_pixels;
        }
    }

    output_pix.xy = clamp(ct / wt, 0.0, 1.0);
    return output_pix;
}
'''
