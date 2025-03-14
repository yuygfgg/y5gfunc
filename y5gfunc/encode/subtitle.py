from typing import Union, Optional
from pathlib import Path
import json
import subprocess
from .utils import get_language_by_trackid
from ..utils import resolve_path
import tempfile
import shutil

def subset_fonts(
    ass_path: Union[list[Union[str, Path]], str, Path], 
    fonts_path: Union[str, Path], 
    output_directory: Union[str, Path]
) -> Path:
    if isinstance(ass_path, (str, Path)):
        ass_path = [ass_path]
    
    ass_paths = [resolve_path(path) for path in ass_path]
    fonts_path = resolve_path(fonts_path)
    output_directory = resolve_path(output_directory)

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

    m2ts_path = resolve_path(m2ts_path)
    
    if output_dir is None:
        output_dir = m2ts_path.parent / f"{m2ts_path.stem}_subs"
    
    output_dir = resolve_path(output_dir)
    
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
        temp_dir = resolve_path(temp_dir)
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