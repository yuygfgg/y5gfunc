import subprocess
from pathlib import Path

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