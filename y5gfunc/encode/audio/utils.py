LOSSLESS_CODECS = {'truehd', 'flac', 'alac', 'mlp', 'pcm'}
LOSSLESS_PROFILES = {'dts-hd ma'}

def check_audio_stream_lossless(stream) -> bool:
    codec_name = stream.get('codec_name', '').lower()
    codec_long_name = stream.get('codec_long_name', '').lower()
    profile = stream.get('profile', '').lower()
    
    return any([
            any(codec in codec_long_name for codec in LOSSLESS_CODECS),
            any(codec in codec_name for codec in LOSSLESS_CODECS),
            profile in LOSSLESS_PROFILES
        ])