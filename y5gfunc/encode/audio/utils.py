LOSSLESS_CODEC_NAMES = {
    "truehd",
    "flac",
    "alac",
    "mlp",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
    "pcm_s16be",
    "pcm_s24be",
    "pcm_s32be",
    "pcm_f32be",
    "pcm_f64be",
    "pcm_alaw",
    "pcm_mulaw",
    "pcm_u8",
}


def check_audio_stream_lossless(stream: dict) -> bool:
    """
    Checks if an audio stream from ffprobe represents a lossless format.

    Args:
        stream: A dictionary representing an audio stream from ffprobe JSON.

    Returns:
        True if the stream is identified as lossless, False otherwise.
    """
    codec_name = str(stream.get("codec_name", "")).lower()
    profile = str(stream.get("profile", "")).lower()

    if "dts" in profile or codec_name == "dts":
        if profile == "dts-hd ma":
            return True
        else:
            return False

    if codec_name in LOSSLESS_CODEC_NAMES:
        return True

    return False
