'''
Yuygfgg's collection for vapoursynth video filtering and encoding stuff, written by others and me.
'''

from .expr import postfix2infix
from .filter import (
    SynDeband, Fast_BM3DWrapper, TIVTC_VFR, DBMask, get_oped_mask,
    cambi_mask, kirsch, retinex_edgemask, rescale, Descale, is_stripe,
    convolution, maximum, minimum
)
from .preview import reset_output_index, output, screen_shot
from .source import WobblySource, load_source, get_frame_timestamp, clip_to_timecodes
from .encode import (
    encode_audio, extract_audio_tracks, ProcessMode, TrackConfig, AudioConfig, get_bd_chapter, get_mkv_chapter,
    mux_mkv, encode_check, subset_fonts, extract_pgs_subtitles, get_language_by_trackid, encode_video
)
from .shader import cfl_shader, KrigBilateral, fsrcnnx_x2, artcnn_c4f16, artcnn_c4f16_DS, artcnn_c4f32, artcnn_c4f32_DS, LazyVariable
from .utils import ranger, PickFrames

__version__ = "0.0.1"

__all__ = [
    'convolution',
    'maximum',
    'minimum',
    'postfix2infix',
    'SynDeband', 
    'Fast_BM3DWrapper',
    'TIVTC_VFR',
    'DBMask',
    'get_oped_mask',
    'cambi_mask',
    'kirsch',
    'retinex_edgemask',
    'rescale',
    'Descale',
    'is_stripe',
    'reset_output_index',
    'output',
    'screen_shot',
    'WobblySource',
    'load_source',
    'get_frame_timestamp',
    'clip_to_timecodes',
    'encode_audio',
    'extract_audio_tracks',
    'ProcessMode',
    'TrackConfig',
    'AudioConfig',
    'get_bd_chapter',
    'get_mkv_chapter',
    'mux_mkv',
    'encode_check',
    'subset_fonts',
    'extract_pgs_subtitles',
    'get_language_by_trackid',
    'encode_video',
    'cfl_shader',
    'KrigBilateral',
    'fsrcnnx_x2',
    'artcnn_c4f16',
    'artcnn_c4f16_DS',
    'artcnn_c4f32',
    'artcnn_c4f32_DS',
    'LazyVariable',
    'ranger',
    'PickFrames'
]