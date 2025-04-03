'''
Yuygfgg's collection for vapoursynth video filtering and encoding stuff, written by others and me.
'''

from .expr import postfix2infix, infix2postfix, optimize_akarin_expr, ex_planes
from .filter import (
    SynDeband, Fast_BM3DWrapper, TIVTC_VFR, DBMask, get_oped_mask,
    cambi_mask, kirsch, retinex_edgemask, comb_mask, rescale, Descale, rgb2opp, 
    opp2rgb, is_stripe, convolution, maximum, minimum, inflate, deflate, scd_koala,
    draw_line, draw_circle, draw_ellipse, draw_bezier_curve, draw_mandelbrot_zoomer,
    scale_value_full, get_peak_value_full, is_optimized_cpu
)
from .preview import reset_output_index, output, screen_shot
from .source import WobblySource, load_source, get_frame_timestamp, clip_to_timecodes
from .encode import (
    encode_audio, extract_audio_tracks, ProcessMode, TrackConfig, AudioConfig, get_bd_chapter, get_mkv_chapter,
    mux_mkv, encode_check, subset_fonts, extract_pgs_subtitles, get_language_by_trackid, encode_video,
    check_audio_stream_lossless, LOSSLESS_CODECS, LOSSLESS_PROFILES
)
from .shader import cfl_shader, KrigBilateral, fsrcnnx_x2, artcnn_c4f16, artcnn_c4f16_DS, artcnn_c4f32, artcnn_c4f32_DS, LazyVariable
from .utils import ranger, PickFrames

__version__ = "0.0.1"

__all__ = [
    'convolution',
    'maximum',
    'minimum',
    'inflate',
    'deflate',
    'scd_koala',
    'postfix2infix',
    'infix2postfix',
    'optimize_akarin_expr',
    'ex_planes',
    'SynDeband', 
    'Fast_BM3DWrapper',
    'TIVTC_VFR',
    'DBMask',
    'get_oped_mask',
    'cambi_mask',
    'kirsch',
    'retinex_edgemask',
    'comb_mask',
    'rescale',
    'Descale',
    'rgb2opp',
    'opp2rgb',
    'draw_line',
    'draw_circle',
    'draw_ellipse',
    'draw_bezier_curve',
    'draw_mandelbrot_zoomer',
    'scale_value_full',
    'get_peak_value_full',
    'is_optimized_cpu',
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
    'check_audio_stream_lossless',
    'LOSSLESS_CODECS',
    'LOSSLESS_PROFILES',
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