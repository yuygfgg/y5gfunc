'''
Yuygfgg's collection for vapoursynth video filtering and encoding stuff.
'''

from .expr import postfix2infix, infix2postfix, optimize_akarin_expr, ex_planes
from .filter import (
    SynDeband, BM3DPreset, Fast_BM3DWrapper, TIVTC_VFR, DBMask, AnimeMask, get_oped_mask,
    kirsch, prewitt, retinex_edgemask, comb_mask, rescale, descale_cropping_args, DescaleMode,
    Descale, rgb2opp, opp2rgb, SSIM_downsample, is_stripe, convolution, maximum, minimum, inflate, deflate, scd_koala, temporal_stabilize,
    nn2x, nn2x_aa, get_peak_value_full, is_optimized_cpu
)
from .vfx import draw_line, draw_circle, draw_ellipse, draw_bezier_curve, draw_mandelbrot_zoomer, draw_spiral, draw_3d_cube, render_triangle_scene, render_model_scene, load_mesh, rotate_image
from .preview import reset_output_index, set_preview, screen_shot
from .source import WobblySource, load_source, get_frame_timestamp, clip_to_timecodes
from .encode import (
    encode_audio, extract_audio_tracks, ProcessMode, TrackConfig, AudioConfig, get_bd_chapter, get_mkv_chapter,
    mux_mkv, QcMode, ReturnType, encode_check, subset_fonts, extract_pgs_subtitles, get_language_by_trackid, encode_video,
    check_audio_stream_lossless
)
from .shader import cfl_shader, KrigBilateral, fsrcnnx_x2, artcnn_c4f16, artcnn_c4f16_DS, artcnn_c4f32, artcnn_c4f32_DS, LazyVariable
from .utils import ranger, PickFrames

__version__ = "0.0.1"

# aliases
output = set_preview

__all__ = [
    'convolution',
    'maximum',
    'minimum',
    'inflate',
    'deflate',
    'scd_koala',
    'temporal_stabilize',
    'postfix2infix',
    'infix2postfix',
    'optimize_akarin_expr',
    'ex_planes',
    'SynDeband',
    'BM3DPreset',
    'Fast_BM3DWrapper',
    'TIVTC_VFR',
    'DBMask',
    'AnimeMask',
    'get_oped_mask',
    'kirsch',
    'prewitt',
    'retinex_edgemask',
    'comb_mask',
    'rescale',
    'descale_cropping_args',
    'DescaleMode',
    'Descale',
    'rgb2opp',
    'opp2rgb',
    'SSIM_downsample',
    'draw_line',
    'draw_circle',
    'draw_ellipse',
    'draw_bezier_curve',
    'draw_mandelbrot_zoomer',
    'draw_spiral',
    'draw_3d_cube',
    'render_triangle_scene',
    'render_model_scene',
    'load_mesh',
    'rotate_image',
    'nn2x',
    'nn2x_aa',
    'get_peak_value_full',
    'is_optimized_cpu',
    'is_stripe',
    'reset_output_index',
    'set_preview',
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
    'get_bd_chapter',
    'get_mkv_chapter',
    'mux_mkv',
    'QcMode',
    'ReturnType',
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