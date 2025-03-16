from vstools import vs
from vstools import core
from typing import Union
import functools

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
            
            cache[scene_start:scene_start+scene_length] = [ratio] * scene_length
        
        fout.props['ratio'] = cache[n]
        return fout
    
    assert clip.format.bits_per_sample == 8
    assert 0 < freq_range < 0.5
    
    freq_drop_range = 1 - freq_range
    freq_drop_lr = int(clip.width * freq_drop_range)
    freq_drop_bt = int(clip.height * freq_drop_range)
    
    fft = core.fftspectrum_rs.FFTSpectrum(clip)

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