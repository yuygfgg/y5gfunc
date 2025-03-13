import vapoursynth as vs
from vapoursynth import core

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