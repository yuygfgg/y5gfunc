from typing import Callable
from vstools import core
from vstools import vs
import numpy as np
from functools import partial
import muvsfunc

def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """cv2.HISTCMP_CORREL"""
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()
    
    x_mean = np.mean(hist1)
    y_mean = np.mean(hist2)
    
    numerator = np.sum((hist1 - x_mean) * (hist2 - y_mean))
    denominator = np.sqrt(np.sum((hist1 - x_mean)**2) * np.sum((hist2 - y_mean)**2))
    
    if denominator == 0:
        return 0
    return numerator / denominator

# TODO: make scd_koala online
# modified from https://github.com/KwaiVGI/Koala-36M/blob/main/trainsition_detect/VideoTransitionAnalyzer.py and https://github.com/Breakthrough/PySceneDetect/pull/459
def scd_koala(
    clip: vs.VideoNode,
    min_scene_len: int = 12,
    filter_size: int = 3,
    window_size: int = 8,
    deviation: float = 3.0,
    edge_func: Callable[[vs.VideoNode], vs.VideoNode] = partial(muvsfunc.AnimeMask, mode=-1)
) -> vs.VideoNode:

    num_frames = clip.num_frames
    
    resized_clip = core.resize.Bicubic(clip, width=256, height=256, format=vs.RGB24)
    
    gray_clip = core.std.ShufflePlanes(resized_clip, planes=0, colorfamily=vs.GRAY)
    small_gray = core.resize.Bicubic(gray_clip, width=128, height=128)
    
    edges = edge_func(small_gray)
    
    combined_edges = core.akarin.Expr([small_gray, edges], expr="x y max")
    prev_edge_clip = combined_edges.std.DuplicateFrames(0).std.Trim(first=0, length=num_frames)
    ssim_result = muvsfunc.SSIM(prev_edge_clip, combined_edges)
    
    scores = []

    for n, curr_frame in enumerate(resized_clip.frames()):
        if n == 0:
            prev_frame = curr_frame
            continue
        
        prev_array = np.dstack([np.array(prev_frame[p]) for p in range(prev_frame.format.num_planes)]) # type: ignore
        curr_array = np.dstack([np.array(curr_frame[p]) for p in range(curr_frame.format.num_planes)]) # type: ignore
        
        prev_hist = []
        curr_hist = []
        
        for c in range(3):
            channel_data = prev_array[:, :, c]
            mask = (channel_data > 0) & (channel_data < 255)
            hist, _ = np.histogram(channel_data[mask], bins=254, range=(1, 255))
            prev_hist.append(hist)
            
            channel_data = curr_array[:, :, c]
            mask = (channel_data > 0) & (channel_data < 255)
            hist, _ = np.histogram(channel_data[mask], bins=254, range=(1, 255))
            curr_hist.append(hist)
        
        delta_histogram = histogram_correlation(np.array(prev_hist), np.array(curr_hist))
        
        ssim_frame = ssim_result.get_frame(n)
        delta_edges = ssim_frame.props.get("PlaneSSIM", 0)
        
        score = 4.61480465 * delta_histogram + 3.75211168 * delta_edges - 5.485968377115124 # type: ignore
        scores.append(score)
        prev_frame = curr_frame
    
    cut_found = [score < 0.0 for score in scores]
    cut_found.append(True)
    
    filter_kernel = np.ones(filter_size) / filter_size
    filtered = np.convolve(scores, filter_kernel, mode="same")
    
    for i in range(len(scores)):
        if i >= window_size and filtered[i] < float(filter_size) / float(filter_size + 1):
            window = filtered[i - window_size : i]
            threshold = np.mean(window) - (deviation * np.std(window))
            if filtered[i] < threshold:
                cut_found[i] = True
    
    scene_cuts = []
    last_cut = 0
    
    for i in range(len(cut_found)):
        if cut_found[i]:
            current_frame = i + 1
            scene_length = current_frame - last_cut
            
            if scene_length >= min_scene_len:
                scene_cuts.append(current_frame)
            
            last_cut = current_frame
    
    def set_scenecut_prop(n, f, scene_cuts):
        fout = f.copy()
        if n in scene_cuts:
            fout.props._Scenecut = 1
        else:
            fout.props._Scenecut = 0
        return fout
    
    marked_clip = core.std.ModifyFrame(clip, clip, partial(set_scenecut_prop, scene_cuts=scene_cuts))
    
    return marked_clip