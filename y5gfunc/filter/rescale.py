import vapoursynth as vs
from vapoursynth import core
import vsutil
from typing import Callable, Union, Optional
import functools
from ..utils import ranger
from .morpho import maximum

# TODO: use vs-jetpack Rescalers, handle asymmetrical descales
# inspired by https://skyeysnow.com/forum.php?mod=viewthread&tid=58390
def rescale(
    clip: vs.VideoNode,
    descale_kernel: Union[str, list[str]] = "Debicubic",
    src_height: Union[Union[float, int], list[Union[float, int]]] = 720,
    bw: Optional[Union[int, list[int]]] = None,
    bh: Optional[Union[int, list[int]]]  = None,
    show_upscaled: bool = False,
    show_fft: bool = False,
    detail_mask_threshold: float = 0.05,
    use_detail_mask: bool = True,
    show_detail_mask: bool = False,
    show_common_mask: bool = False,
    nnedi3_args: dict = {'field': 1, 'nsize': 4, 'nns': 4, 'qual': 2},
    taps: Union[int, list[int]] = 4,
    b: Union[Union[float, int], list[Union[float, int]]] = 0.33,
    c: Union[Union[float, int], list[Union[float, int]]] = 0.33,
    threshold_max: float = 0.007,
    threshold_min: float = -1,
    show_osd: bool = True,
    ex_thr: Union[float, int] = 0.015, 
    norm_order: int = 1, 
    crop_size: int = 5,
    exclude_common_mask: bool = True,
    scene_stable: bool = False,
    scene_descale_threshold_ratio: float = 0.5,
    scenecut_threshold: Union[float, int] = 0.1,
    opencl = True
) -> Union[
    vs.VideoNode,
    tuple[vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode]
]:
    
    '''
    To rescale from multiple native resolution, use this func for every possible src_height, then choose the largest MaxDelta one.
    
    e.g. 
    rescaled1, detail_mask1, osd1 = rescale(clip=srcorg, src_height=ranger(714.5, 715, 0.025)+[713, 714, 716, 717], bw=1920, bh=1080, descale_kernel="Debicubic", b=1/3, c=1/3, show_detail_mask=True)
    rescaled2, detail_mask2, osd2 = rescale(clip=srcorg, src_height=ranger(955, 957,0.1)+[953, 954, 958], bw=1920, bh=1080, descale_kernel="Debicubic", b=1/3, c=1/3, show_detail_mask=True)

    select_expr = "src0.MaxDelta src0.Descaled * src1.MaxDelta src1.Descaled * argmax2"

    osd = core.akarin.Select([osd1, osd2], [rescaled1, rescaled2], select_expr)
    src = core.akarin.Select([rescaled1, rescaled2], [rescaled1, rescaled2], select_expr)
    detail_mask = core.akarin.Select([detail_mask1, detail_mask2], [rescaled1, rescaled2], select_expr)
    '''
    
    from itertools import product
    from getfnative import descale_cropping_args
    from muvsfunc import SSIM_downsample
    
    KERNEL_MAP = {
        "Debicubic": 1,
        "Delanczos": 2,
        "Debilinear": 3,
        "Despline16": 5,
        "Despline36": 6,
        "Despline64": 7
    }
    
    descale_kernel = [descale_kernel] if isinstance(descale_kernel, str) else descale_kernel
    src_height = [src_height] if isinstance(src_height, (int, float)) else src_height
    if bw is None:
        bw = clip.width
    if bh is None:
        bh = clip.height
    bw = [bw] if isinstance(bw, int) else bw
    bh = [bh] if isinstance(bh, int) else bh
    taps = [taps] if isinstance(taps, int) else taps
    b = [b] if isinstance(b, (float, int)) else b
    c = [c] if isinstance(c, (float, int)) else c
    
    clip = vsutil.depth(clip, 32)
    
    def scene_descale(
        n: int,
        f: list[vs.VideoFrame],
        cache: list[int],
        prefetch: vs.VideoNode,
        length: int,
        scene_descale_threshold_ratio: float = scene_descale_threshold_ratio
    ) -> vs.VideoFrame:
    
        fout = f[0].copy()
        if n == 0 or n == prefetch.num_frames:
            fout.props['_SceneChangePrev'] = 1

        if cache[n] == -1: # not cached
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

            min_index_buffer = [0] * length
            num_descaled = 0

            while (i < prefetch.num_frames):
                frame = prefetch.get_frame(i)
                min_index_buffer[frame.props['MinIndex']] += 1 # type: ignore
                if frame.props['Descaled']:
                    num_descaled += 1
                scene_length += 1
                i += 1
                if frame.props['_SceneChangeNext'] == 1: # scene end
                    break

            scene_min_index = max(enumerate(min_index_buffer), key=lambda x: x[1])[0] if num_descaled >= scene_descale_threshold_ratio * scene_length else length
            
            i = scene_start
            for i in ranger(scene_start, scene_start+scene_length, step=1): # write scene prop
                cache[i] = scene_min_index

        fout.props['SceneMinIndex'] = cache[n]
        return fout

    def _get_resize_name(descale_name: str) -> str:
        if descale_name.startswith('De'):
            return descale_name[2:].capitalize()
        return descale_name
    
    # modified from kegefunc._generate_descale_mask()
    def _generate_detail_mask(source: vs.VideoNode, upscaled: vs.VideoNode, detail_mask_threshold: float = detail_mask_threshold) -> vs.VideoNode:
        mask = core.akarin.Expr([source, upscaled], 'src0 src1 - abs').std.Binarize(threshold=detail_mask_threshold)
        mask = vsutil.iterate(mask, maximum, 3)
        mask = vsutil.iterate(mask, core.std.Inflate, 3)
        return mask

    def _mergeuv(clipy: vs.VideoNode, clipuv: vs.VideoNode) -> vs.VideoNode:
        return core.std.ShufflePlanes([clipy, clipuv], [0, 1, 2], vs.YUV)
    
    def _generate_common_mask(detail_mask_clips: list[vs.VideoNode]) -> vs.VideoNode:
        load_expr = [f'src{i} * ' for i in range(len(detail_mask_clips))]
        merge_expr = ' '.join(load_expr)
        merge_expr = merge_expr[:4] + merge_expr[6:]
        return core.akarin.Expr(clips=detail_mask_clips, expr=merge_expr)
    
    def _select_per_frame(
        reference: vs.VideoNode,
        upscaled_clips: list[vs.VideoNode],
        candidate_clips: list[vs.VideoNode],
        params_list: list[dict],
        common_mask_clip: vs.VideoNode,
        threshold_max: float = threshold_max,
        threshold_min: float = threshold_min,
        ex_thr: float = ex_thr,
        norm_order: int = norm_order,
        crop_size: int = crop_size
    ) -> vs.VideoNode:
        
        def _crop(clip: vs.VideoNode, crop_size: int = crop_size) -> vs.VideoNode:
            return clip.std.CropRel(*([crop_size] * 4)) if crop_size > 0 else clip
        
        if len(upscaled_clips) != len(candidate_clips) or len(upscaled_clips) != len(params_list):
            raise ValueError("upscaled_clips, rescaled_clips, and params_list must have the same length.")

        calc_diff_expr = f"src0 src1 - abs dup {ex_thr} > swap {norm_order} pow 0 ? src2 - 0 1 clip"
        
        diffs = [core.akarin.Expr([_crop(reference), _crop(upscaled_clip), _crop(common_mask_clip)], calc_diff_expr).std.PlaneStats() for upscaled_clip in upscaled_clips]

        for diff in diffs:
            diff = diff.akarin.PropExpr(lambda: {'PlaneStatsAverage': f'x.PlaneStatsAverage {1 / norm_order} pow'})
        
        load_PlaneStatsAverage_exprs = [f'src{i}.PlaneStatsAverage' for i in range(len(diffs))]
        diff_expr = ' '.join(load_PlaneStatsAverage_exprs)
    
        min_index_expr = diff_expr + f' argmin{len(diffs)}'
        min_diff_expr = diff_expr + f' sort{len(diffs)} min_diff! drop{len(diffs)-1} min_diff@'
        
        max_index_expr = diff_expr + f' argmax{len(diffs)}'
        max_diff_expr = diff_expr + f' sort{len(diffs)} drop{len(diffs)-1}'
        
        max_delta_expr = max_diff_expr + " " + min_diff_expr + " / "

        def props() -> dict[str, str]:
            d = {
                'MinIndex': min_index_expr,
                'MinDiff': min_diff_expr,
                'MaxIndex': max_index_expr,
                'MaxDiff': max_diff_expr,
                "MaxDelta": max_delta_expr
            }
            for i in range(len(diffs)):
                d[f'Diff{i}'] = load_PlaneStatsAverage_exprs[i]
                params = params_list[i]
                d[f'KernelId{i}'] = KERNEL_MAP.get(params["Kernel"], 0) # type: ignore
                d[f'Bw{i}'] = params.get("BaseWidth", 0), # type: ignore
                d[f'Bh{i}'] = params.get("BaseHeight", 0)
                d[f'SrcHeight{i}'] = params['SrcHeight']
                d[f'B{i}'] = params.get('B', 0)
                d[f'C{i}'] = params.get('C', 0)
                d[f'Taps{i}'] = params.get('Taps', 0)
            return d

        prop_src = core.akarin.PropExpr(diffs, props)
        
        # Kernel is different because it's a string.
        for i in range(len(diffs)):
            prop_src = core.akarin.Text(prop_src, params_list[i]["Kernel"], prop=f"Kernel{i}")

        minDiff_clip = core.akarin.Select(clip_src=candidate_clips, prop_src=[prop_src], expr='x.MinIndex')

        final_clip = core.akarin.Select(clip_src=[reference, minDiff_clip], prop_src=[prop_src], expr=f'x.MinDiff {threshold_max} > x.MinDiff {threshold_min} <= or 0 1 ?')

        final_clip = final_clip.std.CopyFrameProps(prop_src)
        final_clip = core.akarin.PropExpr([final_clip], lambda: {'Descaled': f'x.MinDiff {threshold_max} <= x.MinDiff {threshold_min} > and'})

        return final_clip

    def _fft(clip: vs.VideoNode, grid: bool = True) -> vs.VideoNode:
        return core.fftspectrum_rs.FFTSpectrum(clip=vsutil.depth(clip,8))
    
    
    if hasattr(core, "sneedif") and opencl:
        nnedi3 = functools.partial(core.sneedif.NNEDI3, **nnedi3_args)
        def nn2x(nn2x) -> vs.VideoNode:
            return nnedi3(nnedi3(nn2x, dh=True), dw=True)
    elif hasattr(core, "nnedi3cl") and opencl:
        nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)
        def nn2x(nn2x) -> vs.VideoNode:
            return nnedi3(nnedi3(nn2x, dh=True), dw=True)
    else:
        nnedi3 = functools.partial(core.nnedi3.nnedi3, **nnedi3_args)
        def nn2x(nn2x) -> vs.VideoNode:
            return nnedi3(nnedi3(nn2x, dh=True).std.Transpose(), dh=True).std.Transpose()
    
    upscaled_clips: list[vs.VideoNode] = []
    rescaled_clips: list[vs.VideoNode] = []
    detail_masks: list[vs.VideoNode] = []
    params_list: list[dict] = []
    
    src_luma = vsutil.get_y(clip)
    
    for kernel_name, sh, base_w, base_h, _taps, _b, _c in product(descale_kernel, src_height, bw, bh, taps, b, c):
        extra_params: dict[str, dict[str, Union[float, int]]] = {}
        if kernel_name == "Debicubic":
            extra_params = {
                'dparams': {'b': _b, 'c': _c},
                'rparams': {'filter_param_a': _b, 'filter_param_b': _c}
            }
        elif kernel_name == "Delanczos":
            extra_params = {
                'dparams': {'taps': _taps},
                'rparams': {'filter_param_a': _taps}
            }
        else:
            extra_params = {}

        dargs = descale_cropping_args(clip=clip, src_height=sh, base_height=base_h, base_width=base_w)

        descaled = getattr(core.descale, kernel_name)(
            src_luma,
            **dargs,
            **extra_params.get('dparams', {})
        )

        upscaled = getattr(core.resize2, _get_resize_name(kernel_name))(
            descaled,
            width=clip.width,
            height=clip.height,
            src_left=dargs['src_left'],
            src_top=dargs['src_top'],
            src_width=dargs['src_width'],
            src_height=dargs['src_height'],
            **extra_params.get('rparams', {})
        )

        n2x = nn2x(descaled)
        
        rescaled = SSIM_downsample(
            clip=n2x,
            w=clip.width,
            h=clip.height,
            sigmoid=False,
            src_left=dargs['src_left'] * 2 - 0.5,
            src_top=dargs['src_top'] * 2 - 0.5,
            src_width=dargs['src_width'] * 2,
            src_height=dargs['src_height'] * 2
        )

        upscaled_clips.append(upscaled)
        rescaled_clips.append(rescaled)
        
        if use_detail_mask or show_detail_mask or exclude_common_mask:
            detail_mask = _generate_detail_mask(src_luma, upscaled, detail_mask_threshold)
            detail_masks.append(detail_mask)

        params_list.append({
            'Kernel': kernel_name,
            'SrcHeight': sh,
            'BaseWidth': base_w,
            'BaseHeight': base_h,
            'Taps': _taps,
            'B': _b,
            'C': _c
        })
    
    common_mask_clip = _generate_common_mask(detail_mask_clips=detail_masks) if exclude_common_mask else core.std.BlankClip(clip=detail_masks[0], color=0)  
    if not scene_stable:
        rescaled = _select_per_frame(reference=src_luma, upscaled_clips=upscaled_clips, candidate_clips=rescaled_clips, params_list=params_list, common_mask_clip=common_mask_clip)
        detail_mask = core.akarin.Select(clip_src=detail_masks, prop_src=rescaled, expr="src0.MinIndex")
        detail_mask = core.akarin.Select(clip_src=[core.std.BlankClip(clip=detail_mask), detail_mask], prop_src=rescaled, expr="src0.Descaled")
        upscaled = core.akarin.Select(clip_src=upscaled_clips, prop_src=rescaled, expr="src0.MinIndex")
    else:
        # detail mask: matched one when descaling, otherwise blank clip
        # upscaled clip: matched one when descaling, otherwise frame level decision
        # rescaled clip: mostly-choosed index in a scene when descaling, otherwise src_luma
        # 'Descaled', 'SceneMinIndex': scene-level information
        # other props: frame-level information
        per_frame = _select_per_frame(reference=src_luma, upscaled_clips=upscaled_clips, candidate_clips=upscaled_clips, params_list=params_list, common_mask_clip=common_mask_clip)
        upscaled_per_frame = core.akarin.Select(clip_src=upscaled_clips, prop_src=per_frame, expr="src0.MinIndex")
        scene = core.misc.SCDetect(clip, scenecut_threshold)
        prefetch = core.std.BlankClip(clip)
        prefetch = core.akarin.PropExpr([scene, per_frame], lambda: {'_SceneChangeNext': 'x._SceneChangeNext', '_SceneChangePrev': 'x._SceneChangePrev', 'MinIndex': 'y.MinIndex', 'Descaled': 'y.Descaled'})
        cache = [-1] * clip.num_frames
        length = len(upscaled_clips)
        per_scene = core.std.ModifyFrame(per_frame, [per_frame, per_frame], functools.partial(scene_descale, prefetch=prefetch, cache=cache, length=length, scene_descale_threshold_ratio=scene_descale_threshold_ratio))
        rescaled = core.akarin.Select(rescaled_clips + [src_luma], [per_scene], 'src0.SceneMinIndex')
        rescaled = core.std.CopyFrameProps(per_scene, per_scene)
        rescaled = core.akarin.PropExpr([rescaled], lambda: {'Descaled': f'x.SceneMinIndex {len(upscaled_clips)} = not'})
        detail_mask = core.akarin.Select(clip_src=detail_masks + [core.std.BlankClip(clip=detail_masks[0])], prop_src=rescaled, expr="src0.SceneMinIndex")
        upscaled = core.akarin.Select(clip_src=upscaled_clips + [upscaled_per_frame], prop_src=rescaled, expr="src0.SceneMinIndex")

    if use_detail_mask:
        rescaled = core.std.MaskedMerge(rescaled, src_luma, detail_mask)

    final = _mergeuv(rescaled, clip) if clip.format.color_family == vs.YUV else rescaled
    
    if not scene_stable:
        format_string = (
            "\nMinIndex: {MinIndex}\n"
            "MinDiff: {MinDiff}\n"
            "Descaled: {Descaled}\n"
            f"Threshold_Max: {threshold_max}\n"
            f"Threshold_Min: {threshold_min}\n\n"
        )
    else:
        format_string = (
            "\nMinIndex: {MinIndex}\n"
            "MinDiff: {MinDiff}\n"
            "Descaled: {Descaled}\n"
            "SceneMinIndex: {SceneMinIndex}\n"
            f"Threshold_Max: {threshold_max}\n"
            f"Threshold_Min: {threshold_min}\n"
        )

    format_string += (
        "|    i   |          Diff          |   Kernel   | SrcHeight |    Bw     |     Bh    | B      | C      | Taps   |\n"
        "|--------|------------------------|------------|-----------|-----------|-----------|--------|--------|--------|\n"
    )

    for i in range(len(upscaled_clips)):
        format_string += (
            f"| {i:04}   | {{Diff{i}}}  | {{Kernel{i}}}  | {{SrcHeight{i}}}   | "
            f"{{Bw{i}}} | {{Bh{i}}} |"
            f"{{B{i}}}   | {{C{i}}}   | {{Taps{i}}}   |\n"
        )
    osd_clip = core.akarin.Text(final, format_string)

    if show_fft:
        src_fft = _fft(clip)
        rescaled_fft = _fft(final)
    
    # FUCK YOU PYLINT
    if show_upscaled: 
        if show_common_mask:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, common_mask_clip, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, upscaled, detail_mask, common_mask_clip, osd_clip
                elif show_fft:
                    return final, upscaled, common_mask_clip, src_fft, rescaled_fft, osd_clip
                else:
                    return final, upscaled, common_mask_clip, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, common_mask_clip, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, upscaled, detail_mask, common_mask_clip
                elif show_fft:
                    return final, upscaled, common_mask_clip, src_fft, rescaled_fft
                else:
                    return final, upscaled, common_mask_clip
        else:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, upscaled, detail_mask, osd_clip
                elif show_fft:
                    return final, upscaled, src_fft, rescaled_fft, osd_clip
                else:
                    return final, upscaled, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, upscaled, detail_mask, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, upscaled, detail_mask
                elif show_fft:
                    return final, upscaled, src_fft, rescaled_fft
                else:
                    return final, upscaled
    else:
        if show_common_mask:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, detail_mask, common_mask_clip, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, detail_mask, common_mask_clip, osd_clip
                elif show_fft:
                    return final, common_mask_clip, src_fft, rescaled_fft, osd_clip
                else:
                    return final, common_mask_clip, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, detail_mask, common_mask_clip, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, detail_mask, common_mask_clip
                elif show_fft:
                    return final, common_mask_clip, src_fft, rescaled_fft
                else:
                    return final, common_mask_clip
        else:
            if show_osd:
                if show_detail_mask and show_fft:
                    return final, detail_mask, src_fft, rescaled_fft, osd_clip
                elif show_detail_mask:
                    return final, detail_mask, osd_clip
                elif show_fft:
                    return final, src_fft, rescaled_fft, osd_clip
                else:
                    return final, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, detail_mask, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, detail_mask
                elif show_fft:
                    return final, src_fft, rescaled_fft
                else:
                    return final
                
def Descale(
    src: vs.VideoNode,
    width: int,
    height: int,
    kernel: str,
    custom_kernel: Optional[Callable] = None,
    taps: int = 3,
    b: Union[int, float] = 0.0,
    c: Union[int, float] = 0.5,
    blur: Union[int, float] = 1.0,
    post_conv : Optional[list[Union[float, int]]] = None,
    src_left: Union[int, float] = 0.0,
    src_top: Union[int, float] = 0.0,
    src_width: Optional[Union[int, float]] = None,
    src_height: Optional[Union[int, float]] = None,
    border_handling: int = 0,
    ignore_mask: Optional[vs.VideoNode] = None,
    force: bool = False,
    force_h: bool = False,
    force_v: bool = False,
    opt: int = 0
) -> vs.VideoNode:
    
    def _get_resize_name(kernal_name: str) -> str:
        if kernal_name == 'Decustom':
            return 'ScaleCustom'
        if kernal_name.startswith('De'):
            return kernal_name[2:].capitalize()
        return kernal_name
    
    def _get_descaler_name(kernal_name: str) -> str:
        if kernal_name == 'ScaleCustom':
            return 'Decustom'
        if kernal_name.startswith('De'):
            return kernal_name
        return 'De' + kernal_name[0].lower() + kernal_name[1:]
    
    assert width > 0 and height > 0
    assert opt in [0, 1, 2]
    assert isinstance(src, vs.VideoNode) and src.format.id == vs.GRAYS
    
    kernel = kernel.capitalize()
    
    if src_width is None:
        src_width = width
    if src_height is None:
        src_height = height
    
    if width > src.width or height > src.height:
        kernel = _get_resize_name(kernel)
    else:
        kernel = _get_descaler_name(kernel)
    
    descaler = getattr(core.descale, kernel)
    assert callable(descaler)
    extra_params: dict[str, dict[str, Union[float, int, Callable]]] = {}
    if _get_descaler_name(kernel) == "Debicubic":
        extra_params = {
            'dparams': {'b': b, 'c': c},
        }
    elif _get_descaler_name(kernel) == "Delanczos":
        extra_params = {
            'dparams': {'taps': taps},
        }
    elif _get_descaler_name(kernel) == "Decustom":
        assert callable(custom_kernel)
        extra_params = {
            'dparams': {'custom_kernel': custom_kernel},
        }
    descaled = descaler(
        src=src,
        width=width,
        height=height,
        blur=blur,
        post_conv=post_conv,
        src_left=src_left,
        src_top=src_top,
        src_width=src_width,
        src_height=src_height,
        border_handling=border_handling,
        ignore_mask=ignore_mask,
        force=force,
        force_h=force_h,
        force_v=force_v,
        opt=opt,
        **extra_params.get('dparams', {})
    )
    
    assert isinstance(descaled, vs.VideoNode)
    
    return descaled