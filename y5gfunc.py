import functools
from typing import List, Tuple, Union
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import vsutil

_output_index = 1

def reset_output_index(index=1):
    global _output_index
    _output_index = index

def output(*args, debug=True):
    import inspect
    
    def _get_variable_name(frame, clip):
        for var_name, var_val in frame.f_locals.items():
            if var_val is clip:
                return var_name
        return None

    def _add_text(clip, text, debug=debug):
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(f"_add_text expected a VideoNode, but got {type(clip)}")
        return core.text.Text(clip, text) if debug else clip
    
    global _output_index
    frame = inspect.currentframe().f_back # type: ignore
    used_indices = set()
    clips_to_process = []

    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, tuple) and len(item) == 2:
                    clip, index = item
                    if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                        raise TypeError("Tuple must be (VideoNode, int)")
                    if index < 0:
                        raise ValueError("Output index must be non-negative")
                    clips_to_process.append((clip, index))
                elif isinstance(item, vs.VideoNode):
                    clips_to_process.append((item, None))
                else:
                    raise TypeError(f"Invalid element in list/tuple: {type(item)}")
        elif isinstance(arg, vs.VideoNode):
            clips_to_process.append((arg, None))
        elif isinstance(arg, tuple) and len(arg) == 2:
            clip, index = arg
            if not isinstance(clip, vs.VideoNode) or not isinstance(index, int):
                raise TypeError("Tuple must be (VideoNode, int)")
            if index < 0:
                raise ValueError("Output index must be non-negative")
            clips_to_process.append((clip, index))
        else:
            raise TypeError(f"Invalid argument type: {type(arg)}")

    for clip, index in clips_to_process:
        if index is not None:
            if index in used_indices:
                raise ValueError(f"Output index {index} is already in use")
            used_indices.add(index)
            if index != 0:
                variable_name = _get_variable_name(frame, clip)
                if variable_name:
                    clip = _add_text(clip, f"{variable_name}")
                else:
                    clip = _add_text(clip, "Unknown Variable")
            clip.set_output(index)

    for clip, index in clips_to_process:
        if index is None:
            while _output_index in used_indices:
                _output_index += 1
            if _output_index != 0:
                variable_name = _get_variable_name(frame, clip)
                if variable_name:
                    clip = _add_text(clip, f"{variable_name}")
                else:
                    clip = _add_text(clip, "Unknown Variable")
            clip.set_output(_output_index)
            used_indices.add(_output_index)

# modified from rksfunc.BM3DWrapper()
def Fast_BM3DWrapper(
    clip: vs.VideoNode,
    bm3d=core.bm3dcpu,
    chroma: bool = True,
    
    sigma_Y: Union[float, int] = 1.2,
    radius_Y: int = 1,
    delta_sigma_Y: Union[float, int] = 0.6,
    
    sigma_chroma: Union[float, int] = 2.4,
    radius_chroma: int = 0,
    delta_sigma_chroma: Union[float, int] = 1.2
) -> vs.VideoNode:

    '''
    Note: delta_sigma_xxx is added to sigma_xxx in step basic.
    '''

    assert clip.format.id == vs.YUV420P16
    
    # modified from yvsfunc
    def _rgb2opp(clip: vs.VideoNode) -> vs.VideoNode:
        coef = [1/3, 1/3, 1/3, 0, 1/2, -1/2, 0, 0, 1/4, 1/4, -1/2, 0]
        opp = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.YUV, coef=coef)
        opp = core.std.SetFrameProps(opp, _Matrix=vs.MATRIX_UNSPECIFIED, BM3D_OPP=1)
        return opp

    # modified from yvsfunc
    def _opp2rgb(clip: vs.VideoNode) -> vs.VideoNode:
        coef = [1, 1, 2/3, 0, 1, -1, 2/3, 0, 1, 0, -4/3, 0]
        rgb = core.fmtc.matrix(clip, fulls=True, fulld=True, col_fam=vs.RGB, coef=coef)
        rgb = core.std.SetFrameProps(rgb, _Matrix=vs.MATRIX_RGB)
        rgb = core.std.RemoveFrameProps(rgb, 'BM3D_OPP')
        return rgb

    half_width = clip.width // 2  # half width
    half_height = clip.height // 2  # half height
    srcY_float, srcU_float, srcV_float = vsutil.split(clip.fmtc.bitdepth(bits=32))

    vbasic_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=srcY_float,
        sigma=sigma_Y + delta_sigma_Y,
        radius=radius_Y
    )

    vfinal_y = bm3d.BM3Dv2(
        clip=srcY_float,
        ref=vbasic_y,
        sigma=sigma_Y,
        radius=radius_Y
    )
    
    vyhalf = vfinal_y.resize.Spline36(half_width, half_height, src_left=-0.5)
    srchalf_444 = vsutil.join([vyhalf, srcU_float, srcV_float])
    srchalf_opp = _rgb2opp(mvf.ToRGB(input=srchalf_444, depth=32, matrix="709", sample=1))

    vbasic_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=srchalf_opp,
        sigma=sigma_chroma + delta_sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0
    )

    vfinal_half = bm3d.BM3Dv2(
        clip=srchalf_opp,
        ref=vbasic_half,
        sigma=sigma_chroma,
        chroma=chroma,
        radius=radius_chroma,
        zero_init=0
    )

    vfinal_half = _opp2rgb(vfinal_half).resize.Spline36(format=vs.YUV444PS, matrix=1)
    _, vfinal_u, vfinal_v = vsutil.split(vfinal_half)
    vfinal = vsutil.join([vfinal_y, vfinal_u, vfinal_v])
    return vfinal.fmtc.bitdepth(bits=16)

# modified from rksfunc.SynDeband()
def SynDeband(
    clip: vs.VideoNode, 
    r1: int = 14, 
    y1: int = 72, 
    uv1: int = 48, 
    r2: int = 30,
    y2: int = 48, 
    uv2: int = 32, 
    mstr: int = 6000, 
    inflate: int = 2,
    include_mask: bool = False, 
    kill: Union[vs.VideoNode, None] = None, 
    bmask: Union[vs.VideoNode, None] = None,
    limit: bool = True,
    limit_thry: float = 0.6,
    limit_thrc: float = 0.5,
    limit_elast: float = 1.75,
) -> Union[vs.VideoNode, Tuple[vs.VideoNode, vs.VideoNode]]:
    
    assert clip.format.id == vs.YUV420P16
    
    # copied from kagefunc.retinex_edgemask()
    def _retinex_edgemask(src: vs.VideoNode) -> vs.VideoNode:

        # copied from kagefunc.kirsch()
        def _kirsch(src: vs.VideoNode) -> vs.VideoNode:
            kirsch1 = src.std.Convolution(matrix=[ 5,  5,  5, -3,  0, -3, -3, -3, -3], saturate=False)
            kirsch2 = src.std.Convolution(matrix=[-3,  5,  5, -3,  0,  5, -3, -3, -3], saturate=False)
            kirsch3 = src.std.Convolution(matrix=[-3, -3,  5, -3,  0,  5, -3, -3,  5], saturate=False)
            kirsch4 = src.std.Convolution(matrix=[-3, -3, -3, -3,  0,  5, -3,  5,  5], saturate=False)
            return core.akarin.Expr([kirsch1, kirsch2, kirsch3, kirsch4], 'x y max z max a max')
            
        luma = vsutil.get_y(src)
        max_value = 1 if src.format.sample_type == vs.FLOAT else (1 << vsutil.get_depth(src)) - 1
        ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
        tcanny = ret.tcanny.TCanny(mode=1, sigma=1).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
        return core.akarin.Expr([_kirsch(luma), tcanny], f'x y + {max_value} min')
        
    if kill is None:
        kill = clip.rgvs.RemoveGrain([20, 11]).rgvs.RemoveGrain([20, 11])
    elif not kill:
        kill = clip
    grain = core.std.MakeDiff(clip, kill)
    f3kdb_params = {
        'grainy': 0,
        'grainc': 0,
        'sample_mode': 2,
        'blur_first': True,
        'dither_algo': 2,
    }
    f3k1 = kill.neo_f3kdb.Deband(r1, y1, uv1, uv1, **f3kdb_params)
    f3k2 = f3k1.neo_f3kdb.Deband(r2, y2, uv2, uv2, **f3kdb_params)
    if limit:
        f3k2 = mvf.LimitFilter(f3k2, kill, thr=limit_thry, thrc=limit_thrc, elast=limit_elast)
    if bmask is None:
        bmask = _retinex_edgemask(kill).std.Binarize(mstr)
        bmask = vsutil.iterate(bmask, core.std.Inflate, inflate)
    deband = core.std.MaskedMerge(f3k2, kill, bmask)
    deband = core.std.MergeDiff(deband, grain)
    if include_mask:
        return deband, bmask
    else:
        return deband

# modified from LoliHouse: https://share.dmhy.org/topics/view/478666_LoliHouse_LoliHouse_1st_Anniversary_Announcement_and_Gift.html)
def DBMask(clip: vs.VideoNode) -> vs.VideoNode:
    nr8: vs.VideoNode = mvf.Depth(clip, 8)
    nrmasks = core.tcanny.TCanny(nr8, sigma=0.8, op=2, mode=1, planes=[0, 1, 2]).akarin.Expr(["x 7 < 0 65535 ?",""], vs.YUV420P16)
    nrmaskb = core.tcanny.TCanny(nr8, sigma=1.3, t_h=6.5, op=2, planes=0)
    nrmaskg = core.tcanny.TCanny(nr8, sigma=1.1, t_h=5.0, op=2, planes=0)
    nrmask = core.akarin.Expr([nrmaskg, nrmaskb, nrmasks, nr8],["a 20 < 65535 a 48 < x 256 * a 96 < y 256 * z ? ? ?",""], vs.YUV420P16)
    nrmask = core.std.Maximum(nrmask, 0).std.Maximum(0).std.Minimum(0)
    nrmask = core.rgvs.RemoveGrain(nrmask, [20, 0])
    return nrmask

# inspired by https://skyeysnow.com/forum.php?mod=viewthread&tid=58390
def rescale(
    clip: vs.VideoNode,
    descale_kernel: Union[str, List[str]] = "Debicubic",
    src_height: Union[Union[float, int], List[Union[float, int]]] = 720,
    bw: Union[int, List[int]] = 0,
    bh: Union[int, List[int]] = 0,
    show_fft: bool = False,
    detail_mask_threshold: float = 0.05,
    use_detail_mask: bool = True,
    show_detail_mask: bool = False,
    show_common_mask: bool = False,
    nnedi3_args: dict = {'field': 1, 'nsize': 4, 'nns': 4, 'qual': 2},
    taps: Union[int, List[int]] = 4,
    b: Union[Union[float, int], List[Union[float, int]]] = 0.33,
    c: Union[Union[float, int], List[Union[float, int]]] = 0.33,
    threshold_max: float = 0.007,
    threshold_min: float = -1,
    show_osd: bool = True,
    ex_thr: float = 0.015, 
    norm_order: int = 1, 
    crop_size: int = 5,
    exclude_common_mask: bool = True
) -> Union[
    vs.VideoNode,
    Tuple[vs.VideoNode, vs.VideoNode],
    Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode],
    Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    Tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode]
]:
    
    '''
    To rescale from multiple native resolution, use this func for every possible src_height, then choose the largest MaxDelta one.
    
    e.g. 
    rescaled1, detail_mask1, osd1 = rescale(clip=srcorg, src_height=ranger(714.5, 715, 0.025)+[713, 714, 716, 717], bw=1920, bh=1080, descale_kernel="Debicubic", b=1/3, c=1/3, show_detail_mask=True) # type: ignore
    rescaled2, detail_mask2, osd2 = rescale(clip=srcorg, src_height=ranger(955, 957,0.1)+[953, 954, 958], bw=1920, bh=1080, descale_kernel="Debicubic", b=1/3, c=1/3, show_detail_mask=True) # type: ignore

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
    bw = [bw] if isinstance(bw, int) else bw
    bh = [bh] if isinstance(bh, int) else bh
    taps = [taps] if isinstance(taps, int) else taps
    b = [b] if isinstance(b, (float, int)) else b
    c = [c] if isinstance(c, (float, int)) else c
    
    clip = mvf.Depth(clip, 32)
    
    def _get_resize_name(descale_name: str) -> str:
        if descale_name.startswith('De'):
            return descale_name[2:].capitalize()
        return descale_name
    
    # modified from kegefunc._generate_descale_mask()
    def _generate_detail_mask(source: vs.VideoNode, upscaled: vs.VideoNode, detail_mask_threshold: float = detail_mask_threshold) -> vs.VideoNode:
        mask = core.akarin.Expr([source, upscaled], 'src0 src1 - abs').std.Binarize(threshold=detail_mask_threshold)
        mask = vsutil.iterate(mask, core.std.Maximum, 3)
        mask = vsutil.iterate(mask, core.std.Inflate, 3)
        return mask

    def _mergeuv(clipy: vs.VideoNode, clipuv: vs.VideoNode) -> vs.VideoNode:
        return core.std.ShufflePlanes([clipy, clipuv], [0, 1, 2], vs.YUV)
    
    def _generate_common_mask(detail_mask_clips: List[vs.VideoNode]) -> vs.VideoNode:
        load_expr = [f'src{i} * ' for i in range(len(detail_mask_clips))]
        merge_expr = ' '.join(load_expr)
        merge_expr = merge_expr[:4] + merge_expr[6:]
        return core.akarin.Expr(clips=detail_mask_clips, expr=merge_expr)
    
    def _select(
        reference: vs.VideoNode,
        upscaled_clips: List[vs.VideoNode],
        candidate_clips: List[vs.VideoNode],
        params_list: List[dict],
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

        load_PlaneStatsAverage_exprs = [f'src{i}.PlaneStatsAverage' for i in range(len(diffs))]
        diff_expr = ' '.join(load_PlaneStatsAverage_exprs)
    
        min_index_expr = diff_expr + f' argmin{len(diffs)}'
        min_diff_expr = diff_expr + f' sort{len(diffs)} min_diff! drop{len(diffs)-1} min_diff@'
        
        max_index_expr = diff_expr + f' argmax{len(diffs)}'
        max_diff_expr = diff_expr + f' sort{len(diffs)} drop{len(diffs)-1}'
        
        max_delta_expr = max_diff_expr + " " + min_diff_expr + " / "

        def props():
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
                d[f'Kernel{i}'] = KERNEL_MAP.get(params["Kernel"], 0) # type: ignore
                d[f'Bw{i}'] = params.get("BaseWidth", 0), # type: ignore
                d[f'Bh{i}'] = params.get("BaseHeight", 0)
                d[f'SrcHeight{i}'] = params['SrcHeight']
                d[f'B{i}'] = params.get('B', 0)
                d[f'C{i}'] = params.get('C', 0)
                d[f'Taps{i}'] = params.get('Taps', 0)
            return d

        prop_src = core.akarin.PropExpr(diffs, props)

        minDiff_clip = core.akarin.Select(clip_src=candidate_clips, prop_src=[prop_src], expr='x.MinIndex')

        final_clip = core.akarin.Select(clip_src=[reference, minDiff_clip], prop_src=[prop_src], expr=f'x.MinDiff {threshold_max} > x.MinDiff {threshold_min} <= or 0 1 ?')

        final_clip = final_clip.std.CopyFrameProps(prop_src)
        final_clip = core.akarin.PropExpr([final_clip], lambda: {'Descaled': f'x.MinDiff {threshold_max} <= x.MinDiff {threshold_min} > and'})

        return final_clip

    def _fft(clip: vs.VideoNode, grid: bool = True) -> vs.VideoNode:
        return core.fftspectrum.FFTSpectrum(clip=mvf.Depth(clip,8), grid=grid)
    
    nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)
    
    upscaled_clips: List[vs.VideoNode] = []
    rescaled_clips: List[vs.VideoNode] = []
    detail_masks: List[vs.VideoNode] = []
    params_list: List[dict] = []
    
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

        upscaled = getattr(core.resize, _get_resize_name(kernel_name))(
            descaled,
            width=clip.width,
            height=clip.height,
            src_left=dargs['src_left'],
            src_top=dargs['src_top'],
            src_width=dargs['src_width'],
            src_height=dargs['src_height'],
            **extra_params.get('rparams', {})
        )

        n2x = nnedi3(nnedi3(descaled, dh=True), dw=True)
        
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
    
    rescaled = _select(reference=src_luma, upscaled_clips=upscaled_clips, candidate_clips=rescaled_clips, params_list=params_list, common_mask_clip=common_mask_clip)
    detail_mask = _select(reference=src_luma, upscaled_clips=upscaled_clips, candidate_clips=detail_masks, params_list=params_list, common_mask_clip=common_mask_clip)

    if use_detail_mask:
        rescaled = core.std.MaskedMerge(rescaled, src_luma, detail_mask)

    final = _mergeuv(rescaled, clip) if clip.format.color_family == vs.YUV else rescaled
    
    format_string = (
        "\nMinIndex: {MinIndex}\n"
        "MinDiff: {MinDiff}\n"
        "Descaled: {Descaled}\n"
        f"Threshold_Max: {threshold_max}\n"
        f"Threshold_Min: {threshold_min}\n\n"
    )

    format_string += (
        "|    i   |          Diff          |Kernel| SrcHeight |    Bw     |     Bh    | B      | C      | Taps   |\n"
        "|--------|------------------------|------|-----------|-----------|-----------|--------|--------|--------|\n"
    )

    for i in range(len(upscaled_clips)):
        format_string += (
            f"| {i:04}   | {{Diff{i}}}  | {{Kernel{i}}}  | {{SrcHeight{i}}}   | "
            f"{{Bw{i}}}|{{Bh{i}}}|"
            f"{{B{i}}}   | {{C{i}}}   | {{Taps{i}}}   |\n"
        )
    osd_clip = core.akarin.Text(final, format_string)

    if show_fft:
        src_fft = _fft(clip)
        rescaled_fft = _fft(final)
        
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



def ranger(start, end, step):
    if step == 0:
        raise ValueError("ranger: step must not be 0!")
    return [round(start + i * step, 10) for i in range(int((end - start) / step))]


def screen_shot(clip: vs.VideoNode, frames: Union[List[int], int], path: str, file_name: str, overwrite: bool):
    from pathlib import Path

    if isinstance(frames, int):
        frames = [frames]
        
    clip = clip.resize.Spline36(format=vs.RGB24)
    try: 
        clip = core.akarin.PickFrames(clip, indices=frames)
    except AttributeError:
        try:
            clip = core.pickframes.PickFrames(clip, indices=frames)
        except AttributeError:
            
            # modified from https://github.com/AkarinVS/vapoursynth-plugin/issues/26#issuecomment-1951230729
            def PickFrames(clip: vs.VideoNode, indices: List[int]) -> vs.VideoNode:
                new = clip.std.BlankClip(length=len(indices))
                return new.std.FrameEval(lambda n: clip[indices[n]], None, clip) # type: ignore

            clip = PickFrames(clip=clip, indices=frames)
    
    
    output_path = Path(path).resolve()
    
    for i, frame in enumerate(clip.frames()):
        tmp = clip.std.Trim(first=i, last=i).fpng.Write(filename=(output_path / (file_name%frames[i])).with_suffix('.png'), overwrite=overwrite, compression=2)
        for f in tmp.frames():
            pass

# inspired by https://skyeysnow.com/forum.php?mod=viewthread&tid=41492
def tweak_rgb(clip: vs.VideoNode, delta_r: float, delta_g: float, delta_b: float) -> vs.VideoNode:
    
    shY = 0.2126 * delta_r + 0.7152 * delta_g + 0.0722 * delta_b

    new_r = delta_r - shY
    new_g = delta_g - shY
    new_b = delta_b - shY

    r_expr = f"x {new_r} +"
    g_expr = f"x {new_g} +"
    b_expr = f"x {new_b} +"

    r, g, b = vsutil.split(clip)

    r_adj = core.akarin.Expr(r, r_expr)
    g_adj = core.akarin.Expr(g, g_expr)
    b_adj = core.akarin.Expr(b, b_expr)

    return core.std.ShufflePlanes([r_adj, g_adj, b_adj], planes=[0, 0, 0], colorfamily=vs.RGB)

# inspired by mvf.postfix2infix
def postfix2infix(expr: str):
    import re
    # Preprocessing
    expr = expr.strip()
    expr = re.sub(r'\[\s*(\w+)\s*,\s*(\w+)\s*\]', r'[\1,\2]', expr) # [x, y] => [x,y]
    tokens = re.split(r'\s+', expr)
    
    stack = []
    output_lines = []

    # Regex patterns for numbers
    number_pattern = re.compile(
        r'^('
        r'0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?'
        r'|'
        r'0[0-7]*'
        r'|'
        r'[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)'
        r')$'
    )

    def pop(n=1):
        if n == 1:
            return stack.pop()
        r = stack[-n:]
        del stack[-n:]
        return r

    def push(item):
        stack.append(item)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Skip empty tokens
        if not token:
            i += 1
            continue

        # Single letter
        if token.isalpha() and len(token) == 1:
            push(token)
            i += 1
            continue

        # Numbers
        if number_pattern.match(token):
            push(token)
            i += 1
            continue

        # Source clips (srcN)
        if re.match(r'^src\d+$', token):
            push(token)
            i += 1
            continue

        # Frame property
        if re.match(r'^[a-zA-Z]\w*\.[a-zA-Z]\w*$', token):
            push(token)
            i += 1
            continue

        # Dynamic pixel access
        if token.endswith('[]'):
            clip_identifier = token[:-2]
            absY = pop()
            absX = pop()
            push(f"{clip_identifier}[{absX}, {absY}](dynamic)")
            i += 1
            continue

        # Static relative pixel access
        m = re.match(r'^([a-zA-Z]\w*)(\[\-?\d+\,\-?\d+\])(\:\w)?$', token)
        if m:
            clip_identifier = m.group(1)
            coord_part = m.group(2)
            boundary_suffix = m.group(3)
            boundary_type = "clamped" if not boundary_suffix or boundary_suffix == ":c" else "mirrored"
            push(f"{clip_identifier}{coord_part} ({boundary_type})(static)")
            i += 1
            continue

        # Variable operations
        var_store_match = re.match(r'^([a-zA-Z]\w*)\!$', token)
        var_load_match = re.match(r'^([a-zA-Z]\w*)\@$', token)
        if var_store_match:
            var_name = var_store_match.group(1)
            val = pop()
            output_lines.append(f"{var_name} = {val}")
            i += 1
            continue
        elif var_load_match:
            var_name = var_load_match.group(1)
            push(var_name)
            i += 1
            continue

        # Drop operations
        drop_match = re.match(r'^drop(\d*)$', token)
        if drop_match:
            num = int(drop_match.group(1)) if drop_match.group(1) else 1
            pop(num)
            i += 1
            continue

        # Sort operations
        sort_match = re.match(r'^sort(\d+)$', token)
        if sort_match:
            num = int(sort_match.group(1))
            items = pop(num)
            sorted_items_expr = f"sort({', '.join(items)})"
            for idx in range(len(items)):
                push(f"{sorted_items_expr}[{idx}]")
            i += 1
            continue

        # Duplicate operations
        dup_match = re.match(r'^dup(\d*)$', token)
        if dup_match:
            n = int(dup_match.group(1)) if dup_match.group(1) else 0
            if len(stack) <= n:
                output_lines.append(f"# [Error] dup{n} needs at least index={n}")
            else:
                push(stack[-1 - n])
            i += 1
            continue

        # Swap operations
        swap_match = re.match(r'^swap(\d*)$', token)
        if swap_match:
            n = int(swap_match.group(1)) if swap_match.group(1) else 1
            if len(stack) <= n:
                output_lines.append(f"# [Error] swap{n} needs at least index={n}")
            else:
                stack[-1], stack[-1 - n] = stack[-1 - n], stack[-1]
            i += 1
            continue

        # Special constants
        if token in ('N', 'X', 'Y', 'width', 'height'):
            constants = {
                'N': 'current_frame_number',
                'X': 'current_x',
                'Y': 'current_y',
                'width': 'current_width',
                'height': 'current_height'
            }
            push(constants[token])
            i += 1
            continue

        # Unary functions
        if token in ('sin', 'cos', 'round', 'trunc', 'floor', 'bitnot', 'abs', 'not'):
            a = pop()
            if token == 'not':
                push(f"(!{a})")
            else:
                push(f"{token}({a})")
            i += 1
            continue

        # Binary and special functions
        if token in ('%', '**', 'pow', 'clip', 'clamp', 'bitand', 'bitor', 'bitxor'):
            b = pop()
            a = pop()
            if token == '%':
                push(f"({a} % {b})")
            elif token in ('**', 'pow'):
                push(f"pow({a}, {b})")
            elif token in ('clip', 'clamp'):
                c = pop()
                push(f"clamp({c}, {a}, {b})")
            elif token == 'bitand':
                push(f"({a} & {b})")
            elif token == 'bitor':
                push(f"({a} | {b})")
            elif token == 'bitxor':
                push(f"({a} ^ {b})")
            i += 1
            continue

        # Basic arithmetic, comparison and logical operators
        if token in ('+', '-', '*', '/', 'max', 'min', '>', '<', '>=', '<=', '!=', '==', 'and', 'or', 'xor'):
            b = pop()
            a = pop()
            if token in ('max', 'min'):
                push(f"{token}({a}, {b})")
            elif token == 'and':
                push(f"({a} && {b})")
            elif token == 'or':
                push(f"({a} || {b})")
            elif token == 'xor':
                # (a || b) && !(a && b)
                # or (a && !b) || (!a && b)
                push(f"(({a} && !{b}) || (!{a} && {b}))")
            else:
                push(f"({a} {token} {b})")
            i += 1
            continue

        # Ternary operator
        if token == '?':
            false_val = pop()
            true_val = pop()
            cond = pop()
            push(f"({cond} ? {true_val} : {false_val})")
            i += 1
            continue

        # Unknown tokens
        output_lines.append(f"# [Unknown token]: {token}  (Push as-is)")
        push(token)
        i += 1

    # Handle remaining stack items
    if len(stack) == 1:
        output_lines.append(f"RESULT = {stack[0]}")
    else:
        for idx, item in enumerate(stack):
            if idx == len(stack) - 1:
                output_lines.append(f"RESULT = {item}")
            else:
                output_lines.append(f"# stack[{idx}]: {item}")

    ret = '\n'.join(output_lines)
    print(ret)
    


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

# copied from https://github.com/Artoriuz/glsl-chroma-from-luma-prediction/blob/main/CfL_Prediction.glsl
cfl_shader = R'''
//!PARAM chroma_offset_x
//!TYPE float
0.0

//!PARAM chroma_offset_y
//!TYPE float
0.0

//!HOOK CHROMA
//!BIND LUMA
//!BIND CHROMA
//!SAVE LUMA_LR
//!WIDTH CHROMA.w
//!HEIGHT LUMA.h
//!WHEN CHROMA.w LUMA.w <
//!DESC Chroma From Luma Prediction (Hermite 1st step, Downscaling Luma)

float comp_wd(vec2 v) {
    float x = min(length(v), 1.0);
    return smoothstep(0.0, 1.0, 1.0 - x);
}

vec4 hook() {
    vec2 luma_pos = LUMA_pos;
    luma_pos.x += chroma_offset_x / LUMA_size.x;
    float start  = ceil((luma_pos.x - (1.0 / CHROMA_size.x)) * LUMA_size.x - 0.5);
    float end = floor((luma_pos.x + (1.0 / CHROMA_size.x)) * LUMA_size.x - 0.5);

    float wt = 0.0;
    float luma_sum = 0.0;
    vec2 pos = luma_pos;

    for (float dx = start.x; dx <= end.x; dx++) {
        pos.x = LUMA_pt.x * (dx + 0.5);
        vec2 dist = (pos - luma_pos) * CHROMA_size;
        float wd = comp_wd(dist);
        float luma_pix = LUMA_tex(pos).x;
        luma_sum += wd * luma_pix;
        wt += wd;
    }

    vec4 output_pix = vec4(luma_sum /= wt, 0.0, 0.0, 1.0);
    return clamp(output_pix, 0.0, 1.0);
}

//!HOOK CHROMA
//!BIND LUMA_LR
//!BIND CHROMA
//!BIND LUMA
//!SAVE LUMA_LR
//!WIDTH CHROMA.w
//!HEIGHT CHROMA.h
//!WHEN CHROMA.w LUMA.w <
//!DESC Chroma From Luma Prediction (Hermite 2nd step, Downscaling Luma)

float comp_wd(vec2 v) {
    float x = min(length(v), 1.0);
    return smoothstep(0.0, 1.0, 1.0 - x);
}

vec4 hook() {
    vec2 luma_pos = LUMA_LR_pos;
    luma_pos.y += chroma_offset_y / LUMA_LR_size.y;
    float start  = ceil((luma_pos.y - (1.0 / CHROMA_size.y)) * LUMA_LR_size.y - 0.5);
    float end = floor((luma_pos.y + (1.0 / CHROMA_size.y)) * LUMA_LR_size.y - 0.5);

    float wt = 0.0;
    float luma_sum = 0.0;
    vec2 pos = luma_pos;

    for (float dy = start; dy <= end; dy++) {
        pos.y = LUMA_LR_pt.y * (dy + 0.5);
        vec2 dist = (pos - luma_pos) * CHROMA_size;
        float wd = comp_wd(dist);
        float luma_pix = LUMA_LR_tex(pos).x;
        luma_sum += wd * luma_pix;
        wt += wd;
    }

    vec4 output_pix = vec4(luma_sum /= wt, 0.0, 0.0, 1.0);
    return clamp(output_pix, 0.0, 1.0);
}

//!HOOK CHROMA
//!BIND HOOKED
//!BIND LUMA
//!BIND LUMA_LR
//!WHEN CHROMA.w LUMA.w <
//!WIDTH LUMA.w
//!HEIGHT LUMA.h
//!OFFSET ALIGN
//!DESC Chroma From Luma Prediction (Upscaling Chroma)

#define USE_12_TAP_REGRESSION 1
#define USE_8_TAP_REGRESSIONS 1
#define DEBUG 0

float comp_wd(vec2 v) {
    float d = min(length(v), 2.0);
    float d2 = d * d;
    float d3 = d2 * d;

    if (d < 1.0) {
        return 1.25 * d3 - 2.25 * d2 + 1.0;
    } else {
        return -0.75 * d3 + 3.75 * d2 - 6.0 * d + 3.0;
    }
}

vec4 hook() {
    float ar_strength = 0.8;
    vec2 mix_coeff = vec2(0.8);
    vec2 corr_exponent = vec2(4.0);

    vec4 output_pix = vec4(0.0, 0.0, 0.0, 1.0);
    float luma_zero = LUMA_texOff(0.0).x;

    vec2 pp = HOOKED_pos * HOOKED_size - vec2(0.5);
    vec2 fp = floor(pp);
    pp -= fp;

#ifdef HOOKED_gather
    vec2 quad_idx[4] = {{0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}, {2.0, 2.0}};

    vec4 luma_quads[4];
    vec4 chroma_quads[4][2];

    for (int i = 0; i < 4; i++) {
        luma_quads[i] = LUMA_LR_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 0);
        chroma_quads[i][0] = HOOKED_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 0);
        chroma_quads[i][1] = HOOKED_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 1);
    }

    vec2 chroma_pixels[16];
    chroma_pixels[0]  = vec2(chroma_quads[0][0].w, chroma_quads[0][1].w);
    chroma_pixels[1]  = vec2(chroma_quads[0][0].z, chroma_quads[0][1].z);
    chroma_pixels[2]  = vec2(chroma_quads[1][0].w, chroma_quads[1][1].w);
    chroma_pixels[3]  = vec2(chroma_quads[1][0].z, chroma_quads[1][1].z);
    chroma_pixels[4]  = vec2(chroma_quads[0][0].x, chroma_quads[0][1].x);
    chroma_pixels[5]  = vec2(chroma_quads[0][0].y, chroma_quads[0][1].y);
    chroma_pixels[6]  = vec2(chroma_quads[1][0].x, chroma_quads[1][1].x);
    chroma_pixels[7]  = vec2(chroma_quads[1][0].y, chroma_quads[1][1].y);
    chroma_pixels[8]  = vec2(chroma_quads[2][0].w, chroma_quads[2][1].w);
    chroma_pixels[9]  = vec2(chroma_quads[2][0].z, chroma_quads[2][1].z);
    chroma_pixels[10] = vec2(chroma_quads[3][0].w, chroma_quads[3][1].w);
    chroma_pixels[11] = vec2(chroma_quads[3][0].z, chroma_quads[3][1].z);
    chroma_pixels[12] = vec2(chroma_quads[2][0].x, chroma_quads[2][1].x);
    chroma_pixels[13] = vec2(chroma_quads[2][0].y, chroma_quads[2][1].y);
    chroma_pixels[14] = vec2(chroma_quads[3][0].x, chroma_quads[3][1].x);
    chroma_pixels[15] = vec2(chroma_quads[3][0].y, chroma_quads[3][1].y);

    float luma_pixels[16];
    luma_pixels[0]  = luma_quads[0].w;
    luma_pixels[1]  = luma_quads[0].z;
    luma_pixels[2]  = luma_quads[1].w;
    luma_pixels[3]  = luma_quads[1].z;
    luma_pixels[4]  = luma_quads[0].x;
    luma_pixels[5]  = luma_quads[0].y;
    luma_pixels[6]  = luma_quads[1].x;
    luma_pixels[7]  = luma_quads[1].y;
    luma_pixels[8]  = luma_quads[2].w;
    luma_pixels[9]  = luma_quads[2].z;
    luma_pixels[10] = luma_quads[3].w;
    luma_pixels[11] = luma_quads[3].z;
    luma_pixels[12] = luma_quads[2].x;
    luma_pixels[13] = luma_quads[2].y;
    luma_pixels[14] = luma_quads[3].x;
    luma_pixels[15] = luma_quads[3].y;
#else
    vec2 pix_idx[16] = {{-0.5,-0.5}, {0.5,-0.5}, {1.5,-0.5}, {2.5,-0.5},
                        {-0.5, 0.5}, {0.5, 0.5}, {1.5, 0.5}, {2.5, 0.5},
                        {-0.5, 1.5}, {0.5, 1.5}, {1.5, 1.5}, {2.5, 1.5},
                        {-0.5, 2.5}, {0.5, 2.5}, {1.5, 2.5}, {2.5, 2.5}};

    float luma_pixels[16];
    vec2 chroma_pixels[16];

    for (int i = 0; i < 16; i++) {
        luma_pixels[i] = LUMA_LR_tex(vec2((fp + pix_idx[i]) * HOOKED_pt)).x;
        chroma_pixels[i] = HOOKED_tex(vec2((fp + pix_idx[i]) * HOOKED_pt)).xy;
    }
#endif

#if (DEBUG == 1)
    vec2 chroma_spatial = vec2(0.5);
    mix_coeff = vec2(1.0);
#else
    float wd[16];
    float wt = 0.0;
    vec2 ct = vec2(0.0);

    vec2 chroma_min = min(min(min(chroma_pixels[5], chroma_pixels[6]), chroma_pixels[9]), chroma_pixels[10]);
    vec2 chroma_max = max(max(max(chroma_pixels[5], chroma_pixels[6]), chroma_pixels[9]), chroma_pixels[10]);

    const int dx[16] = {-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2};
    const int dy[16] = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

    for (int i = 0; i < 16; i++) {
        wd[i] = comp_wd(vec2(dx[i], dy[i]) - pp);
        wt += wd[i];
        ct += wd[i] * chroma_pixels[i];
    }

    vec2 chroma_spatial = ct / wt;
    chroma_spatial = clamp(mix(chroma_spatial, clamp(chroma_spatial, chroma_min, chroma_max), ar_strength), 0.0, 1.0);
#endif

#if (USE_12_TAP_REGRESSION == 1 || USE_8_TAP_REGRESSIONS == 1)
    const int i12[12] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14};
    const int i4y[4] = {1, 2, 13, 14};
    const int i4x[4] = {4, 7, 8, 11};
    const int i4[4] = {5, 6, 9, 10};

    float luma_sum_4 = 0.0;
    float luma_sum_4y = 0.0;
    float luma_sum_4x = 0.0;
    vec2 chroma_sum_4 = vec2(0.0);
    vec2 chroma_sum_4y = vec2(0.0);
    vec2 chroma_sum_4x = vec2(0.0);

    for (int i = 0; i < 4; i++) {
        luma_sum_4 += luma_pixels[i4[i]];
        luma_sum_4y += luma_pixels[i4y[i]];
        luma_sum_4x += luma_pixels[i4x[i]];
        chroma_sum_4 += chroma_pixels[i4[i]];
        chroma_sum_4y += chroma_pixels[i4y[i]];
        chroma_sum_4x += chroma_pixels[i4x[i]];
    }

    float luma_avg_12 = (luma_sum_4 + luma_sum_4y + luma_sum_4x) / 12.0;
    float luma_var_12 = 0.0;
    vec2 chroma_avg_12 = (chroma_sum_4 + chroma_sum_4y + chroma_sum_4x) / 12.0;
    vec2 chroma_var_12 = vec2(0.0);
    vec2 luma_chroma_cov_12 = vec2(0.0);

    for (int i = 0; i < 12; i++) {
        luma_var_12 += pow(luma_pixels[i12[i]] - luma_avg_12, 2.0);
        chroma_var_12 += pow(chroma_pixels[i12[i]] - chroma_avg_12, vec2(2.0));
        luma_chroma_cov_12 += (luma_pixels[i12[i]] - luma_avg_12) * (chroma_pixels[i12[i]] - chroma_avg_12);
    }

    vec2 corr = clamp(abs(luma_chroma_cov_12 / max(sqrt(luma_var_12 * chroma_var_12), 1e-6)), 0.0, 1.0);
    mix_coeff = pow(corr, corr_exponent) * mix_coeff;
#endif

#if (USE_12_TAP_REGRESSION == 1)
    vec2 alpha_12 = luma_chroma_cov_12 / max(luma_var_12, 1e-6);
    vec2 beta_12 = chroma_avg_12 - alpha_12 * luma_avg_12;
    vec2 chroma_pred_12 = clamp(alpha_12 * luma_zero + beta_12, 0.0, 1.0);
#endif

#if (USE_8_TAP_REGRESSIONS == 1)
    const int i8y[8] = {1, 2, 5, 6, 9, 10, 13, 14};
    const int i8x[8] = {4, 5, 6, 7, 8, 9, 10, 11};

    float luma_avg_8y = (luma_sum_4 + luma_sum_4y) / 8.0;
    float luma_avg_8x = (luma_sum_4 + luma_sum_4x) / 8.0;
    float luma_var_8y = 0.0;
    float luma_var_8x = 0.0;
    vec2 chroma_avg_8y = (chroma_sum_4 + chroma_sum_4y) / 8.0;
    vec2 chroma_avg_8x = (chroma_sum_4 + chroma_sum_4x) / 8.0;
    vec2 luma_chroma_cov_8y = vec2(0.0);
    vec2 luma_chroma_cov_8x = vec2(0.0);

    for (int i = 0; i < 8; i++) {
        luma_var_8y += pow(luma_pixels[i8y[i]] - luma_avg_8y, 2.0);
        luma_var_8x += pow(luma_pixels[i8x[i]] - luma_avg_8x, 2.0);
        luma_chroma_cov_8y += (luma_pixels[i8y[i]] - luma_avg_8y) * (chroma_pixels[i8y[i]] - chroma_avg_8y);
        luma_chroma_cov_8x += (luma_pixels[i8x[i]] - luma_avg_8x) * (chroma_pixels[i8x[i]] - chroma_avg_8x);
    }

    vec2 alpha_8y = luma_chroma_cov_8y / max(luma_var_8y, 1e-6);
    vec2 alpha_8x = luma_chroma_cov_8x / max(luma_var_8x, 1e-6);
    vec2 beta_8y = chroma_avg_8y - alpha_8y * luma_avg_8y;
    vec2 beta_8x = chroma_avg_8x - alpha_8x * luma_avg_8x;
    vec2 chroma_pred_8y = clamp(alpha_8y * luma_zero + beta_8y, 0.0, 1.0);
    vec2 chroma_pred_8x = clamp(alpha_8x * luma_zero + beta_8x, 0.0, 1.0);
    vec2 chroma_pred_8 = mix(chroma_pred_8y, chroma_pred_8x, 0.5);
#endif

#if (USE_12_TAP_REGRESSION == 1 && USE_8_TAP_REGRESSIONS == 1)
    output_pix.xy = mix(chroma_spatial, mix(chroma_pred_12, chroma_pred_8, 0.5), mix_coeff);
#elif (USE_12_TAP_REGRESSION == 1 && USE_8_TAP_REGRESSIONS == 0)
    output_pix.xy = mix(chroma_spatial, chroma_pred_12, mix_coeff);
#elif (USE_12_TAP_REGRESSION == 0 && USE_8_TAP_REGRESSIONS == 1)
    output_pix.xy = mix(chroma_spatial, chroma_pred_8, mix_coeff);
#else
    output_pix.xy = chroma_spatial;
#endif

    output_pix.xy = clamp(output_pix.xy, 0.0, 1.0);
    return output_pix;
}

//!PARAM distance_coeff
//!TYPE float
//!MINIMUM 0.0
2.0

//!PARAM intensity_coeff
//!TYPE float
//!MINIMUM 0.0
128.0

//!HOOK CHROMA
//!BIND CHROMA
//!BIND LUMA
//!DESC Chroma From Luma Prediction (Smoothing Chroma)

float comp_w(vec2 spatial_distance, float intensity_distance) {
    return max(100.0 * exp(-distance_coeff * pow(length(spatial_distance), 2.0) - intensity_coeff * pow(intensity_distance, 2.0)), 1e-32);
}

vec4 hook() {
    vec4 output_pix = vec4(0.0, 0.0, 0.0, 1.0);
    float luma_zero = LUMA_texOff(0).x;
    float wt = 0.0;
    vec2 ct = vec2(0.0);

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            vec2 chroma_pixels = CHROMA_texOff(vec2(i, j)).xy;
            float luma_pixels = LUMA_texOff(vec2(i, j)).x;
            float w = comp_w(vec2(i, j), luma_zero - luma_pixels);
            wt += w;
            ct += w * chroma_pixels;
        }
    }

    output_pix.xy = clamp(ct / wt, 0.0, 1.0);
    return output_pix;
}
'''
