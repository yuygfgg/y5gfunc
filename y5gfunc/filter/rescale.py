from vstools import vs
from vstools import core
import vstools
from typing import Literal, Union, Optional
from enum import StrEnum
from math import floor
import functools
from .mask import generate_detail_mask
from .resample import nn2x
from itertools import product
from .resample import SSIM_downsample


class DescaleMode(StrEnum):
    W = "w"
    H = "h"
    WH = "wh"
    HW = "hw"


# modified from getfnative.descale_cropping_args()
def descale_cropping_args(
    clip: vs.VideoNode,
    src_height: float,
    base_height: int,
    base_width: int,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    mode: DescaleMode = DescaleMode.WH,
) -> dict[str, Union[int, float]]:
    effective_src_h = clip.height + crop_top + crop_bottom
    if effective_src_h <= 0:
        raise ValueError(
            "descale_cropping_args: Effective source height (clip.height + crop_top + crop_bottom) must be positive."
        )

    ratio = src_height / effective_src_h
    effective_src_w = clip.width + crop_left + crop_right
    if effective_src_w <= 0:
        raise ValueError(
            "descale_cropping_args: Effective source width (clip.width + crop_left + crop_right) must be positive."
        )

    src_width = ratio * effective_src_w

    cropped_src_width = ratio * clip.width
    cropped_src_height = ratio * clip.height

    margin_left = (base_width - src_width) / 2 + ratio * crop_left
    margin_right = (base_width - src_width) / 2 + ratio * crop_right
    margin_top = (base_height - src_height) / 2 + ratio * crop_top
    margin_bottom = (base_height - src_height) / 2 + ratio * crop_bottom

    cropped_width = max(0, base_width - floor(margin_left) - floor(margin_right))
    cropped_height = max(0, base_height - floor(margin_top) - floor(margin_bottom))

    cropped_src_left = margin_left - floor(margin_left)
    cropped_src_top = margin_top - floor(margin_top)

    args: dict[str, Union[int, float]] = dict(width=clip.width, height=clip.height)

    args_w: dict[str, Union[int, float]] = dict(
        width=cropped_width, src_width=cropped_src_width, src_left=cropped_src_left
    )

    args_h: dict[str, Union[int, float]] = dict(
        height=cropped_height, src_height=cropped_src_height, src_top=cropped_src_top
    )

    if mode == DescaleMode.WH:
        args.update(args_w)
        args.update(args_h)
    elif mode == DescaleMode.HW:
        args.update(args_h)
        args.update(args_w)
    elif mode == DescaleMode.W:
        args.update(args_w)
    elif mode == DescaleMode.H:
        args.update(args_h)

    return args


# TODO: use vs-jetpack Rescalers
# inspired by https://skyeysnow.com/forum.php?mod=viewthread&tid=58390
def rescale(
    clip: vs.VideoNode,
    descale_kernel: Union[
        Literal[
            "Debicubic",
            "Delanczos",
            "Despline16",
            "Despline36",
            "Despline64",
            "Debilinear",
            "Depoint",
        ],
        list[str],
    ] = "Debicubic",
    src_height: Union[Union[float, int], list[Union[float, int]]] = 720,
    bw: Optional[Union[int, list[int]]] = None,
    bh: Optional[Union[int, list[int]]] = None,
    descale_mode: DescaleMode = DescaleMode.WH,
    show_upscaled: bool = False,
    show_fft: bool = False,
    detail_mask_threshold: float = 0.05,
    use_detail_mask: bool = True,
    show_detail_mask: bool = False,
    exclude_common_mask: bool = True,
    show_common_mask: bool = False,
    nnedi3_args: dict[str, int] = {"field": 1, "nsize": 4, "nns": 4, "qual": 2},
    taps: Union[int, list[int]] = 4,
    b: Union[Union[float, int], list[Union[float, int]]] = 0.33,
    c: Union[Union[float, int], list[Union[float, int]]] = 0.33,
    threshold_max: float = 0.007,
    threshold_min: float = -1,
    show_osd: bool = True,
    ex_thr: Union[float, int] = 0.015,
    norm_order: int = 1,
    crop_size: int = 5,
    scene_stable: bool = False,
    scene_descale_threshold_ratio: float = 0.5,
    scenecut_threshold: Union[float, int] = 0.1,
    opencl: bool = True,
) -> Union[
    vs.VideoNode,
    tuple[vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
    ],
    tuple[
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
    ],
    tuple[vs.VideoNode, list[vs.VideoNode]],
    tuple[vs.VideoNode, list[vs.VideoNode], vs.VideoNode],
    tuple[vs.VideoNode, list[vs.VideoNode], vs.VideoNode, vs.VideoNode],
    tuple[vs.VideoNode, list[vs.VideoNode], vs.VideoNode, vs.VideoNode, vs.VideoNode],
    tuple[
        vs.VideoNode,
        list[vs.VideoNode],
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
    ],
    tuple[
        vs.VideoNode,
        list[vs.VideoNode],
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
    ],
    tuple[
        vs.VideoNode,
        list[vs.VideoNode],
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
        vs.VideoNode,
    ],
]:
    """
    Automatically descale and rescale

    This function attempts to find the 'native' resolution of a video clip that may have been upscaled during production. It tests various
    descaling parameters by descaling the input clip, re-upscaling it with the *same* kernel, and comparing the result to the original clip luma.

    The parameter set yielding the minimum difference (within thresholds) is considered the 'best' guess for the original upscale parameters. The clip
    descaled with these parameters is then re-upscaled to the original dimensions using a potentially higher-quality method (NNEDI3 + SSIM_downsample).

    A detail mask can be generated and used to merge the rescaled result with the original clip, preserving details that might exist only in the original
    resolution (like credits). Scene-based decision logic can be enabled via `scene_stable`.

    Args:
        clip: Input video node. Expected to be in YUV format for proper handling.
        descale_kernel: Descale kernel(s) to test (e.g., "Debicubic", "Delanczos").
        src_height: Potential native source height(s) to test.
        bw: Base width for descaling calculations.
        bh: Base height for descaling calculations.
        descale_mode: Order of applying width/height descaling.
        show_upscaled: If True, return the list of intermediate same-kernel upscaled clips used for comparison.
        show_fft: If True, return FFT spectrums of the original and final clip.
        detail_mask_threshold: Threshold for generating the detail mask. Lower values detect more 'detail'.
            Useful to mask original resolution contents.
        use_detail_mask: If True, merge the final rescaled clip with the original using the detail mask.
        show_detail_mask: If True, return the final selected detail mask.
        exclude_common_mask: If True, use the common mask to exclude potential native details from the difference calculation.
            Useful when descaling scenes with native resolution contents.
        show_common_mask: If True, return the mask generated by combining all candidate detail masks (used for error calculation exclusion).
        nnedi3_args: Arguments passed to the NNEDI3 function for the high-quality upscale.
        taps: Taps parameter(s) for Delanczos/Despline kernels.
        b: Bicubic 'b' parameter(s) for Debicubic kernel.
        c: Bicubic 'c' parameter(s) for Debicubic kernel.
        threshold_max: Maximum difference metric allowed for a frame to be considered 'descaled' (use large value to disable).
        threshold_min: Minimum difference metric required for a frame to be considered 'descaled' (use negative value to disable).
        show_osd: If True, return a clip with On-Screen Display showing frame stats.
        ex_thr: Exclusion threshold for difference calculation; differences below this are ignored.
        norm_order: Order of the norm used for calculating frame difference (1 = MAE).
        crop_size: Pixels to crop from each border before calculating difference.
        scene_stable: If True, use scene-based logic to choose parameters for entire scenes rather than frame-by-frame.
        scene_descale_threshold_ratio: Minimum ratio of frames in a scene that must be successfully descaled for the scene to use the best descaled parameters.
        scenecut_threshold: Threshold for scene change detection (used if scene_stable=True).
        opencl: If True, attempt to use OpenCL versions of NNEDI3 if available.

    Returns:
        A tuple containing the final rescaled video node as the first element. The exact tuple structure varies depending on which flags are enabled.

    ## Tips:
        To rescale from multiple native resolution, use this function for every possible src_height, then choose the largest MaxDelta one.

        For example, to descale a show with mixed native resolution 714.75P and 956P:
        ```python
        rescaled1, osd1 = rescale(clip=src, src_height=ranger(714.5, 715, 0.025)+[713, 714, 716, 717])
        rescaled2, osd2 = rescale(clip=src, src_height=ranger(955, 957,0.1)+[953, 954, 958])

        select_expr = "src0.MaxDelta src0.Descaled * src1.MaxDelta src1.Descaled * argmax2"

        osd = core.akarin.Select([osd1, osd2], [rescaled1, rescaled2], select_expr)
        rescaled = core.akarin.Select([rescaled1, rescaled2], [rescaled1, rescaled2], select_expr)
        ```
    """

    KERNEL_MAP = {
        "Debicubic": 1,
        "Delanczos": 2,
        "Debilinear": 3,
        "Despline16": 5,
        "Despline36": 6,
        "Despline64": 7,
    }

    descale_kernel = (
        [descale_kernel] if isinstance(descale_kernel, str) else descale_kernel
    )
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

    clip = vstools.depth(clip, 32)

    def scene_descale(
        n: int,
        f: list[vs.VideoFrame],
        cache: list[int],
        prefetch: vs.VideoNode,
        length: int,
        scene_descale_threshold_ratio: float = scene_descale_threshold_ratio,
    ) -> vs.VideoFrame:
        fout = f[0].copy()
        if n == 0 or n == prefetch.num_frames:
            fout.props["_SceneChangePrev"] = 1

        if cache[n] == -1:  # not cached
            i = n
            scene_start = n
            while i >= 0:
                frame = prefetch.get_frame(i)
                if frame.props["_SceneChangePrev"] == 1:  # scene srart
                    scene_start = i
                    break
                i -= 1
            i = scene_start
            scene_length = 0

            min_index_buffer = [0] * length
            num_descaled = 0

            while i < prefetch.num_frames:
                frame = prefetch.get_frame(i)
                min_index_buffer[frame.props["MinIndex"]] += 1  # type: ignore
                if frame.props["Descaled"]:
                    num_descaled += 1
                scene_length += 1
                i += 1
                if frame.props["_SceneChangeNext"] == 1:  # scene end
                    break

            scene_min_index = (
                max(enumerate(min_index_buffer), key=lambda x: x[1])[0]
                if num_descaled >= scene_descale_threshold_ratio * scene_length
                else length
            )

            cache[scene_start : scene_start + scene_length] = [
                scene_min_index
            ] * scene_length

        fout.props["SceneMinIndex"] = cache[n]
        return fout

    def _get_resize_name(descale_name: str) -> str:
        if descale_name.startswith("De"):
            return descale_name[2:].capitalize()
        return descale_name

    def _generate_common_mask(detail_mask_clips: list[vs.VideoNode]) -> vs.VideoNode:
        load_expr = [f"src{i} * " for i in range(len(detail_mask_clips))]
        merge_expr = " ".join(load_expr)
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
        crop_size: int = crop_size,
    ) -> vs.VideoNode:
        def _crop(clip: vs.VideoNode, crop_size: int = crop_size) -> vs.VideoNode:
            return clip.std.CropRel(*([crop_size] * 4)) if crop_size > 0 else clip

        if len(upscaled_clips) != len(candidate_clips) or len(upscaled_clips) != len(
            params_list
        ):
            raise ValueError(
                "upscaled_clips, rescaled_clips, and params_list must have the same length."
            )

        calc_diff_expr = (
            f"src0 src1 - abs dup {ex_thr} > swap {norm_order} pow 0 ? src2 - 0 1 clip"
        )

        diffs = [
            core.akarin.Expr(
                [_crop(reference), _crop(upscaled_clip), _crop(common_mask_clip)],
                calc_diff_expr,
            ).std.PlaneStats()
            for upscaled_clip in upscaled_clips
        ]

        for diff in diffs:
            diff = diff.akarin.PropExpr(
                lambda: {
                    "PlaneStatsAverage": f"x.PlaneStatsAverage {1 / norm_order} pow"
                }
            )

        load_PlaneStatsAverage_exprs = [
            f"src{i}.PlaneStatsAverage" for i in range(len(diffs))
        ]
        diff_expr = " ".join(load_PlaneStatsAverage_exprs)

        min_index_expr = diff_expr + f" argmin{len(diffs)}"
        min_diff_expr = (
            diff_expr + f" sort{len(diffs)} min_diff! drop{len(diffs)-1} min_diff@"
        )

        max_index_expr = diff_expr + f" argmax{len(diffs)}"
        max_diff_expr = diff_expr + f" sort{len(diffs)} drop{len(diffs)-1}"

        max_delta_expr = max_diff_expr + " " + min_diff_expr + " / "

        def props() -> dict[str, str]:
            d = {
                "MinIndex": min_index_expr,
                "MinDiff": min_diff_expr,
                "MaxIndex": max_index_expr,
                "MaxDiff": max_diff_expr,
                "MaxDelta": max_delta_expr,
            }
            for i in range(len(diffs)):
                d[f"Diff{i}"] = load_PlaneStatsAverage_exprs[i]
                params = params_list[i]
                d[f"KernelId{i}"] = KERNEL_MAP.get(params["Kernel"], 0)  # type: ignore
                d[f"Bw{i}"] = (params.get("BaseWidth", 0),)  # type: ignore
                d[f"Bh{i}"] = params.get("BaseHeight", 0)
                d[f"SrcHeight{i}"] = params["SrcHeight"]
                d[f"B{i}"] = params.get("B", 0)
                d[f"C{i}"] = params.get("C", 0)
                d[f"Taps{i}"] = params.get("Taps", 0)
            return d

        prop_src = core.akarin.PropExpr(diffs, props)

        # Kernel is different because it's a string.
        for i in range(len(diffs)):
            prop_src = core.akarin.Text(
                prop_src, params_list[i]["Kernel"], prop=f"Kernel{i}"
            )

        minDiff_clip = core.akarin.Select(
            clip_src=candidate_clips, prop_src=[prop_src], expr="x.MinIndex"
        )

        final_clip = core.akarin.Select(
            clip_src=[reference, minDiff_clip],
            prop_src=[prop_src],
            expr=f"x.MinDiff {threshold_max} > x.MinDiff {threshold_min} <= or 0 1 ?",
        )

        final_clip = final_clip.std.CopyFrameProps(prop_src)
        final_clip = core.akarin.PropExpr(
            [final_clip],
            lambda: {
                "Descaled": f"x.MinDiff {threshold_max} <= x.MinDiff {threshold_min} > and"
            },
        )

        return final_clip

    def _fft(clip: vs.VideoNode) -> vs.VideoNode:
        return core.fftspectrum_rs.FFTSpectrum(clip=vstools.depth(clip, 8))

    upscaled_clips: list[vs.VideoNode] = []
    rescaled_clips: list[vs.VideoNode] = []
    detail_masks: list[vs.VideoNode] = []
    params_list: list[dict] = []

    src_luma = vstools.get_y(clip)

    for kernel_name, sh, base_w, base_h, _taps, _b, _c in product(
        descale_kernel, src_height, bw, bh, taps, b, c
    ):
        extra_params: dict[str, dict[str, Union[float, int]]] = {}
        if kernel_name == "Debicubic":
            extra_params = {
                "dparams": {"b": _b, "c": _c},
                "rparams": {"filter_param_a": _b, "filter_param_b": _c},
            }
        elif kernel_name == "Delanczos":
            extra_params = {
                "dparams": {"taps": _taps},
                "rparams": {"filter_param_a": _taps},
            }
        else:
            extra_params = {}

        dargs = descale_cropping_args(
            clip=clip,
            src_height=sh,
            base_height=base_h,
            base_width=base_w,
            mode=descale_mode,
        )

        descaled = getattr(core.descale, kernel_name)(
            src_luma, **dargs, **extra_params.get("dparams", {})
        )

        upscaled = getattr(core.resize2, _get_resize_name(kernel_name))(
            descaled,
            width=clip.width,
            height=clip.height,
            src_left=dargs["src_left"],
            src_top=dargs["src_top"],
            src_width=dargs["src_width"],
            src_height=dargs["src_height"],
            **extra_params.get("rparams", {}),
        )

        n2x = nn2x(descaled, opencl=opencl, nnedi3_args=nnedi3_args)

        rescaled = SSIM_downsample(
            clip=n2x,
            width=clip.width,
            height=clip.height,
            sigmoid=False,
            src_left=dargs["src_left"] * 2 - 0.5,  # type: ignore
            src_top=dargs["src_top"] * 2 - 0.5,  # type: ignore
            src_width=dargs["src_width"] * 2,  # type: ignore
            src_height=dargs["src_height"] * 2,  # type: ignore
        )

        upscaled_clips.append(upscaled)
        rescaled_clips.append(rescaled)

        if use_detail_mask or show_detail_mask or exclude_common_mask:
            detail_mask = generate_detail_mask(
                src_luma, upscaled, detail_mask_threshold
            )
            detail_masks.append(detail_mask)

        params_list.append(
            {
                "Kernel": kernel_name,
                "SrcHeight": sh,
                "BaseWidth": base_w,
                "BaseHeight": base_h,
                "Taps": _taps,
                "B": _b,
                "C": _c,
            }
        )

    common_mask_clip = (
        _generate_common_mask(detail_mask_clips=detail_masks)
        if exclude_common_mask
        else core.std.BlankClip(clip=detail_masks[0], color=0)
    )
    if not scene_stable:
        rescaled = _select_per_frame(
            reference=src_luma,
            upscaled_clips=upscaled_clips,
            candidate_clips=rescaled_clips,
            params_list=params_list,
            common_mask_clip=common_mask_clip,
        )
        detail_mask = core.akarin.Select(
            clip_src=detail_masks, prop_src=rescaled, expr="src0.MinIndex"
        )
        detail_mask = core.akarin.Select(
            clip_src=[core.std.BlankClip(clip=detail_mask), detail_mask],
            prop_src=rescaled,
            expr="src0.Descaled",
        )
        upscaled = core.akarin.Select(
            clip_src=upscaled_clips, prop_src=rescaled, expr="src0.MinIndex"
        )
    else:
        # detail mask: matched one when descaling, otherwise blank clip
        # upscaled clip: matched one when descaling, otherwise frame level decision
        # rescaled clip: mostly-choosed index in a scene when descaling, otherwise src_luma
        # 'Descaled', 'SceneMinIndex': scene-level information
        # other props: frame-level information
        per_frame = _select_per_frame(
            reference=src_luma,
            upscaled_clips=upscaled_clips,
            candidate_clips=upscaled_clips,
            params_list=params_list,
            common_mask_clip=common_mask_clip,
        )
        upscaled_per_frame = core.akarin.Select(
            clip_src=upscaled_clips, prop_src=per_frame, expr="src0.MinIndex"
        )
        scene = core.misc.SCDetect(clip, scenecut_threshold)
        prefetch = core.std.BlankClip(clip)
        prefetch = core.akarin.PropExpr(
            [scene, per_frame],
            lambda: {
                "_SceneChangeNext": "x._SceneChangeNext",
                "_SceneChangePrev": "x._SceneChangePrev",
                "MinIndex": "y.MinIndex",
                "Descaled": "y.Descaled",
            },
        )
        cache = [-1] * clip.num_frames
        length = len(upscaled_clips)
        per_scene = core.std.ModifyFrame(
            per_frame,
            [per_frame, per_frame],
            functools.partial(
                scene_descale,
                prefetch=prefetch,
                cache=cache,
                length=length,
                scene_descale_threshold_ratio=scene_descale_threshold_ratio,
            ),
        )
        rescaled = core.akarin.Select(
            rescaled_clips + [src_luma], [per_scene], "src0.SceneMinIndex"
        )
        rescaled = core.std.CopyFrameProps(rescaled, per_scene)
        rescaled = core.akarin.PropExpr(
            [rescaled],
            lambda: {"Descaled": f"x.SceneMinIndex {len(upscaled_clips)} = not"},
        )
        detail_mask = core.akarin.Select(
            clip_src=detail_masks + [core.std.BlankClip(clip=detail_masks[0])],
            prop_src=rescaled,
            expr="src0.SceneMinIndex",
        )
        upscaled = core.akarin.Select(
            clip_src=upscaled_clips + [upscaled_per_frame],
            prop_src=rescaled,
            expr="src0.SceneMinIndex",
        )

    if use_detail_mask:
        rescaled = core.std.MaskedMerge(rescaled, src_luma, detail_mask)

    final = (
        vstools.join(rescaled, clip) if clip.format.color_family == vs.YUV else rescaled
    )

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

    for i, upscaled in enumerate(upscaled_clips):
        format_string += (
            f"| {i:04}   | {{Diff{i}}}  | {{Kernel{i}}}  | {{SrcHeight{i}}}   | "
            f"{{Bw{i}}} | {{Bh{i}}} |"
            f"{{B{i}}}   | {{C{i}}}   | {{Taps{i}}}   |\n"
        )
        upscaled_clips[i] = upscaled.text.Text(str(params_list[i]))

    osd_clip = core.akarin.Text(final, format_string)

    if show_fft:
        src_fft = _fft(clip)
        rescaled_fft = _fft(final)

    # FUCK YOU PYLINT
    if show_upscaled:
        if show_common_mask:
            if show_osd:
                if show_detail_mask and show_fft:
                    return (
                        final,
                        upscaled_clips,
                        detail_mask,
                        common_mask_clip,
                        src_fft,
                        rescaled_fft,
                        osd_clip,
                    )
                elif show_detail_mask:
                    return (
                        final,
                        upscaled_clips,
                        detail_mask,
                        common_mask_clip,
                        osd_clip,
                    )
                elif show_fft:
                    return (
                        final,
                        upscaled_clips,
                        common_mask_clip,
                        src_fft,
                        rescaled_fft,
                        osd_clip,
                    )
                else:
                    return final, upscaled_clips, common_mask_clip, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return (
                        final,
                        upscaled_clips,
                        detail_mask,
                        common_mask_clip,
                        src_fft,
                        rescaled_fft,
                    )
                elif show_detail_mask:
                    return final, upscaled_clips, detail_mask, common_mask_clip
                elif show_fft:
                    return (
                        final,
                        upscaled_clips,
                        common_mask_clip,
                        src_fft,
                        rescaled_fft,
                    )
                else:
                    return final, upscaled_clips, common_mask_clip
        else:
            if show_osd:
                if show_detail_mask and show_fft:
                    return (
                        final,
                        upscaled_clips,
                        detail_mask,
                        src_fft,
                        rescaled_fft,
                        osd_clip,
                    )
                elif show_detail_mask:
                    return final, upscaled_clips, detail_mask, osd_clip
                elif show_fft:
                    return final, upscaled_clips, src_fft, rescaled_fft, osd_clip
                else:
                    return final, upscaled_clips, osd_clip
            else:
                if show_detail_mask and show_fft:
                    return final, upscaled_clips, detail_mask, src_fft, rescaled_fft
                elif show_detail_mask:
                    return final, upscaled_clips, detail_mask
                elif show_fft:
                    return final, upscaled_clips, src_fft, rescaled_fft
                else:
                    return final, upscaled_clips
    else:
        if show_common_mask:
            if show_osd:
                if show_detail_mask and show_fft:
                    return (
                        final,
                        detail_mask,
                        common_mask_clip,
                        src_fft,
                        rescaled_fft,
                        osd_clip,
                    )
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
