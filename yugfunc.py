import inspect
import vapoursynth as vs
from vapoursynth import core

_output_index = 1
debug = 1

def enable_debug():
    global debug
    debug = 1

def disable_debug():
    global debug
    debug = 0

def reset_output_index(index=1):
    global _output_index
    _output_index = index

def output(*args):
    def _get_variable_name(frame, clip):
        for var_name, var_val in frame.f_locals.items():
            if var_val is clip:
                return var_name
        return None

    def _add_text(clip, text):
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(f"_add_text expected a VideoNode, but got {type(clip)}")
        return core.text.Text(clip, text)
    
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
            if debug and index != 0:
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
            if debug and _output_index != 0:
                variable_name = _get_variable_name(frame, clip)
                if variable_name:
                    clip = _add_text(clip, f"{variable_name}")
                else:
                    clip = _add_text(clip, "Unknown Variable")
            clip.set_output(_output_index)
            used_indices.add(_output_index)
            _output_index += 1

def auto_descale_nnedi3(clip, possible_heights, loss_threshold=1):
    """
    :param clip: 输入的clip
    :param possible_heights: 可能的原生宽度列表
    :param loss_threshold: 判断loss显著增加的阈值
    """
    
    best_params = [None]
    best_loss = [float('inf')]
    prev_params = [None]

    kernel_list = [
        {'name': 'Debicubic',  'dparams': {'b': 1/3,  'c': 1/3},     'rparams': {'filter_param_a': 1/3,  'filter_param_b': 1/3}},
        {'name': 'Debicubic',  'dparams': {'b': 0,    'c': 0.5},     'rparams': {'filter_param_a': 0,    'filter_param_b': 0.5}},
        {'name': 'Debicubic',  'dparams': {'b': 0,    'c': 0},       'rparams': {'filter_param_a': 0,    'filter_param_b': 0}},
        {'name': 'Debicubic',  'dparams': {'b': -0.5, 'c': 0.25},    'rparams': {'filter_param_a': -0.5, 'filter_param_b': 0.25}},
        {'name': 'Debicubic',  'dparams': {'b': -1,   'c': 0},       'rparams': {'filter_param_a': -1,   'filter_param_b': 0}},
        {'name': 'Debicubic',  'dparams': {'b': -2,   'c': 1},       'rparams': {'filter_param_a': -2,   'filter_param_b': 1}},

        {'name': 'Delanczos',  'dparams': {'taps': 2},               'rparams': {'filter_param_a': 2}},
        {'name': 'Delanczos',  'dparams': {'taps': 3},               'rparams': {'filter_param_a': 3}},
        {'name': 'Delanczos',  'dparams': {'taps': 4},               'rparams': {'filter_param_a': 4}},
        {'name': 'Delanczos',  'dparams': {'taps': 5},               'rparams': {'filter_param_a': 5}},
        {'name': 'Delanczos',  'dparams': {'taps': 6},               'rparams': {'filter_param_a': 6}},
        {'name': 'Delanczos',  'dparams': {'taps': 8},               'rparams': {'filter_param_a': 8}},

        # Spline kernels always get the lowest loss, even thry are clearly not the correct ones
        # {'name': 'Despline16', 'dparams': {}, 'rparams': {}},
        # {'name': 'Despline36', 'dparams': {}, 'rparams': {}},
        # {'name': 'Despline64', 'dparams': {}, 'rparams': {}},

        {'name': 'Debilinear', 'dparams': {}, 'rparams': {}},
    ]

    def iterate(clip, func, count):
        for _ in range(count):
            clip = func(clip)
        return clip
    
    def get_resize_name(descale_name):
        if descale_name.startswith('De'):
            return descale_name[2:].capitalize()
        return descale_name

    def generate_descale_mask(source, upscaled, threshold=0.05):
        mask = core.akarin.Expr([source, upscaled], 'x y - abs').std.Binarize(threshold)
        mask = iterate(mask, core.std.Maximum, 2)
        mask = iterate(mask, core.std.Inflate, 2)
        return mask

    def process_frame(n, f):
        nonlocal best_params, best_loss, prev_params

        if VERBOSE_LOGGING:
            logger.info(f"Processing frame {n}")

        frame = f.copy()

        src_luma = core.std.ShufflePlanes([clip], planes=0, colorfamily=vs.GRAY)
        src_luma_frame = src_luma.get_frame(n)
        src_luma_arr = np.asarray(src_luma_frame[0]).astype(np.float32)

        if prev_params[0]:
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: Using previous parameters")
            params = prev_params[0]
            descaled = getattr(core.descale, params['kernel'])(
                src_luma, 
                width=params.get('width', 0), 
                height=params.get('height', 0), 
                src_left=params.get('src_left', 0),
                src_top=params.get('src_top', 0),
                src_width=params.get('src_width', src_luma.width),
                src_height=params.get('src_height', src_luma.height),
                **params.get('dkernel_params', {})
            )
            resize_name = get_resize_name(params['kernel'])
            upscaled = getattr(core.resize, resize_name)(
                descaled, 
                width=clip.width, 
                height=clip.height,
                src_left=params.get('src_left', 0),
                src_top=params.get('src_top', 0),
                src_width=params.get('src_width', src_luma.width),
                src_height=params.get('src_height', src_luma.height),
                **params.get('rkernel_params', {})
            )
            upscaled_frame = upscaled.get_frame(n)
            upscaled_arr = np.asarray(upscaled_frame[0]).astype(np.float32)
            diff = src_luma_arr - upscaled_arr
            current_loss = np.mean(np.abs(diff))
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: current_loss = {current_loss}")
            if current_loss < best_loss[0] * (1 + loss_threshold):
                best_params[0] = params
                best_loss[0] = current_loss # type: ignore
            else:
                best_params[0] = None
                best_loss[0] = float('inf')

        if best_params[0] is None:
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: Performing full scan")
            
            for kernel_info in kernel_list:
                kernel = kernel_info['name']
                dkernel_params = kernel_info['dparams']
                rkernel_params = kernel_info['rparams']
                for src_height in possible_heights:
                    for base_height in [clip.height, clip.height-1]:
                        for base_width in [clip.width, clip.width-1]:
                            dargs = descale_cropping_args(src_luma, src_height, base_height, base_width)
                            if VERBOSE_LOGGING:
                                logger.info(f"Frame {n}: trying dargs = {dargs}")
                            descaled = getattr(core.descale, kernel)(
                                src_luma,
                                **dargs,
                                **dkernel_params
                            )
                            resize_name = get_resize_name(kernel)
                            upscaled = getattr(core.resize, resize_name)(
                                descaled, 
                                width=clip.width, 
                                height=clip.height,
                                src_left=dargs['src_left'],
                                src_top=dargs['src_top'],
                                src_width=dargs['src_width'],
                                src_height=dargs['src_height'],
                                **rkernel_params
                            )
                            upscaled_frame = upscaled.get_frame(n)
                            upscaled_arr = np.asarray(upscaled_frame[0]).astype(np.float32)
                            diff = src_luma_arr - upscaled_arr
                            loss = np.mean(np.abs(diff))
                            if loss < best_loss[0]:
                                best_loss[0] = loss # type: ignore
                                best_params[0] = { # type: ignore
                                    'kernel': kernel,
                                    'dkernel_params': dkernel_params,
                                    'rkernel_params': rkernel_params,
                                    **dargs
                                }

        if best_params[0] is None:
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: No good parameters found, returning original frame")
            return frame

        params = best_params[0]
        if VERBOSE_LOGGING:
            logger.info(f"Frame {n}: Best parameters - {params}")
        descaled = getattr(core.descale, params['kernel'])(
            src_luma, 
            width=params.get('width', 0), 
            height=params.get('height', 0), 
            src_left=params.get('src_left', 0),
            src_top=params.get('src_top', 0),
            src_width=params.get('src_width', src_luma.width),
            src_height=params.get('src_height', src_luma.height),
            **params.get('dkernel_params', {})
        )
        rescaled = nnedi3_resample.nnedi3_resample(
            descaled, clip.width, clip.height, mode="nnedi3cl", qual=2, nns=4,
                src_left=params.get('src_left', 0),
                src_top=params.get('src_top', 0),
                src_width=params.get('src_width', src_luma.width),
                src_height=params.get('src_height', src_luma.height)
        )
        
        resize_name = get_resize_name(params['kernel'])
        upscaled = getattr(core.resize, resize_name)(
            descaled, 
            width=clip.width, 
            height=clip.height,
            src_left=params.get('src_left', 0),
            src_top=params.get('src_top', 0),
            src_width=params.get('src_width', src_luma.width),
            src_height=params.get('src_height', src_luma.height),
            **params.get('rkernel_params', {})
        )
        mask = generate_descale_mask(src_luma, upscaled)
        luma = core.std.MaskedMerge(rescaled, src_luma, mask)
        luma_frame = luma.get_frame(n)
        luma_arr = np.asanyarray(luma_frame[0]).astype(np.float32)
        
        np.copyto(np.asarray(frame[0]), luma_arr)

        prev_params[0] = best_params[0] # type: ignore

        return frame

    result = clip.std.ModifyFrame(clips=clip, selector=process_frame)
    
    return result
