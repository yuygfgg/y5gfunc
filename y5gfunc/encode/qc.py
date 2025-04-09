from typing import Union, Optional, Literal
from vstools import vs
from vstools import core
import vstools
from ..utils import PickFrames

def encode_check(
    encoded: vs.VideoNode,
    source: Optional[vs.VideoNode] = None,
    mode: Literal["BOTH", "SSIM", "CAMBI"] = "BOTH",
    threshold_cambi: float = 5,
    threshold_ssim: float = 0.9,
    return_type: Literal["encoded", "error", "both"] = "encoded"
) -> Union[
    vs.VideoNode,
    tuple[vs.VideoNode, vs.VideoNode]
]:
    
    from muvsfunc import SSIM
    
    assert 0 <= threshold_cambi <= 24
    assert 0 <= threshold_ssim <= 1
    assert mode in ["BOTH", "SSIM", "CAMBI"]
    assert return_type in ['encoded', 'error', 'both']

    if mode == "BOTH":
        enable_ssim = enable_cambi = True
    elif mode == "SSIM":
        enable_ssim = True
        enable_cambi = False
    else:
        enable_ssim = False
        enable_cambi = True
    
    if enable_ssim:
        assert source
        assert encoded.format.id == source.format.id
        ssim = SSIM(encoded, source)
            
    if enable_cambi:
        cambi = core.cambi.Cambi(encoded if vstools.get_depth(encoded) <= 10 else vstools.depth(encoded, 10, dither_type="none"), prop='CAMBI')
    
    error_frames = []
    def _chk(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        def print_red_bold(text) -> None:
            print("\033[1;31m" + text + "\033[0m")
            
        fout = f[0].copy()
        
        ssim_err = cambi_err = False
        
        if enable_ssim: 
            fout.props['PlaneSSIM'] = ssim_val = f[2].props['PlaneSSIM']
            fout.props['ssim_err'] = ssim_err = (1 if threshold_ssim > f[2].props['PlaneSSIM'] else 0) # type: ignore
        
        if enable_cambi: 
            fout.props['CAMBI'] = cambi_val = f[1].props['CAMBI'] 
            fout.props['cambi_err'] = cambi_err = (1 if threshold_cambi < f[1].props['CAMBI'] else 0) # type: ignore
        
        if cambi_err and enable_cambi:
            print_red_bold (f"frame {n}: Banding detected! CAMBI: {cambi_val}"
                            f"    Note: banding threshold is {threshold_cambi}")
        if ssim_err and enable_ssim:
            print_red_bold (f"frame {n}: Distortion detected! SSIM: {ssim_val}"
                            f"    Note: distortion threshold is {threshold_ssim}")
        if not (cambi_err or ssim_err):
            print(f"Frame {n}: OK!")
        else:
            error_frames.append(n)

        return fout

    if enable_ssim and enable_cambi:
        output = core.std.ModifyFrame(encoded, [encoded, cambi, ssim], _chk)
    elif enable_cambi:
        output = core.std.ModifyFrame(encoded, [encoded, cambi, cambi], _chk)
    else:
        output = core.std.ModifyFrame(encoded, [encoded, ssim, ssim], _chk)
    
    if return_type == "encoded": 
        return output
    
    for _ in output.frames():
        pass
    
    err = PickFrames(encoded, error_frames)

    if return_type == "both":
        return output, err
    else:
        return err

