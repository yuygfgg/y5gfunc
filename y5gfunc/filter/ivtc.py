from vstools import vs
from vstools import core
from pathlib import Path
import time
from typing import Union, Optional
from ..utils import resolve_path

# modified from https://github.com/DJATOM/VapourSynth-atomchtools/blob/34e16238291954206b3f7d5b704324dd6885b224/atomchtools.py#L370
def TIVTC_VFR(
    source: vs.VideoNode,
    clip2: Optional[vs.VideoNode] = None,
    tfmIn: Union[Path, str] = "matches.txt",
    tdecIn: Union[Path, str] = "metrics.txt",
    mkvOut: Union[Path, str] = "timecodes.txt",
    tfm_args: dict = dict(),
    tdecimate_args: dict = dict(),
    overwrite: bool = False
) -> vs.VideoNode:
    '''
    Convenient wrapper on tivtc to perform automatic vfr decimation with one function.
    '''

    analyze = True

    assert isinstance(tfmIn, (str, Path))
    assert isinstance(tdecIn, (str, Path))
    assert isinstance(mkvOut, (str, Path))
    
    tfmIn = resolve_path(tfmIn)
    tdecIn = resolve_path(tdecIn)
    mkvOut = resolve_path(mkvOut)

    if tfmIn.exists() and tdecIn.exists():
        analyze = False

    if clip2 and not overwrite:
        tfm_args.update(dict(clip2=clip2))

    if analyze:
        tfmIn = resolve_path(tfmIn)
        tdecIn = resolve_path(tdecIn)
        mkvOut = resolve_path(mkvOut)
        tfm_pass1_args = tfm_args.copy()
        tdecimate_pass1_args = tdecimate_args.copy()
        tfm_pass1_args.update(dict(output=str(tfmIn)))
        tdecimate_pass1_args.update(dict(output=str(tdecIn), mode=4))
        tmpnode = core.tivtc.TFM(source, **tfm_pass1_args)
        tmpnode = core.tivtc.TDecimate(tmpnode, **tdecimate_pass1_args)

        for i, _ in enumerate(tmpnode.frames()):
            print(f"Analyzing frame #{i}...", end='\r')

        del tmpnode
        time.sleep(0.5) # let it write logs

    tfm_args.update(dict(input=str(tfmIn)))
    tdecimate_args.update(dict(input=str(tdecIn), tfmIn=str(tfmIn), mkvOut=str(mkvOut), mode=5, hybrid=2, vfrDec=1))

    output = core.tivtc.TFM(source, **tfm_args)
    output = core.tivtc.TDecimate(output,  **tdecimate_args)

    return output