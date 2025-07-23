from vstools import vs
from typing import Union, Sequence, IO
from subprocess import Popen
import sys

# TODO: add a function to get encoder params
# TODO: add dovi encoding


# copied from https://skyeysnow.com/forum.php?mod=viewthread&tid=38690
def _y4m_header(clip: vs.VideoNode) -> str:
    y4mformat = ""
    if clip.format.color_family == vs.GRAY:
        y4mformat = "mono"
        if clip.format.bits_per_sample > 8:
            y4mformat = y4mformat + str(clip.format.bits_per_sample)
    else:  # YUV
        if clip.format.subsampling_w == 1 and clip.format.subsampling_h == 1:
            y4mformat = "420"
        elif clip.format.subsampling_w == 1 and clip.format.subsampling_h == 0:
            y4mformat = "422"
        elif clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0:
            y4mformat = "444"
        elif clip.format.subsampling_w == 2 and clip.format.subsampling_h == 2:
            y4mformat = "410"
        elif clip.format.subsampling_w == 2 and clip.format.subsampling_h == 0:
            y4mformat = "411"
        elif clip.format.subsampling_w == 0 and clip.format.subsampling_h == 1:
            y4mformat = "440"

        if clip.format.bits_per_sample > 8:
            y4mformat = y4mformat + "p" + str(clip.format.bits_per_sample)

    y4mformat = "C" + y4mformat + " "
    data = "YUV4MPEG2 {y4mformat}W{width} H{height} F{fps_num}:{fps_den} Ip A0:0 XLENGTH={length}\n".format(
        y4mformat=y4mformat,
        width=clip.width,
        height=clip.height,
        fps_num=clip.fps_num,
        fps_den=clip.fps_den,
        length=len(clip),
    )

    return data


# copied from https://skyeysnow.com/forum.php?mod=viewthread&tid=38690
def _MIMO(clips: Sequence[vs.VideoNode], files: Sequence[IO]) -> None:
    """Multiple-Input-Multiple-Output"""

    # Checks
    num_clips = len(clips)
    num_files = len(files)
    assert num_clips > 0 and num_clips == num_files

    for clip in clips:
        assert clip.format

    is_y4m = [clip.format.color_family in (vs.YUV, vs.GRAY) for clip in clips]

    buffer = [False] * num_files
    for n in range(num_files):
        fileobj = files[n]
        if (fileobj is sys.stdout or fileobj is sys.stderr) and hasattr(
            fileobj, "buffer"
        ):
            buffer[n] = True

    # Interleave
    max_len = max(len(clip) for clip in clips)
    clips_aligned: list[vs.VideoNode] = []
    for clip in clips:
        if len(clip) < max_len:
            clip_aligned = clip + vs.core.std.BlankClip(
                clip, length=max_len - len(clip)
            )
        else:
            clip_aligned = clip
        clips_aligned.append(vs.core.std.Interleave([clip_aligned] * num_clips))
    clips_varfmt = vs.core.std.BlankClip(
        length=max_len * num_clips, varformat=True, varsize=True
    )
    interleaved = vs.core.std.FrameEval(
        clips_varfmt, lambda n, f: clips_aligned[n % num_clips], clips_aligned, clips
    )

    # Y4M header
    for n in range(num_clips):
        if is_y4m[n]:
            clip = clips[n]
            fileobj = files[n].buffer if buffer[n] else files[n]  # type: ignore
            data = _y4m_header(clip)
            fileobj.write(data.encode("ascii"))

    # Output
    for idx, frame in enumerate(interleaved.frames(close=True)):
        n = idx % num_clips
        clip = clips[n]
        fileobj = files[n].buffer if buffer[n] else files[n]  # type: ignore
        finished = idx // num_clips
        if finished < len(clip):
            if is_y4m[n]:
                fileobj.write(b"FRAME\n")
            for planeno, plane in enumerate(frame):  # type: ignore
                if (
                    frame.get_stride(planeno)  # type: ignore
                    != plane.shape[1] * clip.format.bytes_per_sample
                ):  # type: ignore
                    fileobj.write(bytes(plane))
                else:
                    fileobj.write(plane)
            if hasattr(fileobj, "flush"):
                fileobj.flush()


def encode_video(
    clip: Union[vs.VideoNode, list[Union[vs.VideoNode, tuple[vs.VideoNode, int]]]],
    encoder: Union[list[Popen], Popen, IO, None] = None,
    multi: bool = False,
) -> None:
    """
    Encode one or multiple VapourSynth video nodes using external encoders or output directly to stdout.

    Args:
        clip: A VapourSynth video node or a list of video nodes/tuples to encode.
        encoder: External encoder process(es) created with subprocess.Popen, a file-like object, or None to output to stdout.
        multi: If True, handle multiple input clips and multiple encoders. If False, handle a single clip.

    Examples:

        ```python

        # Output to an external encoder
        encoder = subprocess.Popen(['x264', '--demuxer', 'y4m', '-', '-o', 'output.mp4'], stdin=subprocess.PIPE)
        encode_video(clip, encoder)

        # Output directly to stdout (like vspipe)
        encode_video(clip, None)
        # or
        encode_video(clip, sys.stdout)

        # Output to a file
        with open('output.y4m', 'wb') as f:
            encode_video(clip, f)

        # Example with multiple encoders
        encoders = [
            subprocess.Popen(['x264', '--demuxer', 'y4m', '-', '-o', 'output1.mp4'], stdin=subprocess.PIPE),
            subprocess.Popen(['x264', '--demuxer', 'y4m', '-', '-o', 'output2.mp4'], stdin=subprocess.PIPE)
        ]
        encode_video([clip1, clip2], encoders, multi=True)

        ```
    """

    if not multi:
        output_clip = None

        if isinstance(clip, vs.VideoNode):
            output_clip = clip
        elif isinstance(clip, list):
            for item in clip:
                if isinstance(item, tuple):
                    clip, index = item
                    if isinstance(clip, vs.VideoNode) and isinstance(index, int):
                        if index == 0:
                            output_clip = clip
                    else:
                        raise TypeError("encode_video: Tuple must be (VideoNode, int)")
            if not output_clip:
                output_clip = clip[0] if isinstance(clip[0], vs.VideoNode) else None
            if not output_clip:
                raise ValueError("encode_video: Couldn't parse clip!")

        assert isinstance(output_clip, vs.VideoNode)
        assert (
            output_clip.format.color_family == vs.YUV
        ), "encode_video: All clips must be YUV color family"
        assert output_clip.fps != 0, "encode_video: all clips must be CFR"

        # Allow direct output to stdout when encoder is None
        if encoder is None:
            _MIMO([output_clip], [sys.stdout])
        elif hasattr(encoder, "write") and callable(encoder.write):  # type: ignore # type: ignore # File-like object
            _MIMO([output_clip], [encoder])  # type: ignore
        else:  # Subprocess.Popen object
            _MIMO([output_clip], [encoder.stdin])  # type: ignore
            encoder.communicate()  # type: ignore
            encoder.wait()  # type: ignore
    else:
        assert isinstance(
            encoder, list
        ), "encode_video: encoder must be a list when multi=True"
        assert isinstance(
            clip, list
        ), "encode_video: clip must be a list when multi=True"
        assert len(encoder) == len(
            clip
        ), "encode_video: encoder and clip must have the same length"
        assert all(
            isinstance(item, vs.VideoNode) for item in clip
        ), "encode_video: all items in clip must be VideoNodes"
        assert all(
            clip.format.color_family == vs.YUV for clip in clip  # type: ignore
        ), "encode_video: all clips must be YUV color family"
        assert all(
            clip.fps != 0 for clip in clip  # type: ignore
        ), "encode_video: all clips must be CFR"

        output_clips = []
        stdins = []
        for i, clip in enumerate(clip):  # type: ignore
            output_clips.append(clip)
            if hasattr(encoder[i], "write") and callable(encoder[i].write):  # type: ignore # File-like object
                stdins.append(encoder[i])
            else:  # Subprocess.Popen object
                stdins.append(encoder[i].stdin)

        _MIMO(output_clips, stdins)  # type: ignore

        # Only call communicate and wait for Popen objects
        for i, enc in enumerate(encoder):
            if not hasattr(enc, "write") or not callable(enc.write):  # type: ignore # Not a file-like object
                enc.communicate()
                enc.wait()
