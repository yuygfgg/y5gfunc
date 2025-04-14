from vstools import core
from vstools import vs
from vstools import get_neutral_value


# modified from vsTaambk
def temporal_stabilize(
    processed_clip: vs.VideoNode,
    ref_clip: vs.VideoNode,
    delta: int = 3,
    pel: int = 1,
    retain: float = 0.6,
) -> vs.VideoNode:
    """
    Temporally stabilizes a processed video clip using motion compensation based on a reference clip.

    This function aims to reduce temporal inconsistencies (like flickering or unsteadiness) in the `processed_clip` by referencing a `reference_clip`.
    It calculates the difference between `reference_clip` and `processed_clip`, stabilizes this difference using motion vectors derived from `processed_clip`,
    and then applies the stabilized difference back onto `reference_clip`.

    A common use case is when `processed_clip` is the result of filtering applied to `reference_clip`, and that filtering
    introduced temporal instability. This function attempts to apply a temporally consistent version of the filter's effect.

    The core stabilization logic uses motion vector analysis on `processed_clip` and motion compensation on the difference clip.
    A limiting step ensures the stabilized difference doesn't deviate too drastically from the original difference, controlled by the `retain` parameter.

    Args:
        processed_clip: The input video clip that likely contains temporal instabilities.
        ref_clip: The reference video clip,  the original source before the processing that introduced instability.
        delta: Temporal radius for temporal stabilize (1-6). This value corresponds to the `mv.DegrainN` function used.
        pel: Subpixel accuracy for motion estimation passed to `mv.Super`. Higher values increase accuracy but also computational cost.
        retain: Weight for merging the limited stabilized difference with the full stabilized difference. A value of 1.0 means only the full
            stabilized difference is used, while 0.0 means only the limited difference (closer to the original difference) is used. It controls
            the strength of the stabilization effect versus preserving the original difference characteristics. Range [0.0, 1.0].

    Returns:
        A new video clip, ideally resembling `processed_clip` but with reduced temporal instability.

    Raises:
        ValueError: If `processed_clip` and `reference_clip` have different video formats.
        ValueError: if `pel` is not an ingeter in [1, 2, 4].
        ValueError: If `delta` is not an integer between 1 and 6 (inclusive).
    """
    if processed_clip.format != ref_clip.format:
        raise ValueError(
            "temporal_stabilize: format of processed_clip and ref_clip mismatch."
        )
    if pel not in [1, 2, 4]:
        raise ValueError("temporal_stabilize: pel (1, 2, 4) invalid.")
    if delta not in [1, 2, 3, 4, 5, 6]:
        raise ValueError("temporal_stabilize: delta (1~6) invalid.")

    diff = core.std.MakeDiff(ref_clip, processed_clip)
    clip_super = core.mv.Super(processed_clip, pel=pel)
    diff_super = core.mv.Super(diff, pel=pel, levels=1)

    backward_vectors = [
        core.mv.Analyse(clip_super, isb=True, delta=i + 1, overlap=8, blksize=16)
        for i in range(delta)
    ]
    forward_vectors = [
        core.mv.Analyse(clip_super, isb=False, delta=i + 1, overlap=8, blksize=16)
        for i in range(delta)
    ]
    vectors = [
        vector
        for vector_group in zip(backward_vectors, forward_vectors)
        for vector in vector_group
    ]

    stabilize_func = getattr(core.mv, f"Degrain{delta}")
    diff_stabilized = stabilize_func[delta](diff, diff_super, *vectors)

    diff_stabilized_limited = core.akarin.Expr(
        [diff, diff_stabilized],
        f"x {get_neutral_value(processed_clip)} - abs y {get_neutral_value(processed_clip)} - abs < x y ?",
    )
    diff_stabilized = core.std.Merge(
        diff_stabilized_limited, diff_stabilized, weight=retain
    )
    clip_stabilized = core.std.MakeDiff(ref_clip, diff_stabilized)
    return clip_stabilized
