from vapoursynth import core
import vapoursynth as vs
from math import atan2
from y5gfunc.expr import infix2postfix, postfix2infix, math_functions


def calc_atan2_expr(y, x):
    expr = infix2postfix(
        postfix2infix(infix2postfix(math_functions + "\nRESULT = atan2($src0, $src1)")),
        force_std=True,
    )
    akarin = get_pixel_value(
        core.akarin.Expr(
            [
                core.std.BlankClip(format=vs.GRAYS, color=y),
                core.std.BlankClip(format=vs.GRAYS, color=x),
            ],
            expr,
        )
    )
    std = get_pixel_value(
        core.std.Expr(
            [
                core.std.BlankClip(format=vs.GRAYS, color=y),
                core.std.BlankClip(format=vs.GRAYS, color=x),
            ],
            expr,
        )
    )
    py = atan2(y, x)
    max_diff = max(max(abs(akarin - std), abs(akarin - py)), abs(std - py))
    if_pass = max_diff < 1e-5
    print(
        f"atan2({y}, {x}) => Akarin: {akarin}, Std: {std}, Python: {py}; max_diff: {max_diff} {'Pass' if if_pass else 'Fail'}"
    )
    return if_pass, max_diff


def get_pixel_value(clip):
    frame = clip.get_frame(0)
    arr = frame[0]
    return arr[0, 0]


def test_atan2():
    # Test with various y, x pairs
    success = fail = total = max_diff = 0
    test_cases = [
        (0, 1),  # atan2(0, 1) = 0
        (1, 0),  # atan2(1, 0) = π/2
        (0, -1),  # atan2(0, -1) = π
        (-1, 0),  # atan2(-1, 0) = -π/2
        (1, 1),  # atan2(1, 1) = π/4
        (-1, -1),  # atan2(-1, -1) = -3π/4
        (1, -1),  # atan2(1, -1) = 3π/4
        (-1, 1),  # atan2(-1, 1) = -π/4
    ]

    for y, x in test_cases:
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    import random

    for _ in range(50):
        y = random.uniform(-1e8, 1e8)
        x = random.uniform(-1e8, 1e8)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    for _ in range(100):
        y = random.uniform(-2, 2)
        x = 0
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    for _ in range(100):
        y = 0
        x = random.uniform(-2, 2)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    for _ in range(100):
        y = random.uniform(-2, 2)
        x = random.uniform(-2, 2)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    for _ in range(50):
        y = random.uniform(-50, 50)
        x = random.uniform(-50, 50)
        is_success, diff = calc_atan2_expr(y, x)
        if is_success:
            success += 1
        else:
            fail += 1
        total += 1
        max_diff = max(max_diff, diff)

    print(f"Max difference: {max_diff}")
    print(f"Success: {success}, Fail: {fail}, Total: {total}")


test_atan2()

print("All tests completed.")
