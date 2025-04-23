import platform
from vstools import get_peak_value, ColorRange
from functools import partial

get_peak_value_full = partial(get_peak_value, range_in=ColorRange.FULL)


def is_optimized_cpu() -> bool:
    machine = platform.machine()

    if "arm64" or "x86_64" in machine.lower():
        return True

    return False
