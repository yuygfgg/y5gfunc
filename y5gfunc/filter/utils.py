from vstools import scale_value, get_peak_value, ColorRange
from functools import partial

scale_value_full = partial(scale_value, range_in=ColorRange.FULL, range_out=ColorRange.FULL)
get_peak_value_full = partial(get_peak_value, range_in=ColorRange.FULL)