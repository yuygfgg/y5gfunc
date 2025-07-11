from .convert_akarin_to_std import (
    convert_drop,
    convert_sort,
    convert_var,
    convert_clip_names,
    convert_pow,
    convert_clip_clamp,
)
from .verify import verify_std_expr


def to_std_expr(expr: str) -> str:
    """
    Convert an infix expression to a std.Expr expression.
    """
    # FIXME: convert math functions (trunc / round / floor / sin / cos / fmod) (possible?)
    return convert_drop(
        convert_clip_names(
            convert_var(convert_sort(convert_pow(convert_clip_clamp(expr))))
        )
    )


__all__ = [
    "convert_drop",
    "convert_sort",
    "convert_var",
    "to_std_expr",
    "verify_std_expr",
    "convert_clip_names",
    "convert_pow",
    "convert_clip_clamp",
]
