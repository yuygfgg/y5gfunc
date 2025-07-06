from .convert_akarin_to_std import (
    replace_drop_in_expr,
    expand_rpn_sort,
    convert_var_expr,
    replace_clip_names,
)
from .verify import verify_std_expr


def to_std_expr(expr: str) -> str:
    """
    Convert an infix expression to a std.Expr expression.
    """
    # FIXME: convert math functions (trunc / round / floor / sin / cos / fmod) (possible?)
    return replace_drop_in_expr(replace_clip_names(convert_var_expr(expand_rpn_sort(expr))))


__all__ = [
    "replace_drop_in_expr",
    "expand_rpn_sort",
    "convert_var_expr",
    "to_std_expr",
    "verify_std_expr",
    "replace_clip_names",
]
