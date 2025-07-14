import regex as re

from vstools import vs
from typing import Union, Optional

_UNARY_OPS = {
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "abs",
    "not",
    "bitnot",
    "trunc",
    "round",
    "floor",
}

_BINARY_OPS = {
    "+",
    "-",
    "*",
    "/",
    "max",
    "min",
    "pow",
    "**",
    ">",
    "<",
    "=",
    ">=",
    "<=",
    "and",
    "or",
    "xor",
    "%",
    "bitand",
    "bitor",
    "bitxor",
}

_CONSTANTS = {
    "N",
    "X",
    "Y",
    "width",
    "height",
    "pi",
}

_TERNARY_OPS = {"?"}
_CLIP_OPS = {"clip", "clamp"}

_TOKEN_PATTERN = re.compile(r"(\w+)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?::(?:c|m))?")
_SPLIT_PATTERN = re.compile(r"\s+")
_NUMBER_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"^0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?$",  # Hexadecimal
        r"^0[0-7]+$",  # Octal
        r"^[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)$",  # Decimal and scientific notation
    ]
]
_HEX_PATTERN = re.compile(r"^0x")
_HEX_PARTS_PATTERN = re.compile(
    r"^(0x[0-9A-Fa-f]+)(?:\.([0-9A-Fa-f]+))?(?:p([+\-]?\d+))?$"
)
_OCTAL_PATTERN = re.compile(r"^0[0-7]")
_DROP_PATTERN = re.compile(r"^drop([1-9]\d*)?$")


def is_token_numeric(token: str) -> bool:
    """Check if a token string represents a numeric constant."""
    if token.isdigit() or (
        token.startswith("-") and len(token) > 1 and token[1:].isdigit()
    ):
        return True

    for pattern in _NUMBER_PATTERNS:
        if pattern.match(token):
            return True
    return False


def tokenize_expr(expr: str) -> list[str]:
    """Convert expression string to a list of tokens"""
    expr = expr.strip()
    if not expr:
        return []

    placeholders = {}
    placeholder_prefix = "__PXACCESS"
    placeholder_suffix = "__"
    count = 0

    def repl(matchobj):
        nonlocal count
        key = f"{placeholder_prefix}{count}{placeholder_suffix}"
        placeholders[key] = matchobj.group(0)
        count += 1
        return key

    expr_with_placeholders = _TOKEN_PATTERN.sub(repl, expr)

    raw_tokens = _SPLIT_PATTERN.split(expr_with_placeholders)

    tokens = []
    for token in raw_tokens:
        if token in placeholders:
            tokens.append(placeholders[token])
        elif token:
            tokens.append(token)

    return tokens


def get_stack_effect(tk: str) -> int:
    """Return net stack delta for a token."""
    if (
        is_token_numeric(tk)
        or _TOKEN_PATTERN.match(tk)
        or tk in _CONSTANTS
        or (tk.startswith("dup") and not (tk.endswith("!") or tk.endswith("@")))
    ):
        return 1
    if tk in _UNARY_OPS or tk.startswith(("swap", "sort")):
        return 0
    if tk in _BINARY_OPS:
        return -1
    if tk in _TERNARY_OPS or tk in _CLIP_OPS:
        return -2
    if tk.endswith("[]") and len(tk) > 2 and not _TOKEN_PATTERN.match(tk):
        return -1
    if (
        tk.endswith("!")
        and len(tk) > 1
        and not tk.startswith("[")
        and not _TOKEN_PATTERN.match(tk)
    ):
        return -1
    if (
        tk.endswith("@")
        and len(tk) > 1
        and not tk.startswith("[")
        and not _TOKEN_PATTERN.match(tk)
    ):
        return 1

    if tk.startswith("drop"):
        m = _DROP_PATTERN.fullmatch(tk)
        if m:
            n_str = m.group(1)
            return -(int(n_str) if n_str else 1)

    return 0


def get_op_arity(token: str) -> int:
    """Return number of operands for a token."""
    if token in _UNARY_OPS:
        return 1
    if token in _BINARY_OPS:
        return 2
    if token in _TERNARY_OPS or token in _CLIP_OPS:
        return 3
    if token.endswith("[]") and len(token) > 2 and not _TOKEN_PATTERN.match(token):
        return 2
    if (
        token.endswith("!")
        and len(token) > 1
        and not token.startswith("[")
        and not _TOKEN_PATTERN.match(token)
    ):
        return 1

    for prefix in ("dup", "drop", "sort", "swap"):
        if token.startswith(prefix):
            pattern = re.compile(rf"{prefix}(\d*)")
            m = pattern.fullmatch(token)
            if m:
                n_str = m.group(1)

                if prefix == "dup":
                    # dup is dup0. Arity is N+1.
                    n = int(n_str) if n_str != "" else 0
                    return n + 1

                if prefix == "swap":
                    # swap is swap1. Arity is N+1.
                    n = int(n_str) if n_str != "" else 1
                    if n < 1:
                        raise ValueError(
                            f"Invalid swap operator: {token}, N must be >= 1"
                        )
                    return n + 1

                if prefix in ("drop", "sort"):
                    # drop is drop1, sort is sort1. Arity is N.
                    n = int(n_str) if n_str != "" else 1
                    if n < 1:
                        raise ValueError(
                            f"Invalid {prefix} operator: {token}, N must be >= 1"
                        )
                    return n
    return 0


# modified from jvsfunc.ex_planes()
def ex_planes(
    clip: vs.VideoNode, expr: list[str], planes: Optional[Union[int, list[int]]] = None
) -> list[str]:
    if planes:
        plane_range = range(clip.format.num_planes)
        planes = [planes] if isinstance(planes, int) else planes
        expr = [expr[0] if i in planes else "" for i in plane_range]
    return expr
