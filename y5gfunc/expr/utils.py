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

_TERNARY_OPS = {"?"}
_CLIP_OPS = {"clip", "clamp"}

token_pattern = re.compile(r"(\w+)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?::(?:c|m))?")
split_pattern = re.compile(r"\s+")
number_patterns = [
    re.compile(pattern)
    for pattern in [
        r"^0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?$",  # Hexadecimal
        r"^0[0-7]+$",  # Octal
        r"^[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)$",  # Decimal and scientific notation
    ]
]
hex_pattern = re.compile(r"^0x")
hex_parts_pattern = re.compile(
    r"^(0x[0-9A-Fa-f]+)(?:\.([0-9A-Fa-f]+))?(?:p([+\-]?\d+))?$"
)
octal_pattern = re.compile(r"^0[0-7]")

def is_token_numeric(token: str) -> bool:
    """Check if a token string represents a numeric constant."""
    if token.isdigit() or (
        token.startswith("-") and len(token) > 1 and token[1:].isdigit()
    ):
        return True

    for pattern in number_patterns:
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

    expr_with_placeholders = token_pattern.sub(repl, expr)

    raw_tokens = split_pattern.split(expr_with_placeholders)

    tokens = []
    for token in raw_tokens:
        if token in placeholders:
            tokens.append(placeholders[token])
        elif token:
            tokens.append(token)

    return tokens

# modified from jvsfunc.ex_planes()
def ex_planes(
    clip: vs.VideoNode, expr: list[str], planes: Optional[Union[int, list[int]]] = None
) -> list[str]:
    if planes:
        plane_range = range(clip.format.num_planes)
        planes = [planes] if isinstance(planes, int) else planes
        expr = [expr[0] if i in planes else "" for i in plane_range]
    return expr
