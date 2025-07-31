import regex as re
from functools import lru_cache
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
_CLAMP_OPS = {"clip", "clamp"}

_REL_STATIC_PATTERN_INFIX = re.compile(
    r"(\$?\w+)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?::(?:c|m))?"
)
_REL_STATIC_PATTERN_POSTFIX = re.compile(
    r"(\w+)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?::(?:c|m))?"
)
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
_CLIP_NAME_PATTERN = re.compile(r"(?:[a-z]|src\d+)$")
_SRC_PATTERN = re.compile(r"^src\d+$")
_FRAME_PROP_PATTERN = re.compile(r"^[a-zA-Z]\w*\.[a-zA-Z]\w*$")
_STATIC_PIXEL_PATTERN = re.compile(
    r"^([a-zA-Z]\w*)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](\:\w+)?$"
)
_VAR_STORE_PATTERN = re.compile(r"^([a-zA-Z_]\w*)\!$")
_VAR_LOAD_PATTERN = re.compile(r"^([a-zA-Z_]\w*)\@$")
_DROP_PATTERN = re.compile(r"^drop(\d*)$")
_SORT_PATTERN = re.compile(r"^sort(\d+)$")
_DUP_PATTERN = re.compile(r"^dup(\d*)$")
_SWAP_PATTERN = re.compile(r"^swap(\d*)$")


@lru_cache
def is_clip_postfix(token: str) -> bool:
    """Check if a token string represents a clip."""
    return _CLIP_NAME_PATTERN.match(token) is not None


@lru_cache
def is_clip_infix(token: str) -> bool:
    """Check if a token string represents a clip."""
    return _CLIP_NAME_PATTERN.match(token.lstrip("$")) is not None and token.startswith(
        "$"
    )


@lru_cache
def is_constant_infix(token: str) -> bool:
    """
    Check if the token is a built-in constant (must start with $).
    """
    if not token.startswith("$"):
        return False

    # Remove the $ prefix and check if it's a valid constant
    constant_name = token[1:]
    constants_set = {
        "N",
        "X",
        "Y",
        "width",
        "height",
        "pi",
    }
    if constant_name in constants_set:
        return True
    if is_clip_postfix(constant_name):
        return True
    return False


@lru_cache
def is_constant_postfix(token: str) -> bool:
    """
    Check if the token is a built-in constant.
    """
    constants_set = {
        "N",
        "X",
        "Y",
        "width",
        "height",
        "pi",
    }
    if token in constants_set:
        return True
    if is_clip_postfix(token):
        return True
    return False


@lru_cache
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


@lru_cache
def parse_numeric(token: str) -> Union[int, float]:
    """Parse a numeric token string to its actual value (int or float)."""
    if not is_token_numeric(token):
        raise ValueError(
            f"parse_numeric: Token '{token}' is not a valid numeric format for parsing."
        )

    if _HEX_PATTERN.match(token):  # Hexadecimal
        if "." in token or "p" in token.lower():
            try:
                return float.fromhex(token)
            except ValueError:
                parts = _HEX_PARTS_PATTERN.match(token.lower())
                if parts:
                    integer_part = int(parts.group(1), 16)
                    fractional_part = 0
                    if parts.group(2):
                        frac_hex = parts.group(2)
                        fractional_part = int(frac_hex, 16) / (16 ** len(frac_hex))
                    exponent = 0
                    if parts.group(3):
                        exponent = int(parts.group(3))
                    return (integer_part + fractional_part) * (2**exponent)
                else:
                    raise ValueError(
                        f"parse_numeric: Could not parse complex hex token: {token}"
                    )  # Should not happen
        else:
            return int(token, 16)  # Simple hex integer
    elif (
        _OCTAL_PATTERN.match(token)
        and len(token) > 1
        and all(c in "01234567" for c in token[1:])
    ):  # Octal
        return int(token, 8)
    else:  # Decimal / Scientific / Integer
        try:
            if "." in token or "e" in token.lower():
                return float(token)
            return int(token)
        except ValueError:
            raise ValueError(
                f"parse_numeric: Internal error: Could not parse supposedly numeric token: {token}"
            )


def tokenize_expr(expr: str) -> list[str]:
    """Convert expression string to a list of tokens"""
    expr = expr.strip()
    if not expr:
        return []

    placeholders = {}
    placeholder_prefix = "__PXACCESS"
    placeholder_suffix = "__"
    count = 0

    def repl(matchobj: re.Match[str]) -> str:
        nonlocal count
        key = f"{placeholder_prefix}{count}{placeholder_suffix}"
        placeholders[key] = matchobj.group(0)
        count += 1
        return key

    expr_with_placeholders = _REL_STATIC_PATTERN_POSTFIX.sub(repl, expr)

    raw_tokens = _SPLIT_PATTERN.split(expr_with_placeholders)

    tokens = []
    for token in raw_tokens:
        if token in placeholders:
            tokens.append(placeholders[token])
        elif token:
            reconstructed = False
            if token.startswith(placeholder_prefix):
                # This token is a placeholder that may have extra characters appended,
                # like __PXACCESS0__:x. We need to find the original placeholder
                # and reconstruct the original malformed token.
                for p_key, p_val in placeholders.items():
                    if token.startswith(p_key):
                        # Reconstruct the token, e.g., 'x[1, 1]' + ':x'
                        reconstructed_token = p_val + token[len(p_key) :]
                        tokens.append(reconstructed_token)
                        reconstructed = True
                        break
            if not reconstructed:
                tokens.append(token)

    return tokens


@lru_cache
def get_stack_effect(token: str) -> int:
    """Return net stack delta for a token."""
    if (
        is_token_numeric(token)
        or _REL_STATIC_PATTERN_POSTFIX.match(token)
        or is_constant_postfix(token)
        or (
            token.startswith("dup") and not (token.endswith("!") or token.endswith("@"))
        )
    ):
        return 1
    if token in _UNARY_OPS or token.startswith(("swap", "sort")):
        return 0
    if token in _BINARY_OPS:
        return -1
    if token in _TERNARY_OPS or token in _CLAMP_OPS:
        return -2
    if (
        token.endswith("[]")
        and len(token) > 2
        and not _REL_STATIC_PATTERN_POSTFIX.match(token)
    ):
        return -1
    if (
        token.endswith("!")
        and len(token) > 1
        and not token.startswith("[")
        and not _REL_STATIC_PATTERN_POSTFIX.match(token)
    ):
        return -1
    if (
        token.endswith("@")
        and len(token) > 1
        and not token.startswith("[")
        and not _REL_STATIC_PATTERN_POSTFIX.match(token)
    ):
        return 1

    if token.startswith("drop"):
        m = _DROP_PATTERN.fullmatch(token)
        if m:
            n_str = m.group(1)
            return -(int(n_str) if n_str else 1)

    return 0


@lru_cache
def get_op_arity(token: str) -> int:
    """Return number of operands for a token."""
    if token in _UNARY_OPS:
        return 1
    if token in _BINARY_OPS:
        return 2
    if token in _TERNARY_OPS or token in _CLAMP_OPS:
        return 3
    if (
        token.endswith("[]")
        and len(token) > 2
        and not _REL_STATIC_PATTERN_POSTFIX.match(token)
    ):
        return 2
    if (
        token.endswith("!")
        and len(token) > 1
        and not token.startswith("[")
        and not _REL_STATIC_PATTERN_POSTFIX.match(token)
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


def get_used_variable_names(tokens: Union[list[str], str]) -> set[str]:
    """Extract all variable names used in the expression."""

    if isinstance(tokens, str):
        tokens = tokenize_expr(tokens)

    return set(
        varname[:-1]
        for varname in tokens
        if varname.endswith("!") or varname.endswith("@")
    )


# modified from jvsfunc.ex_planes()
def ex_planes(
    clip: vs.VideoNode, expr: list[str], planes: Optional[Union[int, list[int]]] = None
) -> list[str]:
    if planes:
        plane_range = range(clip.format.num_planes)
        planes = [planes] if isinstance(planes, int) else planes
        expr = [expr[0] if i in planes else "" for i in plane_range]
    return expr
