from .optimize import optimize_akarin_expr
from .postfix2infix import postfix2infix
from .infix2postfix import infix2postfix
from .utils import is_clip_postfix, tokenize_expr, _FRAME_PROP_PATTERN
from typing import Optional
import math
from dataclasses import dataclass, asdict


@dataclass
class ConstantValues:
    pi: float = math.pi
    N: Optional[int] = None
    X: Optional[int] = None
    Y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


# FIXME: support pixel access emulation (how?)
def emulate_expr(
    expr: str,
    constants: ConstantValues = ConstantValues(),
    clip_value: Optional[dict[str, float]] = None,
    clip_prop: Optional[dict[str, list[tuple[str, float]]]] = None,
) -> float:
    """
    Emulate Akarin.Expr with the given constants, clip values and properties.

    Args:
        expr: The expression to emulate.
        constants: An object containing constant values (N, X, Y, width, height, pi).
        clip_value: The values of the clips. (e.g. {"src0": 1.0, "src1": 2.0})
        clip_prop: The properties of the clips. (e.g. {"src0": [("prop1", 1234), ("prop2", 4567.8)], "src1": [("yajyuu_senpai", 114514)]})

    Returns:
        The result of the expression.

    Raises:
        ValueError: If the expression is invalid or if a clip or property is not found.
    """
    clip_value = clip_value or {}
    clip_prop = clip_prop or {}

    all_clip_names = list(clip_value.keys()) + list(clip_prop.keys())
    for name in all_clip_names:
        if not (is_clip_postfix(name) and name.startswith("src")):
            raise ValueError(f"emulate_expr: Invalid clip name provided: '{name}'")

    constant_values_dict = asdict(constants)

    expr = optimize_akarin_expr(expr)
    infix_expr = postfix2infix(expr)
    expanded_expr = infix2postfix(infix_expr)
    tokens = tokenize_expr(expanded_expr)

    for i, token in enumerate(tokens):
        if is_clip_postfix(token):
            if token in clip_value:
                value = clip_value[token]
                tokens[i] = str(value)
            else:
                raise ValueError(
                    f"emulate_expr: Clip '{token}' not found in provided values or properties."
                )
        elif _FRAME_PROP_PATTERN.match(token):
            clip_name, prop_name = token.split(".")
            found_prop_val = False
            for clip_nm, props in clip_prop.items():
                if clip_nm == clip_name:
                    for prop in props:
                        if prop[0] == prop_name:
                            if not found_prop_val:
                                found_prop_val = True
                                tokens[i] = str(prop[1])
                            else:
                                raise ValueError(
                                    f"emulate_expr: Duplicate property '{prop_name}' found for clip '{clip_name}'."
                                )
            if not found_prop_val:
                raise ValueError(
                    f"emulate_expr: Property '{prop_name}' not found for clip '{clip_name}'."
                )
        elif token in {"pi", "N", "X", "Y", "width", "height"}:
            if token in constant_values_dict:
                tokens[i] = str(constant_values_dict[token])
            else:
                raise ValueError(
                    f"emulate_expr: Constant '{token}' not found in provided constants."
                )

    try:
        ret = float(optimize_akarin_expr(" ".join(tokens)))
    except Exception as e:
        raise ValueError(
            f"emulate_expr: Error occurred while optimizing expression: {e}"
        )

    return ret
