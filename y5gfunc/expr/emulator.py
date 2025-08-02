from .postfix2infix import postfix2infix
from .utils import is_clip_postfix
from typing import Optional
import math
import regex as re
from dataclasses import dataclass, asdict


@dataclass
class ConstantValues:
    pi: float = math.pi
    N: Optional[int] = None
    X: Optional[int] = None
    Y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


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
        ValueError: If the expression contains invalid identifiers or if required constants are not set.
        ValueError: If the clip names do not follow the expected format.
        ValueError: If the nth function is called with insufficient arguments.
        ValueError: If a constant is not set when referenced in the expression.
        ValueError: If an unknown identifier is encountered in the expression.
        ValueError: If the clip names do not follow the expected format.
    """
    clip_value = clip_value or {}
    clip_prop = clip_prop or {}

    all_clip_names = list(clip_value.keys()) + list(clip_prop.keys())
    for name in all_clip_names:
        if not (is_clip_postfix(name) and name.startswith("src")):
            raise ValueError(f"emulate_expr: Invalid clip name provided: '{name}'")

    constant_values_dict = asdict(constants)

    infix_expr = postfix2infix(expr)

    exec_env = {
        "sin": math.sin,
        "cos": math.cos,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "floor": math.floor,
        "trunc": math.trunc,
        "clamp": lambda x, min_val, max_val: max(min_val, min(max_val, x)),
    }

    def create_nth_function(n):
        def nth_func(*args):
            if len(args) < n:
                raise ValueError(
                    f"emulate_expr: nth_{n} requires at least {n} arguments"
                )
            sorted_args = sorted(args)
            return sorted_args[n - 1]

        return nth_func

    nth_functions = re.findall(r"\bnth_(\d+)\s*\(", infix_expr)
    for n in set(nth_functions):
        n = int(n)
        exec_env[f"nth_{n}"] = create_nth_function(n)

    for clip_name, prop_list in clip_prop.items():
        for prop_name, prop_value in prop_list:
            pattern = f"{clip_name}.{prop_name}"
            infix_expr = infix_expr.replace(pattern, str(float(prop_value)))

    def replace_dollar(match) -> str:
        identifier = match.group(1)

        if identifier in constant_values_dict:
            if constant_values_dict[identifier] is not None:
                return str(float(constant_values_dict[identifier]))
            else:
                raise ValueError(f"emulate_expr: Constant '{identifier}' is not set.")

        if identifier in clip_value:
            return str(float(clip_value[identifier]))

        raise ValueError(f"emulate_expr: Unknown identifier: {identifier}")

    infix_expr = re.sub(r"\$([a-zA-Z_]\w*)", replace_dollar, infix_expr)
    infix_expr = infix_expr.replace("!", "not")

    exec(infix_expr, exec_env)

    return float(exec_env["RESULT"])
