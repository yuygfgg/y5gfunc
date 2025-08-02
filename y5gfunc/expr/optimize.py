from typing import Optional, Union, Any
import math
from enum import IntEnum
from functools import lru_cache

from .utils import (
    _UNARY_OPS,
    _BINARY_OPS,
    _TERNARY_OPS,
    _CLAMP_OPS,
    _REL_STATIC_PATTERN_POSTFIX,
    is_token_numeric,
    tokenize_expr,
    parse_numeric,
)
from .transform import (
    convert_sort,
    convert_drop,
    convert_var,
    convert_pi,
)


class OptimizeLevel(IntEnum):
    """
    Optimization level.

    Attributes:
        O0: No optimization.
        O1: Basic constant folding that preserves all variables.
        O2: Slightly more aggressive optimization that may remove some variables.
        O3: Aggressive optimization that sacrifices readability for performance.
        OFast: Most aggressive optimization that may slightly change the result. (Generally more accurate though)
    """

    O0 = 0
    O1 = 1
    O2 = 2
    O3 = 3
    OFast = 4


def optimize_akarin_expr(
    expr: str, optimize_level: OptimizeLevel = OptimizeLevel.OFast
) -> str:
    """Fold constants and convert dynamic pixel access to static when possible."""
    expr = expr.strip()
    if not expr:
        return expr

    if optimize_level == OptimizeLevel.O0:
        return expr

    expr = convert_sort(expr)

    if optimize_level == OptimizeLevel.OFast:
        expr = convert_pi(expr)

    prev_expr = None
    current_expr = expr

    while prev_expr != current_expr:
        prev_expr = current_expr
        current_expr = eliminate_immediate_store_load(
            fold_constants(current_expr, optimize_level)
        )
        if optimize_level >= OptimizeLevel.O2:
            current_expr = remove_unused_variables(current_expr)
        if optimize_level >= OptimizeLevel.O3:
            current_expr = convert_drop(current_expr)

    optimized_expr = convert_dynamic_to_static(current_expr)
    return (
        fold_constants(convert_var(optimized_expr), optimize_level)
        if optimize_level >= OptimizeLevel.O3
        else optimized_expr
    )


@lru_cache
def calculate_unary(
    op: str, a: Union[int, float], optimize_level: OptimizeLevel
) -> Optional[Union[int, float]]:
    """Calculate result of unary operation"""
    ofast_only_ops = {"exp", "log", "sqrt", "sin", "cos"}

    if op in ofast_only_ops and optimize_level < OptimizeLevel.OFast:
        return None

    operators = {
        "exp": math.exp,  # OFast only
        "log": math.log,  # OFast only
        "sqrt": math.sqrt,  # OFast only
        "sin": math.sin,  # OFast only
        "cos": math.cos,  # OFast only
        "abs": abs,
        "not": lambda x: 0.0 if float(x) > 0.0 else 1.0,
        "bitnot": lambda x: ~round(x),  # Integer specific
        "trunc": math.trunc,  # Returns int
        "round": round,  # Returns int if possible
        "floor": math.floor,  # Returns float
    }
    if op not in operators:
        return None

    try:
        arg = a
        if op in ["exp", "log", "sqrt", "sin", "cos", "not", "floor"]:
            arg = float(a)
        elif op == "bitnot":
            if isinstance(a, float):
                if not a.is_integer():
                    return None  # Cannot bitwise-not a non-integer float
                arg = int(a)
            else:  # Already int
                arg = int(a)

        result = operators[op](arg)

        if (
            isinstance(result, float)
            and result.is_integer()
            and op in ["abs", "trunc", "round", "floor", "bitnot"]
        ):
            return int(result)
        return result
    except (ValueError, OverflowError, TypeError):
        return None


@lru_cache
def calculate_binary(
    op: str, a: Union[int, float], b: Union[int, float]
) -> Optional[Union[int, float]]:
    """Calculate result of binary operation"""
    operators = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: float(x) / float(y) if float(y) != 0 else None,
        "max": max,
        "min": min,
        "pow": lambda x, y: pow(float(x), float(y)),
        "**": lambda x, y: pow(float(x), float(y)),
        ">": lambda x, y: 1.0 if float(x) > float(y) else 0.0,
        "<": lambda x, y: 1.0 if float(x) < float(y) else 0.0,
        "=": lambda x, y: 1.0 if float(x) == float(y) else 0.0,
        ">=": lambda x, y: 1.0 if float(x) >= float(y) else 0.0,
        "<=": lambda x, y: 1.0 if float(x) <= float(y) else 0.0,
        "and": lambda x, y: 1.0 if float(x) > 0 and float(y) > 0 else 0.0,
        "or": lambda x, y: 1.0 if float(x) > 0 or float(y) > 0 else 0.0,
        "xor": lambda x, y: 1.0 if (float(x) > 0) != (float(y) > 0) else 0.0,
        "%": lambda x, y: x % y if y != 0 else None,
        "bitand": lambda x, y: round(x) & round(y),
        "bitor": lambda x, y: round(x) | round(y),
        "bitxor": lambda x, y: round(x) ^ round(y),
    }
    if op not in operators:
        return None

    try:
        arg1, arg2 = a, b
        if op.startswith("bit") or op == "%":
            f_args = []
            for v in [a, b]:
                if isinstance(v, float):
                    if not v.is_integer():
                        return None  # Cannot operate on non-integer float
                    f_args.append(int(v))
                else:
                    f_args.append(int(v))
            arg1, arg2 = f_args[0], f_args[1]
            if (op == "%" or op == "/") and arg2 == 0:
                return None  # check zero division
        elif op in [
            ">",
            "<",
            "=",
            ">=",
            "<=",
            "and",
            "or",
            "xor",
        ]:  # Comparisons use float logic
            arg1, arg2 = float(a), float(b)

        result = operators[op](arg1, arg2)
        if result is None:
            return None

        if (
            isinstance(result, float)
            and result.is_integer()
            and op in ["+", "-", "*", "%", "max", "min", "bitand", "bitor", "bitxor"]
        ):
            return int(result)
        return result
    except (ZeroDivisionError, ValueError, OverflowError, TypeError):
        return None


@lru_cache
def calculate_ternary(
    cond: Union[int, float], true_val: Union[int, float], false_val: Union[int, float]
) -> Union[int, float]:
    """Calculate result of ternary operation"""
    return true_val if float(cond) > 0 else false_val


@lru_cache
def format_number(num: Union[int, float]) -> str:
    """Format number back to string representation for expression"""
    if isinstance(num, int):
        return str(num)
    if isinstance(num, float):
        formatted = f"{num:g}"

        if "E" in formatted:
            formatted = formatted.replace("E", "e")

        return formatted


def fold_constants(
    expr: str, optimize_level: OptimizeLevel = OptimizeLevel.OFast
) -> str:
    """Perform constant folding optimization"""
    tokens = tokenize_expr(expr)
    stack: list[Any] = []
    result_tokens: list[str] = []
    variable_values: dict[str, Union[int, float, None]] = {}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if is_token_numeric(token):
            try:
                value = parse_numeric(token)
                stack.append(value)
                result_tokens.append(token)
            except ValueError:
                stack.append(None)
                result_tokens.append(token)
            i += 1
            continue

        is_store = (
            token.endswith("!")
            and len(token) > 1
            and not token.startswith("[")
            and not _REL_STATIC_PATTERN_POSTFIX.match(token)
        )
        if is_store:
            var_name = token[:-1]
            if not stack:
                raise ValueError(f"fold_constants: Stack underflow at store '{token}'")

            value_to_store = stack.pop()
            variable_values[var_name] = (
                value_to_store if isinstance(value_to_store, (int, float)) else None
            )

            result_tokens.append(token)
            i += 1
            continue

        is_load = (
            token.endswith("@")
            and len(token) > 1
            and not token.startswith("[")
            and not _REL_STATIC_PATTERN_POSTFIX.match(token)
        )
        if is_load:
            var_name = token[:-1]
            constant_value = variable_values.get(
                var_name
            )  # Check if known constant in this pass

            if isinstance(constant_value, (int, float)):
                stack.append(constant_value)
                result_tokens.append(format_number(constant_value))
            else:
                stack.append(None)
                result_tokens.append(token)
            i += 1
            continue

        if token in _UNARY_OPS:
            can_fold = False
            if stack and result_tokens:  # Need operand on stack and its token in result
                op1_stack_val = stack[-1]
                op1_token = result_tokens[-1]

                if isinstance(op1_stack_val, (int, float)) and is_token_numeric(
                    op1_token
                ):
                    result = calculate_unary(token, op1_stack_val, optimize_level)
                    if result is not None:
                        stack.pop()
                        stack.append(result)
                        result_tokens.pop()
                        result_tokens.append(format_number(result))
                        can_fold = True

            if not can_fold:
                if stack:
                    stack.pop()
                stack.append(None)
                result_tokens.append(token)

            i += 1
            continue

        if token in _BINARY_OPS:
            can_fold = False
            if len(stack) >= 2 and len(result_tokens) >= 2:
                op2_stack_val = stack[-1]
                op1_stack_val = stack[-2]
                op2_token = result_tokens[-1]
                op1_token = result_tokens[-2]

                if (
                    isinstance(op1_stack_val, (int, float))
                    and isinstance(op2_stack_val, (int, float))
                    and is_token_numeric(op1_token)
                    and is_token_numeric(op2_token)
                ):
                    result = calculate_binary(token, op1_stack_val, op2_stack_val)
                    if result is not None:
                        stack.pop()
                        stack.pop()
                        stack.append(result)
                        result_tokens.pop()
                        result_tokens.pop()  # Remove operand tokens
                        result_tokens.append(
                            format_number(result)
                        )  # Append result token
                        can_fold = True

            if not can_fold:
                if len(stack) >= 2:
                    stack.pop()
                    stack.pop()
                elif len(stack) == 1:
                    stack.pop()
                stack.append(None)  # Result is unknown
                result_tokens.append(token)  # Keep the operator token

            i += 1
            continue

        if token in _TERNARY_OPS:
            can_fold = False
            if len(stack) >= 3 and len(result_tokens) >= 3:
                false_val_stack = stack[-1]
                true_val_stack = stack[-2]
                cond_stack = stack[-3]
                false_token = result_tokens[-1]
                true_token = result_tokens[-2]
                cond_token = result_tokens[-3]

                if (
                    isinstance(cond_stack, (int, float))
                    and isinstance(true_val_stack, (int, float))
                    and isinstance(false_val_stack, (int, float))
                    and is_token_numeric(cond_token)
                    and is_token_numeric(true_token)
                    and is_token_numeric(false_token)
                ):
                    result = calculate_ternary(
                        cond_stack, true_val_stack, false_val_stack
                    )
                    stack.pop()
                    stack.pop()
                    stack.pop()
                    stack.append(result)
                    result_tokens.pop()
                    result_tokens.pop()
                    result_tokens.pop()  # Remove operand tokens
                    result_tokens.append(format_number(result))  # Append result token
                    can_fold = True

            if not can_fold:
                if len(stack) >= 3:
                    stack.pop()
                    stack.pop()
                    stack.pop()
                elif len(stack) == 2:
                    stack.pop()
                    stack.pop()
                elif len(stack) == 1:
                    stack.pop()
                stack.append(None)
                result_tokens.append(token)
            i += 1
            continue

        if token in _CLAMP_OPS:
            can_fold = False
            if len(stack) >= 3 and len(result_tokens) >= 3:
                max_val_stack = stack[-1]
                min_val_stack = stack[-2]
                value_val_stack = stack[-3]
                max_token = result_tokens[-1]
                min_token = result_tokens[-2]
                value_token = result_tokens[-3]

                if (
                    isinstance(value_val_stack, (int, float))
                    and isinstance(min_val_stack, (int, float))
                    and isinstance(max_val_stack, (int, float))
                    and is_token_numeric(value_token)
                    and is_token_numeric(min_token)
                    and is_token_numeric(max_token)
                ):
                    min_v = min(min_val_stack, max_val_stack)
                    max_v = max(min_val_stack, max_val_stack)
                    result = max(min_v, min(max_v, value_val_stack))

                    stack.pop()
                    stack.pop()
                    stack.pop()
                    stack.append(result)
                    result_tokens.pop()
                    result_tokens.pop()
                    result_tokens.pop()
                    result_tokens.append(format_number(result))
                    can_fold = True

            if not can_fold:
                if len(stack) >= 3:
                    stack.pop()
                    stack.pop()
                    stack.pop()
                elif len(stack) == 2:
                    stack.pop()
                    stack.pop()
                elif len(stack) == 1:
                    stack.pop()
                stack.append(None)
                result_tokens.append(token)

            i += 1
            continue

        if token.startswith("swap"):
            n = 1
            if len(token) > 4:
                try:
                    n = int(token[4:])
                except ValueError:
                    pass  # Treat as unknown token if N is invalid
            if n < 0:
                raise ValueError("fold_constants: Swap count cannot be negative")

            if len(stack) <= n:
                raise ValueError(f"fold_constants: Stack underflow for {token}")

            if n > 0:  # Simulate swap on evaluation stack
                stack[-1], stack[-(n + 1)] = stack[-(n + 1)], stack[-1]
            result_tokens.append(token)
            i += 1
            continue

        if token.startswith("dup"):
            n = 0
            if len(token) > 3:
                try:
                    n = int(token[3:])
                except ValueError:
                    pass
            if n < 0:
                raise ValueError("fold_constants: Dup index cannot be negative")

            if len(stack) <= n:
                raise ValueError(f"fold_constants: Stack underflow for {token}")
            stack.append(stack[-(n + 1)])
            result_tokens.append(token)
            i += 1
            continue

        if token.startswith("drop"):
            n = 1
            if len(token) > 4:
                try:
                    n = int(token[4:])
                except ValueError:
                    pass
            if n < 0:
                raise ValueError("fold_constants: Drop count cannot be negative")
            if n == 0:  # drop0 is no-op, just keep token
                pass
            elif len(stack) < n:
                raise ValueError(f"fold_constants: Stack underflow for {token}")
            else:  # Simulate drop on evaluation stack
                del stack[-n:]
            result_tokens.append(token)
            i += 1
            continue

        if token.startswith("sort"):
            n = 0
            if len(token) > 4:
                try:
                    n = int(token[4:])
                except ValueError:
                    pass
            if n <= 0:
                raise ValueError("fold_constants: Sort count must be positive")
            if len(stack) < n:
                raise ValueError(f"fold_constants: Stack underflow for {token}")
            del stack[-n:]
            for _ in range(n):
                stack.append(None)
            result_tokens.append(token)
            i += 1
            continue

        is_dynamic_access = (
            token.endswith("[]")
            and len(token) > 2
            and not token.startswith("[")
            and not _REL_STATIC_PATTERN_POSTFIX.match(token)
            and not is_token_numeric(token)
        )
        if is_dynamic_access:
            if len(stack) < 2:
                raise ValueError(
                    f"fold_constants: Stack underflow for dynamic access '{token}'"
                )
            stack.pop()
            stack.pop()  # Consume y, x coords from stack
            stack.append(None)  # Result is unknown
            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        match = _REL_STATIC_PATTERN_POSTFIX.match(token)
        if match:
            stack.append(None)  # Result is unknown during folding pass
            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        stack.append(None)
        result_tokens.append(token)
        i += 1

    return " ".join(result_tokens)


def convert_dynamic_to_static(expr: str) -> str:
    """Convert dynamic pixel access to static when possible"""
    tokens = tokenize_expr(expr)
    if not tokens:
        return ""

    result_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        converted = False

        is_dynamic_access = (
            token.endswith("[]")
            and len(token) > 2
            and not token.startswith("[")
            and not _REL_STATIC_PATTERN_POSTFIX.match(token)
            and not is_token_numeric(token)
        )

        if is_dynamic_access and len(result_tokens) >= 2:
            y_token = result_tokens[-1]
            x_token = result_tokens[-2]

            if is_token_numeric(x_token) and is_token_numeric(y_token):
                try:
                    y_val = parse_numeric(y_token)
                    x_val = parse_numeric(x_token)

                    is_x_int = isinstance(x_val, int) or (
                        isinstance(x_val, float) and x_val.is_integer()
                    )
                    is_y_int = isinstance(y_val, int) or (
                        isinstance(y_val, float) and y_val.is_integer()
                    )

                    if is_x_int and is_y_int:
                        x_int = int(x_val)
                        y_int = int(y_val)

                        clip_identifier = token[:-2]  # Get clip name
                        suffix = ""
                        if ":" in clip_identifier:
                            parts = clip_identifier.split(":", 1)
                            clip_identifier = parts[0]
                            suffix = ":" + parts[1]

                        result_tokens.pop()  # Remove y token
                        result_tokens.pop()  # Remove x token

                        result_tokens.append(
                            f"{clip_identifier}[{x_int},{y_int}]{suffix}"
                        )
                        converted = True

                except (
                    ValueError,
                    OverflowError,
                ):  # Handle parse errors or large floats
                    pass  # Fall through to append original token

        if not converted:
            result_tokens.append(token)

        i += 1  # Move to the next token in the *original* list

    return " ".join(result_tokens)


def eliminate_immediate_store_load(expr: str) -> str:
    """Remove redundant store-load pairs like `x! x@` not referenced later."""
    tokens = tokenize_expr(expr)
    if not tokens:
        return ""

    result_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        is_store = (
            token.endswith("!")
            and len(token) > 1
            and not token.startswith("[")
            and not _REL_STATIC_PATTERN_POSTFIX.match(token)
        )

        if is_store and i + 1 < len(tokens):
            var_name = token[:-1]
            next_token = tokens[i + 1]
            expected_load = f"{var_name}@"
            var_store = f"{var_name}!"

            if next_token == expected_load:
                later_tokens = tokens[i + 2 :]

                next_load = math.inf
                next_store = math.inf
                variable_used_later = False
                for j, tk in enumerate(later_tokens):
                    if tk == expected_load:
                        next_load = min(next_load, j)
                    elif tk == var_store:
                        next_store = min(next_store, j)

                if next_load < next_store:
                    variable_used_later = True

                if not variable_used_later:
                    i += 2
                    continue  # Do *not* append them to result_tokens

        result_tokens.append(token)
        i += 1

    return " ".join(result_tokens)


def remove_unused_variables(expr: str) -> str:
    """Remove variables that are not used in the expression."""
    tokens = tokenize_expr(expr)
    if not tokens:
        return ""

    result: list[str] = []
    i = 0

    used_vars = set(tk[:-1] for tk in tokens if tk.endswith("@"))

    while i < len(tokens):
        token = tokens[i]

        if token.endswith("!"):
            var_name = token[:-1]
            if var_name in used_vars:
                result.append(token)
            else:
                result.append("drop")
        else:
            result.append(token)

        i += 1

    return " ".join(result)
