import regex as re
from typing import Optional, Union, Any
import math

token_pattern = re.compile(r"(\w+)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?::(?:c|m))?")
split_pattern = re.compile(r"\s+")
number_patterns = [re.compile(pattern) for pattern in [
    r"^0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?$",  # Hexadecimal
    r"^0[0-7]+$",  # Octal
    r"^[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)$",  # Decimal and scientific notation
]]
hex_pattern = re.compile(r"^0x")
hex_parts_pattern = re.compile(r"^(0x[0-9A-Fa-f]+)(?:\.([0-9A-Fa-f]+))?(?:p([+\-]?\d+))?$")
octal_pattern = re.compile(r"^0[0-7]")

def optimize_akarin_expr(expr: str) -> str:
    """Optimize akarin.Expr expressions:
    1. Constant folding
    2. Convert dynamic pixel access to static when possible
    """
    # Initial expression preprocessing
    expr = expr.strip()
    if not expr:
        return expr

    # Multi-round constant folding until no further optimization is possible
    prev_expr = None
    current_expr = expr

    while prev_expr != current_expr:
        prev_expr = current_expr
        current_expr = fold_constants(current_expr)

    # Dynamic to static pixel access conversion
    optimized_expr = convert_dynamic_to_static(current_expr)

    return optimized_expr


def tokenize_expr(expr: str) -> list[str]:
    """Convert expression string to a list of tokens"""
    expr = expr.strip()
    # Special handling for pixel access syntax to prevent splitting
    expr = token_pattern.sub(r"\1[\2,\3]", expr)
    return split_pattern.split(expr)


def is_numeric(token: str) -> bool:
    """Check if a token is a numeric constant"""
    # Support hexadecimal, octal, and decimal numbers

    for pattern in number_patterns:
        if pattern.match(token):
            return True
    return False


def parse_numeric(token: str) -> Union[int, float]:
    """Parse a numeric token to its actual value"""
    if hex_pattern.match(token):  # Hexadecimal
        # Handle hexadecimal with fraction and exponent
        if "." in token or "p" in token.lower():
            parts = hex_parts_pattern.match(token.lower())
            if parts:
                integer_part = int(parts.group(1), 16)
                fractional_part = 0
                if parts.group(2):
                    fractional_part = int(parts.group(2), 16) / (
                        16 ** len(parts.group(2))
                    )
                exponent = 0
                if parts.group(3):
                    exponent = int(parts.group(3))
                return (integer_part + fractional_part) * (2**exponent)
        return int(token, 16)
    elif octal_pattern.match(token):  # Octal
        return int(token, 8)
    else:  # Decimal
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)


def calculate_unary(op: str, a: Union[int, float]) -> Optional[Union[int, float]]:
    """Calculate result of unary operation"""
    operators = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "abs": abs,
        "not": lambda x: 0.0 if x > 0 else 1.0,
        "bitnot": lambda x: ~int(x),
        "trunc": math.trunc,
        "round": round,
        "floor": math.floor,
    }

    if op in operators:
        try:
            return operators[op](a)
        except (ValueError, OverflowError, TypeError):
            return None
    return None


def calculate_binary(
    op: str, a: Union[int, float], b: Union[int, float]
) -> Optional[Union[int, float]]:
    """Calculate result of binary operation"""
    operators = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "max": max,
        "min": min,
        "pow": pow,
        "**": pow,
        ">": lambda x, y: 1.0 if x > y else 0.0,
        "<": lambda x, y: 1.0 if x < y else 0.0,
        "=": lambda x, y: 1.0 if x == y else 0.0,
        ">=": lambda x, y: 1.0 if x >= y else 0.0,
        "<=": lambda x, y: 1.0 if x <= y else 0.0,
        "and": lambda x, y: 1.0 if x > 0 and y > 0 else 0.0,
        "or": lambda x, y: 1.0 if x > 0 or y > 0 else 0.0,
        "xor": lambda x, y: 1.0 if (x > 0) != (y > 0) else 0.0,
        "%": lambda x, y: x % y,
        "bitand": lambda x, y: int(x) & int(y),
        "bitor": lambda x, y: int(x) | int(y),
        "bitxor": lambda x, y: int(x) ^ int(y),
    }

    if op in operators:
        try:
            return operators[op](a, b)
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return None
    return None


def calculate_ternary(
    cond: Union[int, float], true_val: Union[int, float], false_val: Union[int, float]
) -> Union[int, float]:
    """Calculate result of ternary operation"""
    return true_val if cond > 0 else false_val


def fold_constants(expr: str) -> str:
    """Perform constant folding optimization"""
    tokens = tokenize_expr(expr)
    stack: list[
        Any
    ] = []  # Stores actual values (numbers) or None for non-constants during evaluation
    result_tokens: list[str] = []  # Stores the resulting optimized token list
    variable_values: dict[
        str, Union[int, float, None]
    ] = {}  # Tracks known constant variable values

    i = 0
    while i < len(tokens):
        token = tokens[i]
        is_processed = (
            False  # Flag to check if the token was handled and loop should continue
        )

        # Handle numeric constants
        if is_numeric(token):
            value = parse_numeric(token)
            stack.append(value)
            result_tokens.append(token)
            i += 1
            continue

        # --- Handle Variable Store (!) ---
        if (
            token.endswith("!") and len(token) > 1 and not token.startswith("[")
        ):  # Avoid matching array access
            var_name = token[:-1]
            if len(stack) < 1:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Store '!' requires 1 operand."
                )

            value_to_store = stack.pop()
            if isinstance(value_to_store, (int, float)):
                variable_values[var_name] = value_to_store  # Store constant value
            else:
                variable_values[var_name] = None  # Mark as non-constant

            result_tokens.append(token)  # Keep the store token in the output
            i += 1
            continue  # Skip other checks

        # --- Handle Variable Load (@) ---
        if (
            token.endswith("@") and len(token) > 1 and not token.startswith("[")
        ):  # Avoid matching array access
            var_name = token[:-1]
            constant_value = variable_values.get(var_name)

            if isinstance(constant_value, (int, float)):
                # Substitute load with the constant value
                stack.append(constant_value)
                if isinstance(constant_value, int):
                    result_tokens.append(str(constant_value))
                else:
                    # Use 'g' format for concise float representation, avoid trailing zeros
                    result_tokens.append(f"{constant_value:g}")
                is_processed = True  # Mark as processed to skip adding original token
            else:
                # Variable not known constant, treat as non-constant
                stack.append(None)
                result_tokens.append(token)  # Keep the original load token

            if is_processed:
                i += 1
                continue  # Move to next token

        # Handle unary operators
        if token in {
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
        }:
            if len(stack) < 1:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Unary operator requires 1 operand."
                )

            if isinstance(stack[-1], (int, float)):
                a = stack.pop()
                result = calculate_unary(token, a)
                if result is not None:
                    stack.append(result)
                    # Remove operand
                    result_tokens.pop()
                    # Add result
                    if isinstance(result, int):
                        result_tokens.append(str(result))
                    else:
                        result_tokens.append(f"{result:g}")
                    is_processed = True
                else:
                    stack.append(
                        None
                    )  # Calculation failed (e.g., sqrt(-1)), result is non-constant
                    result_tokens.append(token)  # Keep operator
            else:
                stack.append(None)  # Operand was non-constant
                result_tokens.append(token)  # Keep operator

            if is_processed:
                i += 1
                continue

        # Handle binary operators
        if token in {
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
        }:
            if len(stack) < 2:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Binary operator requires 2 operands."
                )

            op2 = stack.pop()
            op1 = stack.pop()
            if isinstance(op1, (int, float)) and isinstance(op2, (int, float)):
                result = calculate_binary(token, op1, op2)
                if result is not None:
                    stack.append(result)
                    result_tokens.pop()  # Remove op2 token
                    result_tokens.pop()  # Remove op1 token
                    if isinstance(result, int):
                        result_tokens.append(str(result))
                    else:
                        result_tokens.append(f"{result:g}")
                    is_processed = True
                else:
                    stack.append(None)  # Calculation failed, result non-constant
                    result_tokens.append(token)  # Keep operator
            else:
                stack.append(None)  # At least one operand was non-constant
                result_tokens.append(token)  # Keep operator

            if is_processed:
                i += 1
                continue

        # Handle ternary operator "?"
        if token == "?":
            if len(stack) < 3:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Ternary operator requires 3 operands."
                )

            false_val_stack = stack.pop()
            true_val_stack = stack.pop()
            cond_stack = stack.pop()

            if (
                isinstance(cond_stack, (int, float))
                and isinstance(true_val_stack, (int, float))
                and isinstance(false_val_stack, (int, float))
            ):
                result = calculate_ternary(cond_stack, true_val_stack, false_val_stack)
                stack.append(result)
                # Remove operands
                for _ in range(3):
                    result_tokens.pop()
                # Add result
                if isinstance(result, int):
                    result_tokens.append(str(result))
                else:
                    result_tokens.append(f"{result:g}")
                is_processed = True
            else:
                stack.append(None)  # Operands not all constant
                result_tokens.append(token)  # Keep operator

            if is_processed:
                i += 1
                continue

        # Handle clamp/clip operators
        if token in {"clamp", "clip"}:
            if len(stack) < 3:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Clamp operator requires 3 operands."
                )

            max_val_stack = stack.pop()
            min_val_stack = stack.pop()
            value_stack = stack.pop()

            if (
                isinstance(value_stack, (int, float))
                and isinstance(min_val_stack, (int, float))
                and isinstance(max_val_stack, (int, float))
            ):
                # Ensure min <= max for correct clamp behavior during folding
                min_val = min(min_val_stack, max_val_stack)
                max_val = max(min_val_stack, max_val_stack)

                result = max(min_val, min(max_val, value_stack))
                stack.append(result)
                result_tokens.pop()  # max val token
                result_tokens.pop()  # min val token
                result_tokens.pop()  # value token
                if isinstance(result, int):
                    result_tokens.append(str(result))
                else:
                    result_tokens.append(f"{result:g}")
                is_processed = True
            else:
                stack.append(None)  # Operands not all constant
                result_tokens.append(token)  # Keep operator

            if is_processed:
                i += 1
                continue

        # --- Stack manipulation ops (swap, dup, drop, sort) ---
        # These don't change constant values directly, but we need to update the evaluation stack (`stack`)
        # They remain in the result_tokens as they affect runtime execution flow

        # Handle swap operations
        if token.startswith("swap"):
            n = 1  # Default for 'swap'
            if len(token) > 4:  # Extract N from 'swapN'
                try:
                    n = int(token[4:])
                except ValueError:
                    pass

            if len(stack) <= n:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Swap requires at least {n+1} items."
                )
            # Simulate swap on the evaluation stack
            if n > 0:  # swap0 is a no-op
                stack[-1], stack[-(n + 1)] = stack[-(n + 1)], stack[-1]
            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        # Handle dup operations
        if token.startswith("dup"):
            n = 0  # Default for 'dup'
            if len(token) > 3:  # Extract N from 'dupN'
                try:
                    n = int(token[3:])
                except ValueError:
                    pass

            if len(stack) <= n:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Dup requires at least {n+1} items."
                )
            # Simulate dup on the evaluation stack
            stack.append(stack[-(n + 1)])
            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        # Handle drop operations
        if token.startswith("drop"):
            n = 1  # Default for 'drop'
            if len(token) > 4:  # Extract N from 'dropN'
                try:
                    n = int(token[4:])
                except ValueError:
                    pass

            if len(stack) < n:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Drop requires at least {n} items."
                )
            # Simulate drop on the evaluation stack
            for _ in range(n):
                stack.pop()

            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        # sort operations
        if token.startswith("sort"):
            n = 0
            if len(token) > 4:  # Extract N from 'sortN'
                try:
                    n = int(token[4:])
                except ValueError:
                    pass

            if n <= 0:
                raise ValueError(f"Invalid sort count. {token} requires a positive number.")

            if len(stack) < n:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Sort requires at least {n} items."
                )

            sorted_segment = stack[-n:]
            all_const = all(isinstance(x, (int, float)) for x in sorted_segment)

            del stack[-n:]  # Remove items to be sorted

            if all_const:
                for _ in range(n):
                    stack.append(None)
            else:
                for _ in range(n):
                    stack.append(None)  # Non-constant result after sort

            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        # Other tokens, no constant folding
        if token.endswith("[]") and not token.startswith("["):
            # Dynamic access requires 2 preceding items (coords) which become non-constant after access
            if len(stack) < 2:
                raise ValueError(
                    f"Stack underflow for dynamic access '{token}'. Requires x y coordinates."
                )
            stack.pop()  # y coord
            stack.pop()  # x coord
            stack.append(None)  # Result of access is non-constant
            result_tokens.append(token)
        elif token_pattern.match(token):
            stack.append(None)  # Result of access is non-constant
            result_tokens.append(token)
        else:
            if not (
                token.endswith("!") or token.endswith("@")
            ):  # Avoid double-adding var ops
                stack.append(None)
                result_tokens.append(token)

        i += 1

    return " ".join(result_tokens)


def convert_dynamic_to_static(expr: str) -> str:
    """Convert dynamic pixel access to static when possible.
    Assumes constant folding (including variable substitution) has already occurred."""
    tokens = tokenize_expr(expr)
    result_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        can_convert = False

        # Check for dynamic pixel access pattern (clip[])
        if (
            token.endswith("[]")
            and not token.startswith("[")
            and len(result_tokens) >= 2
        ):
            # Check if the last two tokens in the *result* list are numeric constants
            y_token = result_tokens[-1]
            x_token = result_tokens[-2]

            if is_numeric(x_token) and is_numeric(y_token):
                # Perform conversion
                clip_identifier = token[:-2]
                try:
                    y = int(parse_numeric(y_token))
                    x = int(parse_numeric(x_token))

                    # Pop the constant coordinates from result_tokens
                    result_tokens.pop()  # Remove y coordinate token
                    result_tokens.pop()  # Remove x coordinate token

                    # Append the new static access token
                    result_tokens.append(f"{clip_identifier}[{x},{y}]")
                    can_convert = True
                except ValueError:
                    # If parsing fails for some reason, don't convert
                    pass

        if not can_convert:
            # If not converting, just append the original token
            result_tokens.append(token)

        i += 1

    return " ".join(result_tokens)
