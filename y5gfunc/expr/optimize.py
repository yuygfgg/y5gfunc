import re
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
    stack = []
    result_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Handle numeric constants
        if is_numeric(token):
            stack.append(parse_numeric(token))
            result_tokens.append(token)
            i += 1
            continue

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
                    i += 1
                    continue
                else:
                    stack.append(None)  # Put placeholder back
            else:
                stack.append(None)  # Non-constant
            result_tokens.append(token)
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

            if isinstance(stack[-1], (int, float)) and isinstance(
                stack[-2], (int, float)
            ):
                b = stack.pop()
                a = stack.pop()
                result = calculate_binary(token, a, b)
                if result is not None:
                    stack.append(result)
                    # Remove operands
                    result_tokens.pop()
                    result_tokens.pop()
                    # Add result
                    if isinstance(result, int):
                        result_tokens.append(str(result))
                    else:
                        result_tokens.append(f"{result:g}")
                    i += 1
                    continue
                else:
                    stack.append(a)
                    stack.append(b)
            else:
                # At least one operand is not a constant
                stack.append(None)
            result_tokens.append(token)
            i += 1
            continue

        # Handle ternary operator
        if token == "?":
            if len(stack) < 3:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Ternary operator requires 3 operands."
                )

            if all(isinstance(x, (int, float)) for x in stack[-3:]):
                false_val = stack.pop()
                true_val = stack.pop()
                cond = stack.pop()
                result = calculate_ternary(cond, true_val, false_val)
                stack.append(result)
                # Remove operands
                for _ in range(3):
                    result_tokens.pop()
                # Add result
                if isinstance(result, int):
                    result_tokens.append(str(result))
                else:
                    result_tokens.append(f"{result:g}")
                i += 1
                continue
            else:
                stack.append(None)
            result_tokens.append(token)
            i += 1
            continue

        # Handle clamp/clip operators
        if token in {"clamp", "clip"}:
            if len(stack) < 3:
                raise ValueError(
                    f"Stack underflow at token '{token}' at position {i}. Clamp operator requires 3 operands."
                )

            if all(isinstance(x, (int, float)) for x in stack[-3:]):
                max_val = stack.pop()
                min_val = stack.pop()
                value = stack.pop()
                result = max(min_val, min(max_val, value))
                stack.append(result)
                # Remove operands
                for _ in range(3):
                    result_tokens.pop()
                # Add result
                if isinstance(result, int):
                    result_tokens.append(str(result))
                else:
                    result_tokens.append(f"{result:g}")
                i += 1
                continue
            else:
                stack.append(None)
            result_tokens.append(token)
            i += 1
            continue

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
                    f"Stack underflow at token '{token}' at position {i}. Swap requires at least {n+1} items on stack."
                )

            # Simulate swap operation for stack tracking
            if all(isinstance(stack[-(i + 1)], (int, float)) for i in range(n + 1)):
                stack[-1], stack[-(n + 1)] = stack[-(n + 1)], stack[-1]
            else:
                stack.append(None)

            result_tokens.append(token)
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
                    f"Stack underflow at token '{token}' at position {i}. Dup requires at least {n+1} items on stack."
                )

            # Simulate dup operation for stack tracking
            if isinstance(stack[-(n + 1)], (int, float)):
                stack.append(stack[-(n + 1)])
            else:
                stack.append(None)

            result_tokens.append(token)
            i += 1
            continue

        # Other tokens, no constant folding
        stack.append(None)  # Non-constant placeholder
        result_tokens.append(token)
        i += 1

    return " ".join(result_tokens)


def convert_dynamic_to_static(expr: str) -> str:
    """Convert dynamic pixel access to static when possible"""
    tokens = tokenize_expr(expr)
    stack = []
    result_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check for dynamic pixel access pattern (clip identifier followed by [])
        if token.endswith("[]") and len(stack) >= 2:
            y_token_idx = len(result_tokens) - 1
            x_token_idx = len(result_tokens) - 2

            # Check if x and y are constants
            if (
                x_token_idx >= 0
                and is_numeric(result_tokens[x_token_idx])
                and y_token_idx >= 0
                and is_numeric(result_tokens[y_token_idx])
            ):
                # Get clip identifier and coordinates
                clip_identifier = token[:-2]
                y = parse_numeric(result_tokens[y_token_idx])
                x = parse_numeric(result_tokens[x_token_idx])

                # Remove coordinate tokens
                result_tokens.pop()  # Remove y coordinate
                result_tokens.pop()  # Remove x coordinate

                # Add static access token
                result_tokens.append(f"{clip_identifier}[{int(x)},{int(y)}]")

                # Update stack state
                stack.pop()  # y
                stack.pop()  # x
                stack.append(None)  # Placeholder for access result

                i += 1
                continue

        # Handle numeric constants (update stack)
        if is_numeric(token):
            stack.append(parse_numeric(token))
        else:
            # Update stack state for other tokens
            try:
                stack = update_stack_for_token(token, stack)
            except ValueError as e:
                raise ValueError(f"Error at token '{token}' at position {i}: {str(e)}")

        result_tokens.append(token)
        i += 1

    return " ".join(result_tokens)


def update_stack_for_token(token: str, stack: list[Any]) -> list[Any]:
    """Simulate stack operations to update stack state"""
    # Unary operators
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
                f"Stack underflow. Unary operator '{token}' requires 1 operand."
            )
        stack.pop()
        stack.append(None)  # Result placeholder

    # Binary operators
    elif token in {
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
                f"Stack underflow. Binary operator '{token}' requires 2 operands."
            )
        stack.pop()
        stack.pop()
        stack.append(None)  # Result placeholder

    # Ternary operator
    elif token == "?":
        if len(stack) < 3:
            raise ValueError("Stack underflow. Ternary operator requires 3 operands.")
        stack.pop()
        stack.pop()
        stack.pop()
        stack.append(None)  # Result placeholder

    # clamp/clip operator
    elif token in {"clamp", "clip"}:
        if len(stack) < 3:
            raise ValueError("Stack underflow. Clamp operator requires 3 operands.")
        stack.pop()
        stack.pop()
        stack.pop()
        stack.append(None)  # Result placeholder

    # swap operations
    elif token.startswith("swap"):
        n = 1  # Default for 'swap'
        if len(token) > 4:  # Extract N from 'swapN'
            try:
                n = int(token[4:])
            except ValueError:
                pass

        if len(stack) <= n:
            raise ValueError(
                f"Stack underflow. {token} requires at least {n+1} items on stack."
            )

        # Simulate swap operation
        if n > 0:
            stack[-1], stack[-(n + 1)] = stack[-(n + 1)], stack[-1]

    # dup operations
    elif token.startswith("dup"):
        n = 0  # Default for 'dup'
        if len(token) > 3:  # Extract N from 'dupN'
            try:
                n = int(token[3:])
            except ValueError:
                pass

        if len(stack) <= n:
            raise ValueError(
                f"Stack underflow. {token} requires at least {n+1} items on stack."
            )

        # Simulate dup operation
        stack.append(stack[-(n + 1)])

    # drop operations
    elif token.startswith("drop"):
        n = 1  # Default for 'drop'
        if len(token) > 4:  # Extract N from 'dropN'
            try:
                n = int(token[4:])
            except ValueError:
                pass

        if len(stack) < n:
            raise ValueError(
                f"Stack underflow. {token} requires at least {n} items on stack."
            )

        # Simulate drop operation
        for _ in range(n):
            stack.pop()

    # sort operations
    elif token.startswith("sort"):
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
                f"Stack underflow. {token} requires at least {n} items on stack."
            )

        # For simulation, just keep the stack with placeholders
        for _ in range(n):
            stack.pop()
        for _ in range(n):
            stack.append(None)

    # Variable operations
    elif token.endswith("!") or token.endswith("@"):
        if token.endswith("!"):  # Store
            if len(stack) < 1:
                raise ValueError(
                    "Stack underflow. Variable store requires 1 value on stack."
                )
            stack.pop()
        else:  # Load
            stack.append(None)

    # Other tokens (variables, clip loads, etc.)
    else:
        stack.append(None)  # Add placeholder

    return stack
