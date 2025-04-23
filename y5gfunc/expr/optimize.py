import regex as re
from typing import Optional, Union, Any
import math

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


def optimize_akarin_expr(expr: str) -> str:
    """
    Optimize akarin.Expr expressions:
    1. Constant folding
    2. Convert dynamic pixel access to static when possible

    Args:
        expr: Input expr.

    Returns:
        Optimized expr.
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
    if not expr:
        return []
    # Use placeholders for pixel access to prevent splitting inside brackets
    placeholders = {}
    placeholder_prefix = "__PXACCESS"
    placeholder_suffix = "__"
    count = 0

    def repl(matchobj):
        nonlocal count
        key = f"{placeholder_prefix}{count}{placeholder_suffix}"
        placeholders[key] = matchobj.group(0)  # Store the original full match
        count += 1
        return key

    expr_with_placeholders = token_pattern.sub(repl, expr)

    # Split by whitespace
    raw_tokens = split_pattern.split(expr_with_placeholders)

    # Restore placeholders and filter empty tokens
    tokens = []
    for token in raw_tokens:
        if token in placeholders:
            tokens.append(placeholders[token])
        elif token:  # Filter out empty strings resulting from multiple spaces
            tokens.append(token)

    return tokens


def parse_numeric(token: str) -> Union[int, float]:
    """Parse a numeric token string to its actual value (int or float)."""
    if not is_token_numeric(token):
        raise ValueError(f"Token '{token}' is not a valid numeric format for parsing.")

    if hex_pattern.match(token):  # Hexadecimal
        if "." in token or "p" in token.lower():
            # Attempt parsing hex float
            try:
                return float.fromhex(token)
            except ValueError:
                # Fallback for complex hex patterns if fromhex fails (though less likely needed now)
                parts = hex_parts_pattern.match(token.lower())
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
                        f"Could not parse complex hex token: {token}"
                    )  # Should not happen
        else:
            return int(token, 16)  # Simple hex integer
    elif (
        octal_pattern.match(token)
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
            # This should not be reached if is_token_numeric passed
            raise ValueError(
                f"Internal error: Could not parse supposedly numeric token: {token}"
            )


def calculate_unary(op: str, a: Union[int, float]) -> Optional[Union[int, float]]:
    """Calculate result of unary operation"""
    operators = {
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "abs": abs,
        "not": lambda x: 0.0 if float(x) > 0.0 else 1.0,
        "bitnot": lambda x: ~int(x),  # Integer specific
        "trunc": math.trunc,  # Returns int
        "round": round,  # Returns int if possible
        "floor": math.floor,  # Returns float
    }
    if op not in operators:
        return None

    try:
        arg = a
        # Ensure correct type for the operation
        if op in ["exp", "log", "sqrt", "sin", "cos", "not", "floor"]:
            arg = float(a)
        elif op == "bitnot":
            # Check if float is actually an integer before converting
            if isinstance(a, float):
                if not a.is_integer():
                    return None  # Cannot bitwise-not a non-integer float
                arg = int(a)
            else:  # Already int
                arg = int(a)
        # For abs, trunc, round, Python handles int/float input okay

        result = operators[op](arg)

        # Attempt to return int if result is numerically an integer where appropriate
        if (
            isinstance(result, float)
            and result.is_integer()
            and op in ["abs", "trunc", "round", "floor", "bitnot"]
        ):
            return int(result)
        return result
    except (ValueError, OverflowError, TypeError):
        return None


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
        "bitand": lambda x, y: int(x) & int(y),
        "bitor": lambda x, y: int(x) | int(y),
        "bitxor": lambda x, y: int(x) ^ int(y),
    }
    if op not in operators:
        return None

    try:
        arg1, arg2 = a, b
        # Ensure correct types for specific operations
        if op.startswith("bit") or op == "%":
            # Check if floats are integers before converting
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

        # Attempt int conversion for appropriate ops if result is integer
        if (
            isinstance(result, float)
            and result.is_integer()
            and op in ["+", "-", "*", "%", "max", "min", "bitand", "bitor", "bitxor"]
        ):
            return int(result)
        # Boolean results are always float 1.0/0.0
        return result
    except (ZeroDivisionError, ValueError, OverflowError, TypeError):
        return None


def calculate_ternary(
    cond: Union[int, float], true_val: Union[int, float], false_val: Union[int, float]
) -> Union[int, float]:
    """Calculate result of ternary operation"""
    return true_val if float(cond) > 0 else false_val


def format_number(num: Union[int, float]) -> str:
    """Format number back to string representation for expression (more robust)."""
    if isinstance(num, int):
        return str(num)
    elif isinstance(num, float):
        formatted = f"{num:g}"

        if "E" in formatted:
            formatted = formatted.replace("E", "e")

        return formatted
    else:
        # Should not happen
        return str(num)  # Fallback


def fold_constants(expr: str) -> str:
    """Perform constant folding optimization"""
    tokens = tokenize_expr(expr)
    # stack stores actual values (numbers) or None for non-constants/unknowns during evaluation
    stack: list[Any] = []
    # result_tokens stores the list of tokens for the potentially optimized expression
    result_tokens: list[str] = []
    # Tracks known constant variable values *during this pass*
    variable_values: dict[str, Union[int, float, None]] = {}  # Reset each pass

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # 1. Handle Numeric Constants
        if is_token_numeric(token):
            try:
                value = parse_numeric(token)
                stack.append(value)
                result_tokens.append(token)  # Append the original token string
            except ValueError:
                # Should not happen
                stack.append(None)  # Treat as unknown if parsing fails
                result_tokens.append(token)
            i += 1
            continue

        # 2. Handle Variable Store (!)
        # Store consumes one value from stack, updates variable map, adds '!' token.
        is_store = (
            token.endswith("!")
            and len(token) > 1
            and not token.startswith("[")
            and not token_pattern.match(token)
        )
        if is_store:
            var_name = token[:-1]
            if not stack:
                raise ValueError(f"Stack underflow at store '{token}'")

            value_to_store = stack.pop()  # Get value from evaluation stack
            # Store value (or None) associated with var_name for this pass
            variable_values[var_name] = (
                value_to_store if isinstance(value_to_store, (int, float)) else None
            )

            # Store operation only adds its own token. It doesn't remove the token
            # corresponding to the value popped from the stack. That token remains.
            result_tokens.append(token)
            i += 1
            continue

        # 3. Handle Variable Load (@)
        # Load checks variable map. If constant, pushes value to stack and adds value token.
        # If not constant, pushes None to stack and adds '@' token.
        is_load = (
            token.endswith("@")
            and len(token) > 1
            and not token.startswith("[")
            and not token_pattern.match(token)
        )
        if is_load:
            var_name = token[:-1]
            constant_value = variable_values.get(
                var_name
            )  # Check if known constant in this pass

            if isinstance(constant_value, (int, float)):
                stack.append(constant_value)
                # Replace the load token (@) with the constant value's token string
                result_tokens.append(format_number(constant_value))
            else:
                # Variable value not known or not constant, result is unknown
                stack.append(None)
                result_tokens.append(token)  # Keep the original load token 'x@'
            i += 1
            continue

        # --- Operator Folding ---
        # General strategy: Check stack values AND corresponding result tokens.

        # 4. Handle Unary Operators
        unary_ops = {
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
        if token in unary_ops:
            can_fold = False
            if stack and result_tokens:  # Need operand on stack and its token in result
                op1_stack_val = stack[-1]
                op1_token = result_tokens[-1]

                # Check if stack value is number AND token is numeric string
                if isinstance(op1_stack_val, (int, float)) and is_token_numeric(
                    op1_token
                ):
                    result = calculate_unary(token, op1_stack_val)
                    if result is not None:
                        # Folded successfully! Update stack and result_tokens.
                        stack.pop()
                        stack.append(result)
                        result_tokens.pop()  # Remove operand token
                        result_tokens.append(
                            format_number(result)
                        )  # Append result token
                        can_fold = True

            if not can_fold:
                # Cannot fold: Pop stack operand (if any), push None, append operator token.
                if stack:
                    stack.pop()  # Consume the operand value from stack
                stack.append(None)  # Result is unknown
                result_tokens.append(token)  # Keep the operator token

            i += 1
            continue

        # 5. Handle Binary Operators
        BINARY_OPS = {
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
        if token in BINARY_OPS:
            can_fold = False
            if len(stack) >= 2 and len(result_tokens) >= 2:
                op2_stack_val = stack[-1]
                op1_stack_val = stack[-2]
                op2_token = result_tokens[-1]
                op1_token = result_tokens[-2]

                # Check stack values AND tokens
                if (
                    isinstance(op1_stack_val, (int, float))
                    and isinstance(op2_stack_val, (int, float))
                    and is_token_numeric(op1_token)
                    and is_token_numeric(op2_token)
                ):
                    result = calculate_binary(token, op1_stack_val, op2_stack_val)
                    if result is not None:
                        # Folded successfully!
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
                # Cannot fold: Pop stack operands (if any), push None, append operator token.
                if len(stack) >= 2:
                    stack.pop()
                    stack.pop()
                elif len(stack) == 1:
                    stack.pop()
                stack.append(None)  # Result is unknown
                result_tokens.append(token)  # Keep the operator token

            i += 1
            continue

        # 6. Handle Ternary Operator (?)
        if token == "?":
            can_fold = False
            if len(stack) >= 3 and len(result_tokens) >= 3:
                false_val_stack = stack[-1]
                true_val_stack = stack[-2]
                cond_stack = stack[-3]
                false_token = result_tokens[-1]
                true_token = result_tokens[-2]
                cond_token = result_tokens[-3]

                # Check stack values AND tokens
                if (
                    isinstance(cond_stack, (int, float))
                    and isinstance(true_val_stack, (int, float))
                    and isinstance(false_val_stack, (int, float))
                    and is_token_numeric(cond_token)
                    and is_token_numeric(true_token)
                    and is_token_numeric(false_token)
                ):
                    # Note: calculate_ternary itself doesn't fail easily if inputs are numbers
                    result = calculate_ternary(
                        cond_stack, true_val_stack, false_val_stack
                    )
                    # Folded successfully!
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
                # Cannot fold
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

        # 7. Handle clamp/clip operators
        if token in {"clamp", "clip"}:
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
                    # Clamp logic assumes value_val_stack is the value to clamp
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

        # --- Stack Manipulation Ops (No Folding, Just Stack Simulation and Token Append) ---
        # These ops inherently prevent folding across them because they add non-numeric tokens.

        # 8. Handle swapN
        if token.startswith("swap"):
            n = 1
            if len(token) > 4:
                try:
                    n = int(token[4:])
                except ValueError:
                    pass  # Treat as unknown token if N is invalid
            if n < 0:
                raise ValueError("Swap count cannot be negative")

            if len(stack) <= n:
                raise ValueError(f"Stack underflow for {token}")

            if n > 0:  # Simulate swap on evaluation stack
                stack[-1], stack[-(n + 1)] = stack[-(n + 1)], stack[-1]
            # Always keep the token
            result_tokens.append(token)
            i += 1
            continue

        # 9. Handle dupN
        if token.startswith("dup"):
            n = 0
            if len(token) > 3:
                try:
                    n = int(token[3:])
                except ValueError:
                    pass
            if n < 0:
                raise ValueError("Dup index cannot be negative")

            if len(stack) <= n:
                raise ValueError(f"Stack underflow for {token}")
            # Simulate dup on evaluation stack
            stack.append(stack[-(n + 1)])
            # Always keep the token
            result_tokens.append(token)
            i += 1
            continue

        # 10. Handle dropN
        if token.startswith("drop"):
            n = 1
            if len(token) > 4:
                try:
                    n = int(token[4:])
                except ValueError:
                    pass
            if n < 0:
                raise ValueError("Drop count cannot be negative")
            if n == 0:  # drop0 is no-op, just keep token
                pass
            elif len(stack) < n:
                raise ValueError(f"Stack underflow for {token}")
            else:  # Simulate drop on evaluation stack
                del stack[-n:]
            # Always keep the token
            result_tokens.append(token)
            i += 1
            continue

        # 11. Handle sortN
        if token.startswith("sort"):
            n = 0
            if len(token) > 4:
                try:
                    n = int(token[4:])
                except ValueError:
                    pass
            if n <= 0:
                raise ValueError("Sort count must be positive")
            if len(stack) < n:
                raise ValueError(f"Stack underflow for {token}")
            # Simulate sort effect: Removes N items, pushes N unknown results
            del stack[-n:]
            for _ in range(n):
                stack.append(None)
            # Always keep the token
            result_tokens.append(token)
            i += 1
            continue

        # --- Pixel Access (Treated as Non-foldable Operations) ---

        # 12. Handle Dynamic Access like `clip[]`
        # Consumes 2 stack items (coords), pushes None, keeps token.
        is_dynamic_access = (
            token.endswith("[]")
            and len(token) > 2
            and not token.startswith("[")
            and not token_pattern.match(token)
            and not is_token_numeric(token)
        )  # Heuristic check
        if is_dynamic_access:
            if len(stack) < 2:
                raise ValueError(f"Stack underflow for dynamic access '{token}'")
            stack.pop()
            stack.pop()  # Consume y, x coords from stack
            stack.append(None)  # Result is unknown
            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        # 13. Handle Static Access like `clip[1,2]`
        # Pushes None, keeps token.
        match = token_pattern.match(token)
        if match:
            stack.append(None)  # Result is unknown during folding pass
            result_tokens.append(token)  # Keep the token
            i += 1
            continue

        # Default: Unknown/Other Tokens
        # Assume it pushes one unknown result onto the stack and keep the token.
        # This includes unrecognized functions, malformed tokens, etc.
        stack.append(None)
        result_tokens.append(token)
        i += 1

    return " ".join(result_tokens)


def convert_dynamic_to_static(expr: str) -> str:
    """Convert dynamic pixel access to static when possible"""
    tokens = tokenize_expr(expr)  # Re-tokenize the folded expression
    if not tokens:
        return ""

    result_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        converted = False

        # Check for dynamic pixel access pattern (clip[])
        is_dynamic_access = (
            token.endswith("[]")
            and len(token) > 2
            and not token.startswith("[")
            and not token_pattern.match(token)
            and not is_token_numeric(token)
        )  # Heuristic check

        if is_dynamic_access and len(result_tokens) >= 2:
            # Check if the last two tokens in the result list being built are integer constants
            # TODO: convert float to int; need to check akarin.Expr code
            y_token = result_tokens[-1]
            x_token = result_tokens[-2]

            if is_token_numeric(x_token) and is_token_numeric(y_token):
                try:
                    y_val = parse_numeric(y_token)
                    x_val = parse_numeric(x_token)

                    # Check if they are effectively integers
                    is_x_int = isinstance(x_val, int) or (
                        isinstance(x_val, float) and x_val.is_integer()
                    )
                    is_y_int = isinstance(y_val, int) or (
                        isinstance(y_val, float) and y_val.is_integer()
                    )

                    if is_x_int and is_y_int:
                        x_int = int(x_val)
                        y_int = int(y_val)

                        # Perform conversion
                        clip_identifier = token[:-2]  # Get clip name
                        # Handle potential :c or :m suffix
                        suffix = ""
                        if ":" in clip_identifier:
                            parts = clip_identifier.split(":", 1)
                            clip_identifier = parts[0]
                            suffix = ":" + parts[1]

                        # Pop the constant coordinate tokens from result_tokens
                        result_tokens.pop()  # Remove y token
                        result_tokens.pop()  # Remove x token

                        # Append the new static access token
                        result_tokens.append(
                            f"{clip_identifier}[{x_int},{y_int}]{suffix}"
                        )
                        converted = True

                except (
                    ValueError,
                    OverflowError,
                ):  # Handle parse errors or large floats
                    # If parsing or int conversion fails, don't convert
                    pass  # Fall through to append original token

        if not converted:
            # If not converting this token, just append it
            result_tokens.append(token)

        i += 1  # Move to the next token in the *original* list

    return " ".join(result_tokens)
