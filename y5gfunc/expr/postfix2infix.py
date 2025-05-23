import sys

if sys.version_info >= (3, 11):
    from typing import LiteralString
else:
    LiteralString = str
import re


# inspired by mvf.postfix2infix
def postfix2infix(expr: str) -> LiteralString:
    """
    Convert postfix expr to infix code

    Args:
        expr: Input postfix expr.

    Returns:
        Converted infix code.

    Raises:
        ValueError: If an error was found in the input expr.
    """
    # Preprocessing
    expr = expr.strip()
    expr = re.sub(r"\[\s*(\w+)\s*,\s*(\w+)\s*\]", r"[\1,\2]", expr)  # [x, y] => [x,y]
    tokens = re.split(r"\s+", expr)

    stack = []
    output_lines = []

    # Regex patterns for numbers
    number_pattern = re.compile(
        r"^("
        r"0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?"
        r"|"
        r"0[0-7]*"
        r"|"
        r"[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)"
        r")$"
    )

    i = 0
    while i < len(tokens):

        def pop(n=1):
            try:
                if n == 1:
                    return stack.pop()
                r = stack[-n:]
                del stack[-n:]
                return r
            except IndexError:
                raise ValueError(
                    f"postfix2infix: Stack Underflow at token at {i}th token {token}."
                )

        def push(item):
            stack.append(item)

        token = tokens[i]

        # Single letter
        if token.isalpha() and len(token) == 1:
            push(token)
            i += 1
            continue

        # Numbers
        if number_pattern.match(token):
            push(token)
            i += 1
            continue

        # Source clips (srcN)
        if re.match(r"^src\d+$", token):
            push(token)
            i += 1
            continue

        # Frame property
        if re.match(r"^[a-zA-Z]\w*\.[a-zA-Z]\w*$", token):
            push(token)
            i += 1
            continue

        # Dynamic pixel access
        if token.endswith("[]"):
            clip_identifier = token[:-2]
            absY = pop()
            absX = pop()
            push(f"dyn({clip_identifier}({absX}, {absY}))")
            i += 1
            continue

        # Static relative pixel access
        m = re.match(r"^([a-zA-Z]\w*)\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](\:\w+)?$", token)
        if m:
            clip_identifier = m.group(1)
            statX = int(m.group(2))
            statY = int(m.group(3))
            boundary_suffix = m.group(4)
            if boundary_suffix not in [None, ":c", ":m"]:
                raise ValueError(
                    f"postfix2infix: unknown boundary_suffix {boundary_suffix} at {i}th token {token}"
                )
            push(f"{clip_identifier}[{statX},{statY}]{boundary_suffix or ""}")
            i += 1
            continue

        # Variable operations
        var_store_match = re.match(r"^([a-zA-Z_]\w*)\!$", token)
        var_load_match = re.match(r"^([a-zA-Z_]\w*)\@$", token)
        if var_store_match:
            var_name = var_store_match.group(1)
            val = pop()
            output_lines.append(f"{var_name} = {val}")
            i += 1
            continue
        elif var_load_match:
            var_name = var_load_match.group(1)
            push(var_name)
            i += 1
            continue

        # Drop operations
        drop_match = re.match(r"^drop(\d*)$", token)
        if drop_match:
            num = int(drop_match.group(1)) if drop_match.group(1) else 1
            pop(num)
            i += 1
            continue

        # Sort operations
        sort_match = re.match(r"^sort(\d+)$", token)
        if sort_match:
            num = int(sort_match.group(1))
            items = pop(num)
            sorted_items_expr = f"nth_{{}}({', '.join(items)})"
            for idx in range(len(items)):
                push(sorted_items_expr.format(idx))
            i += 1
            continue

        # Duplicate operations
        dup_match = re.match(r"^dup(\d*)$", token)
        if dup_match:
            n = int(dup_match.group(1)) if dup_match.group(1) else 0
            if len(stack) <= n:
                raise ValueError(
                    f"postfix2infix: {i}th token {token} needs at least {n} values, while only {len(stack)} in stack."
                )
            else:
                push(stack[-1 - n])
            i += 1
            continue

        # Swap operations
        swap_match = re.match(r"^swap(\d*)$", token)
        if swap_match:
            n = int(swap_match.group(1)) if swap_match.group(1) else 1
            if len(stack) <= n:
                raise ValueError(
                    f"postfix2infix: {i}th token {token} needs at least {n} values, while only {len(stack)} in stack."
                )
            else:
                stack[-1], stack[-1 - n] = stack[-1 - n], stack[-1]
            i += 1
            continue

        # Special constants
        if token in ("N", "X", "Y", "width", "height"):
            constants = {
                "N": "current_frame_number",
                "X": "current_x",
                "Y": "current_y",
                "width": "current_width",
                "height": "current_height",
            }
            push(constants[token])
            i += 1
            continue

        # Unary operators
        if token in (
            "sin",
            "cos",
            "round",
            "trunc",
            "floor",
            "bitnot",
            "abs",
            "sqrt",
            "not",
        ):
            a = pop()
            if token == "not":
                push(f"(!({a}))")
            else:
                push(f"{token}({a})")
            i += 1
            continue

        # Binary operators
        if token in ("%", "**", "pow", "bitand", "bitor", "bitxor"):
            b = pop()
            a = pop()
            if token == "%":
                push(f"({a} % {b})")
            elif token in ("**", "pow"):
                push(f"pow({a}, {b})")
            elif token == "bitand":
                push(f"({a} & {b})")
            elif token == "bitor":
                push(f"({a} | {b})")
            elif token == "bitxor":
                push(f"({a} ^ {b})")
            i += 1
            continue

        # Basic arithmetic, comparison and logical operators
        if token in (
            "+",
            "-",
            "*",
            "/",
            "max",
            "min",
            ">",
            "<",
            ">=",
            "<=",
            "=",
            "and",
            "or",
            "xor",
        ):
            b = pop()
            a = pop()
            if token in ("max", "min"):
                push(f"{token}({a}, {b})")
            elif token == "and":
                push(f"({a} && {b})")
            elif token == "or":
                push(f"({a} || {b})")
            elif token == "xor":
                # (a || b) && !(a && b)
                # (a && !b) || (!a && b)
                push(f"(({a} && !{b}) || (!{a} && {b}))")
            elif token == "=":
                push(f"{a} == {b}")
            else:
                push(f"({a} {token} {b})")
            i += 1
            continue

        # Ternary operator
        if token == "?":
            false_val = pop()
            true_val = pop()
            cond = pop()
            push(f"({cond} ? {true_val} : {false_val})")
            i += 1
            continue
        if token == "clip" or token == "clamp":
            max = pop()
            min = pop()
            value = pop()
            push(f"(clamp({value}, {min}, {max}))")
            i += 1
            continue

        # Unknown tokens
        output_lines.append(f"# [Unknown token]: {token}  (Push as-is)")
        push(token)
        i += 1

    # Handle remaining stack items
    if len(stack) == 1:
        output_lines.append(f"RESULT = {stack[0]}")
        ret = "\n".join(output_lines)
        print(ret)
    else:
        for idx, item in enumerate(stack):
            output_lines.append(f"# stack[{idx}]: {item}")
        ret = "\n".join(output_lines)
        raise ValueError(
            f"postfix2infix: Invalid expression: the stack contains not exactly one value after evaluation. \n {ret}"
        )
    return ret
