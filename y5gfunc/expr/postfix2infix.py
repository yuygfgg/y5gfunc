from .utils import (
    tokenize_expr,
    _NUMBER_PATTERNS,
    _FRAME_PROP_PATTERN,
    _STATIC_PIXEL_PATTERN,
    _VAR_STORE_PATTERN,
    _VAR_LOAD_PATTERN,
    _DROP_PATTERN,
    _SORT_PATTERN,
    _DUP_PATTERN,
    _SWAP_PATTERN,
    is_clip_postfix,
    is_constant_postfix,
    ensure_akarin_clip_name,
)


# inspired by mvf.postfix2infix
def postfix2infix(expr: str, check_mode: bool = False) -> str:
    """
    Convert postfix expr to infix code
    If check_mode is True, it only checks for expression validity without building the result.

    Args:
        expr: Input postfix expr.
        check_mode: If True, only perform validation.

    Returns:
        Converted infix code. Or an empty string if check_mode is True and expr is valid.

    Raises:
        ValueError: If an error was found in the input expr.
    """
    # Preprocessing
    expr = expr.strip()
    tokens = tokenize_expr(expr)

    stack = []
    output_lines = []
    defined_vars = set()

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
            if not check_mode:
                stack.append(item)
            else:
                stack.append(None)

        token = tokens[i]

        # Constants
        if is_constant_postfix(token):
            push(f"${ensure_akarin_clip_name(token)}")
            i += 1
            continue

        # Numbers
        if any(pattern.match(token) for pattern in _NUMBER_PATTERNS):
            push(token)
            i += 1
            continue

        # Frame property
        if _FRAME_PROP_PATTERN.match(token):
            clip_name = token.split(".")[0]
            clip_name_akarin = ensure_akarin_clip_name(clip_name)
            push(f"{clip_name_akarin}.{token.split(".")[1]}")
            i += 1
            continue

        # Dynamic pixel access
        if token.endswith("[]"):
            clip_identifier = token[:-2]
            absY = pop()
            absX = pop()

            if not is_clip_postfix(clip_identifier):
                raise ValueError(
                    f"postfix2infix: {i}th token {token} expected a clip identifier, but got {clip_identifier}."
                )

            # Add $ prefix for clip identifiers
            clip_name = f"${clip_identifier}"
            push(f"dyn({clip_name},{absX},{absY})")
            i += 1
            continue

        # Static relative pixel access
        m = _STATIC_PIXEL_PATTERN.match(token)
        if m:
            clip_identifier = m.group(1)
            statX = int(m.group(2))
            statY = int(m.group(3))
            boundary_suffix = m.group(4)
            if boundary_suffix not in [None, ":c", ":m"]:
                raise ValueError(
                    f"postfix2infix: unknown boundary_suffix {boundary_suffix} at {i}th token {token}"
                )
            if not is_clip_postfix(clip_identifier):
                raise ValueError(
                    f"postfix2infix: {i}th token {token} expected a clip identifier, but got {clip_identifier}."
                )
            # Add $ prefix for clip identifiers
            clip_name = f"${clip_identifier}"
            push(f"{clip_name}[{statX},{statY}]{boundary_suffix or ""}")
            i += 1
            continue

        # Variable operations
        var_store_match = _VAR_STORE_PATTERN.match(token)
        var_load_match = _VAR_LOAD_PATTERN.match(token)
        if var_store_match:
            var_name = var_store_match.group(1)
            val = pop()
            if not check_mode:
                output_lines.append(f"{var_name} = {val}")
            defined_vars.add(var_name)
            i += 1
            continue
        elif var_load_match:
            var_name = var_load_match.group(1)
            if var_name not in defined_vars:
                raise ValueError(
                    f"postfix2infix: {i}th token {token} loads an undefined variable {var_name}."
                )
            push(var_name)
            i += 1
            continue

        # Drop operations
        drop_match = _DROP_PATTERN.match(token)
        if drop_match:
            num = int(drop_match.group(1)) if drop_match.group(1) else 1
            pop(num)
            i += 1
            continue

        # Sort operations
        sort_match = _SORT_PATTERN.match(token)
        if sort_match:
            num = int(sort_match.group(1))
            items = pop(num)
            sorted_items_expr = f"nth_{{}}({', '.join(items)})"
            for idx in range(len(items)):
                push(sorted_items_expr.format(len(items) - idx))
            i += 1
            continue

        # Duplicate operations
        dup_match = _DUP_PATTERN.match(token)
        if dup_match:
            n = int(dup_match.group(1)) if dup_match.group(1) else 0
            if len(stack) <= n:
                raise ValueError(
                    f"postfix2infix: {i}th token {token} needs at least {n+1} values, while only {len(stack)} in stack."
                )
            else:
                push(stack[-1 - n])
            i += 1
            continue

        # Swap operations
        swap_match = _SWAP_PATTERN.match(token)
        if swap_match:
            n = int(swap_match.group(1)) if swap_match.group(1) else 1
            if len(stack) <= n:
                raise ValueError(
                    f"postfix2infix: {i}th token {token} needs at least {n+1} values, while only {len(stack)} in stack."
                )
            else:
                stack[-1], stack[-1 - n] = stack[-1 - n], stack[-1]
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
            elif token == "bitnot":
                push(f"(~round({a}))")
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
                push(f"({a} ** {b})")
            elif token == "bitand":
                push(f"(round({a}) & round({b}))")
            elif token == "bitor":
                push(f"(round({a}) | round({b}))")
            elif token == "bitxor":
                push(f"(round({a}) ^ round({b}))")
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
                push(f"({a} == {b})")
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
        if not check_mode:
            output_lines.append(f"# [Unknown token]: {token}  (Push as-is)")
        push(token)
        i += 1

    # Handle remaining stack items
    if len(stack) == 1:
        if check_mode:
            return ""
        output_lines.append(f"RESULT = {stack[0]}")
        ret = "\n".join(output_lines)
    else:
        if check_mode:
            msg = f"postfix2infix: Invalid expression: the stack contains {len(stack)} value(s), but should contain exactly one."
        else:
            for idx, item in enumerate(stack):
                output_lines.append(f"# stack[{idx}]: {item}")
            ret = "\n".join(output_lines)
            msg = f"postfix2infix: Invalid expression: the stack contains not exactly one value after evaluation. \n {ret}"
        raise ValueError(msg)
    return ret
