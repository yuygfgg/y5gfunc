import regex as re


def verify_std_expr(expr: str) -> bool:
    """
    Verify if an expression is valid in std.Expr.
    @see: https://www.vapoursynth.com/doc/functions/video/expr.html
    """
    try:
        expr = expr.strip()
        if not expr:
            # Empty string is a valid expression (plane copy)
            return True

        tokens = re.split(r"\\s+", expr)

        stack_size = 0

        # This regex is from y5gfunc/expr/postfix2infix.py
        number_pattern = re.compile(
            r"^("
            r"0x[0-9A-Fa-f]+(\\.[0-9A-Fa-f]+(p[+\\-]?\\d+)?)?"
            r"|"
            r"0[0-7]*"
            r"|"
            r"[+\\-]?(\\d+(\\.\\d+)?([eE][+\\-]?\\d+)?)"
            r")$"
        )

        one_arg_ops = {"exp", "log", "sqrt", "sin", "cos", "abs", "not"}
        two_arg_ops = {
            "+",
            "-",
            "*",
            "/",
            "max",
            "min",
            "pow",
            ">",
            "<",
            "=",
            ">=",
            "<=",
            "and",
            "or",
            "xor",
        }
        three_arg_ops = {"?"}

        for token in tokens:
            if number_pattern.match(token):
                stack_size += 1
                continue

            if token.isalpha() and len(token) == 1 and "a" <= token <= "z":
                stack_size += 1
                continue

            if token in one_arg_ops:
                if stack_size < 1:
                    return False
                continue

            if token in two_arg_ops:
                if stack_size < 2:
                    return False
                stack_size -= 1
                continue

            if token in three_arg_ops:
                if stack_size < 3:
                    return False
                stack_size -= 2
                continue

            dup_match = re.match(r"^dup(\\d*)$", token)
            if dup_match:
                n = int(dup_match.group(1)) if dup_match.group(1) else 0
                if stack_size <= n:
                    return False
                stack_size += 1
                continue

            swap_match = re.match(r"^swap(\\d*)$", token)
            if swap_match:
                n = int(swap_match.group(1)) if swap_match.group(1) else 1
                if stack_size <= n:
                    return False
                continue

            return False

        return stack_size == 1
    except Exception:
        return False
