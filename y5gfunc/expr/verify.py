from .postfix2infix import postfix2infix
import regex as re
from .utils import tokenize_expr, _DUP_PATTERN, _SWAP_PATTERN


def verify_akarin_expr(expr: str) -> bool:
    """
    Verify if an expression is valid in akarin.Expr.
    """
    try:
        postfix2infix(expr, check_mode=True)
        return True
    except Exception as e:
        print(str(e).replace("postfix2infix", "verify_akarin_expr"))
        return False


def verify_std_expr(expr: str) -> bool:
    """
    Verify if an expression is valid in std.Expr.
    """
    try:
        expr = expr.strip()
        if not expr:
            return True

        tokens = tokenize_expr(expr)

        stack_size = 0

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

        number_pattern = re.compile(
            r"^[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)$",  # Decimal and scientific notation only
        )

        for i, token in enumerate(tokens):
            if number_pattern.match(token):
                stack_size += 1
                continue

            if token.isalpha() and len(token) == 1 and "a" <= token <= "z":
                stack_size += 1
                continue

            if token in one_arg_ops:
                if stack_size < 1:
                    raise ValueError(
                        f"{i}th token '{token}' requires 1 argument, but stack has {stack_size}."
                    )
                continue

            if token in two_arg_ops:
                if stack_size < 2:
                    raise ValueError(
                        f"{i}th token '{token}' requires 2 arguments, but stack has {stack_size}."
                    )
                stack_size -= 1
                continue

            if token in three_arg_ops:
                if stack_size < 3:
                    raise ValueError(
                        f"{i}th token '{token}' requires 3 arguments, but stack has {stack_size}."
                    )
                stack_size -= 2
                continue

            dup_match = _DUP_PATTERN.match(token)
            if dup_match:
                n = int(dup_match.group(1)) if dup_match.group(1) else 0
                if stack_size <= n:
                    raise ValueError(
                        f"{i}th token '{token}' needs to duplicate the {n}-th element, but stack size is only {stack_size}."
                    )
                stack_size += 1
                continue

            swap_match = _SWAP_PATTERN.match(token)
            if swap_match:
                n = int(swap_match.group(1)) if swap_match.group(1) else 1
                if stack_size <= n:
                    raise ValueError(
                        f"{i}th token '{token}' needs to swap with the {n}-th element, but stack size is only {stack_size}."
                    )
                continue

            raise ValueError(f"{i}th token '{token}' is unknown.")

        if stack_size != 1:
            raise ValueError(
                f"Expression left {stack_size} items on the stack, but 1 was expected."
            )
        return True
    except Exception as e:
        print(f"verify_std_expr: {e}")
        return False
