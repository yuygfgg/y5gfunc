from enum import StrEnum
from .front_end import parse_infix_to_postfix
from .middle_end import optimize_akarin_expr
from .back_end.std import to_std_expr, verify_std_expr
from .back_end.akarin import verify_akarin_expr

class BackEnd(StrEnum):
    AKARIN = "akarin"
    STD = "std"

def compile(expr: str, back_end: BackEnd = BackEnd.AKARIN) -> str:
    """
    Convert an infix expression to a postfix expression using std.Expr operators.
    """
    postfix = parse_infix_to_postfix(expr)
    optimized = optimize_akarin_expr(postfix)
    if back_end == BackEnd.STD:
        optimized = to_std_expr(postfix)
        if not verify_std_expr(optimized):
            raise ValueError("Invalid expression")
    elif back_end == BackEnd.AKARIN:
        if not verify_akarin_expr(optimized):
            raise ValueError("Invalid expression")
    return optimized