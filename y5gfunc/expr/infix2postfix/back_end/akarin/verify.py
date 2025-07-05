from ....postfix2infix import postfix2infix

def verify_akarin_expr(expr: str) -> bool:
    """
    Verify if an expression is valid in akarin.Expr.
    """
    try:
        postfix2infix(expr)
        return True
    except Exception:
        return False