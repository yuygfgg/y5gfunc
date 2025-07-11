from ....postfix2infix import postfix2infix

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