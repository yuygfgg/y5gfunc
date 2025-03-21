import re
import sys

if sys.version_info >= (3, 11):
    from typing import LiteralString, Dict, List, Tuple, Optional, Set
else:
    from typing import Dict, List, Tuple, Optional, Set

    LiteralString = str


class SyntaxError(Exception):
    """Custom syntax error class with line information"""

    def __init__(self, message, line_num=None, function_name=None):
        self.line_num = line_num
        self.function_name = function_name
        if function_name and line_num is not None:
            super().__init__(
                f"Line {line_num}: In function '{function_name}', {message}"
            )
        elif line_num is not None:
            super().__init__(f"Line {line_num}: {message}")
        else:
            super().__init__(message)


def is_clip(token: str) -> bool:
    """
    Determine whether token is a source clip.
    (std.Expr: single letter; or akarin.Expr: srcN)
    """
    if re.fullmatch(r"[a-zA-Z]", token):
        return True
    if re.fullmatch(r"src\d+", token):
        return True
    return False


def is_constant(token: str) -> bool:
    """
    Check if the token is a built-in constant.
    """
    constants_set = {
        "N",
        "X",
        "current_x",
        "Y",
        "current_y",
        "width",
        "current_width",
        "height",
        "current_height",
    }
    if token in constants_set:
        return True
    if is_clip(token):
        return True
    return False


def strip_outer_parentheses(expr: str) -> str:
    """
    Remove outer parentheses if the entire expression is enclosed.
    """
    if not expr.startswith("(") or not expr.endswith(")"):
        return expr
    count = 0
    for i, char in enumerate(expr):
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        if count == 0 and i < len(expr) - 1:
            return expr
    return expr[1:-1]


def match_full_function_call(expr: str) -> Optional[Tuple[str, str]]:
    """
    Try to match a complete function call with proper nesting.
    Returns (func_name, args_str) if matched; otherwise returns None.
    """
    expr = expr.strip()
    m = re.match(r"(\w+)\s*\(", expr)
    if not m:
        return None
    func_name = m.group(1)
    start = m.end() - 1  # position of '('
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
            if depth == 0:
                if i == len(expr) - 1:
                    args_str = expr[start + 1 : i]
                    return func_name, args_str
                else:
                    return None
    return None


def extract_function_info(
    internal_var: str, current_function: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a renamed internal variable (e.g. __internal_funcname_varname),
    extract the original function name and variable name.
    """
    if current_function is not None and internal_var.startswith(
        f"__internal_{current_function}_"
    ):
        return current_function, internal_var[len(f"__internal_{current_function}_") :]
    match = re.match(r"__internal_([a-zA-Z_]\w*)_([a-zA-Z_]\w+)$", internal_var)
    if match:
        return match.group(1), match.group(2)
    return None, None


def expand_loops(code: str, base_line: int = 1) -> str:
    """
    Expand the loop syntax sugar.
    For example, the following:

        loop(2, i) { ... }

    is expanded into:

        <unrolled code for i=0>;<unrolled code for i=1>

    Newlines in the original loop block are preserved.
    Only replaces variables in <var> format, not standalone variables.
    """
    pattern = re.compile(r"loop\s*\(\s*(\d+)\s*,\s*([a-zA-Z_]\w*)\s*\)\s*\{")
    while True:
        m = pattern.search(code)
        if not m:
            break
        n = int(m.group(1))
        var = m.group(2)
        loop_start_index = m.start()
        # Calculate the line number by counting newlines before this match
        line_num = base_line + code[:loop_start_index].count("\n")
        if n < 1:
            raise SyntaxError(
                "Loop count {n} is not validated! Expected integer >= 1, got {n}!",
                line_num,
            )
        brace_start = m.end() - 1
        count = 0
        end_index = None
        for i in range(brace_start, len(code)):
            if code[i] == "{":
                count += 1
            elif code[i] == "}":
                count -= 1
                if count == 0:
                    end_index = i
                    break
        if end_index is None:
            raise SyntaxError("Could not find matching '}'.", line_num)
        # The block inside the loop braces
        block = code[brace_start + 1 : end_index]
        block_base_line = base_line + code[: brace_start + 1].count("\n")
        expanded_block = expand_loops(block, block_base_line)
        unrolled = []
        for k in range(n):
            # Replace only variables in <var> format, not standalone variables
            iter_block = re.sub(r"<" + re.escape(var) + r">", str(k), expanded_block)
            unrolled.append(iter_block)
        # Join unrolled fragments using semicolons.
        unrolled_code = "".join(map(lambda s: s.strip() + ";", unrolled)).strip(";")
        # Preserve the original newline count
        original_block = code[m.start() : end_index + 1]
        newline_placeholder_half = "\n" * (original_block.count("\n") // 2)
        replacement = (
            newline_placeholder_half + unrolled_code + newline_placeholder_half
        )
        code = code[: m.start()] + replacement + code[end_index + 1 :]
    return code


def find_duplicate_functions(code: str):
    """
    Check if any duplicate function is defined.
    """
    pattern = re.compile(r"\bfunction\s+(\w+)\s*\(.*?\)")
    function_lines = {}
    lines = code.split("\n")
    for line_num, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if match:
            func_name = match.group(1)
            if func_name in function_lines:
                raise SyntaxError(
                    f"Duplicated function '{func_name}' defined at lines {function_lines[func_name]} and {line_num}",
                    line_num=line_num,
                )
            function_lines[func_name] = line_num


def compute_stack_effect(
    postfix_expr: str, line_num: Optional[int] = None, func_name: Optional[str] = None
) -> int:
    """
    Compute the net stack effect of a postfix expression.
    Supports built-in operators, dropN (which removes N items), and sortN (which reorders
    the top N items without changing the stack count).
    """
    tokens = postfix_expr.split()
    op_arity = {
        "+": 2,
        "-": 2,
        "*": 2,
        "/": 2,
        "%": 2,
        "pow": 2,
        "and": 2,
        "or": 2,
        "=": 2,
        "<": 2,
        ">": 2,
        "<=": 2,
        ">=": 2,
        "bitand": 2,
        "bitor": 2,
        "bitxor": 2,
        "min": 2,
        "max": 2,
        "clamp": 3,
        "?": 3,
        "not": 1,
        "sin": 1,
        "cos": 1,
        "round": 1,
        "floor": 1,
        "abs": 1,
        "sqrt": 1,
        "trunc": 1,
        "bitnot": 1,
    }
    stack = []
    for i, token in enumerate(tokens):
        if token.endswith("!"):
            if not stack:
                raise SyntaxError("Stack underflow in assignment", line_num, func_name)
            stack.pop()
            continue
        elif token.endswith("[]"):  # dynamic access operator
            if len(stack) < 2:
                raise SyntaxError(
                    f"Stack underflow for operator {token} at token index {i}",
                    line_num,
                    func_name,
                )
            stack.pop()
            stack.pop()
            stack.append(1)
            continue

        # Support dropN operator (default is drop1)
        m_drop = re.fullmatch(r"drop(\d*)", token)
        if m_drop:
            n_str = m_drop.group(1)
            n = int(n_str) if n_str != "" else 1
            if len(stack) < n:
                raise SyntaxError(
                    f"Stack underflow for operator {token}", line_num, func_name
                )
            for _ in range(n):
                stack.pop()
            continue

        # Support sortN operator: reorder top N items without changing the stack count.
        m_sort = re.fullmatch(r"sort(\d+)", token)
        if m_sort:
            n = int(m_sort.group(1))
            if len(stack) < n:
                raise SyntaxError(
                    f"Stack underflow for operator {token}", line_num, func_name
                )
            # Sorting reorders items but does not change the stack count.
            continue

        if token in op_arity:
            arity = op_arity[token]
            if len(stack) < arity:
                raise SyntaxError(
                    f"Stack underflow for operator {token} at token index {i}",
                    line_num,
                    func_name,
                )
            for _ in range(arity):
                stack.pop()
            stack.append(1)
        else:
            stack.append(1)
    return len(stack)


def infix2postfix(infix_code: str) -> LiteralString:
    """
    Convert infix expressions to postfix expressions.
    Supports function definitions, function calls, built-in functions, nth_N operator (Nth smallest, N starts from 1), and loop syntax sugar.
    User input code must not contain semicolons.
    """
    # Reject user input that contains semicolons.
    if ";" in infix_code:
        raise SyntaxError(
            "User input code cannot contain semicolons. Use newlines instead."
        )

    # Check for duplicate function definitions.
    find_duplicate_functions(infix_code)

    # Expand loops; the top-level code begins at line 1.
    expanded_code = expand_loops(infix_code, base_line=1)

    # Process global declarations.
    # Support declaration of multiple globals in one line: <global<var1><var2>...>
    # Also record for functions if global declaration appears immediately before function definition.
    declared_globals: Dict[
        str, int
    ] = {}  # mapping global variable -> declaration line number
    global_vars_for_functions = {}
    lines = expanded_code.split("\n")
    modified_lines = []

    # Build a mapping from line number to function name for function declarations.
    function_lines = {}
    for i, line in enumerate(lines):
        func_match = re.match(r"function\s+(\w+)", line.strip())
        if func_match:
            func_name = func_match.group(1)
            function_lines[i] = func_name

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for global declaration syntax: e.g., <global<var1><var2>...>
        global_decl_match = re.match(r"^<global((?:<[a-zA-Z_]\w*>)+)>$", line)
        if global_decl_match:
            # Extract all global variable names from this line.
            globals_list = re.findall(r"<([a-zA-Z_]\w*)>", global_decl_match.group(1))
            for gv in globals_list:
                if gv not in declared_globals:
                    declared_globals[gv] = i + 1  # store line number (1-indexed)
            # Group consecutive global declaration lines.
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                next_decl_match = re.match(
                    r"^<global((?:<[a-zA-Z_]\w*>)+)>$", next_line
                )
                if next_decl_match:
                    extra_globals = re.findall(
                        r"<([a-zA-Z_]\w*)>", next_decl_match.group(1)
                    )
                    for gv in extra_globals:
                        if gv not in declared_globals:
                            declared_globals[gv] = j + 1
                    j += 1
                else:
                    break
            # If the next non-global line is a function definition, apply these globals for that function.
            if j < len(lines):
                func_match = re.match(r"^function\s+(\w+)", lines[j])
                if func_match:
                    func_name = func_match.group(1)
                    if func_name not in global_vars_for_functions:
                        global_vars_for_functions[func_name] = set()
                    global_vars_for_functions[func_name].update(globals_list)
            # Comment out the global declaration lines.
            for k in range(i, j):
                modified_lines.append("# " + lines[k])
            i = j
        else:
            modified_lines.append(lines[i])
            i += 1

    expanded_code = "\n".join(modified_lines)

    # Replace function definitions with newlines to preserve line numbers.
    functions: Dict[str, Tuple[List[str], str, int, Set[str]]] = {}
    function_pattern = (
        r"function\s+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    )

    def replace_function(match: re.Match) -> str:
        func_name = match.group(1)
        params_str = match.group(2)
        body = match.group(3)
        func_start_index = match.start()
        line_num = 1 + expanded_code[:func_start_index].count("\n")
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        if is_builtin_function(func_name):
            raise SyntaxError(
                f"Function name '{func_name}' conflicts with built-in functions!",
                line_num,
            )
        for param in params:
            if param.startswith("__internal_"):
                raise SyntaxError(
                    f"Parameter name '{param}' cannot start with '__internal_' (reserved prefix)",
                    line_num,
                )
        # Get globals declared for this function if any.
        global_vars = global_vars_for_functions.get(func_name, set())
        functions[func_name] = (params, body, line_num, global_vars)
        return "\n" * match.group(0).count("\n")

    cleaned_code = re.sub(function_pattern, replace_function, expanded_code)

    # Process global code by splitting on physical newlines.
    global_statements = []
    for i, line in enumerate(cleaned_code.splitlines(), start=1):
        if line.strip():
            global_statements.append((line.strip(), i))

    variables: Set[str] = set()  # Global variables defined via assignment
    postfix_tokens = []

    for stmt, line_num in global_statements:
        # Skip comment lines.
        if stmt.startswith("#"):
            continue
        # Process assignment statements.
        if "=" in stmt and not re.search(r"[<>!]=|==", stmt):
            var_name, expr = stmt.split("=", 1)
            var_name = var_name.strip()
            if var_name.startswith("__internal_"):
                raise SyntaxError(
                    f"Variable name '{var_name}' cannot start with '__internal_' (reserved prefix)",
                    line_num,
                )
            if is_constant(var_name):
                raise SyntaxError(f"Cannot assign to constant '{var_name}'.", line_num)
            expr = expr.strip()
            if var_name not in variables and re.search(
                r"\b" + re.escape(var_name) + r"\b", expr
            ):
                raise SyntaxError(
                    f"Variable '{var_name}' used before definition", line_num
                )
            variables.add(var_name)
            expr_postfix = convert_expr(expr, variables, functions, line_num)
            postfix_tokens.append(f"{expr_postfix} {var_name}!")
        else:
            expr_postfix = convert_expr(stmt, variables, functions, line_num)
            if compute_stack_effect(expr_postfix, line_num) != 0:
                raise SyntaxError(f"Unused global expression: {stmt}", line_num)
            postfix_tokens.append(expr_postfix)

    # Check that all declared global variables are defined.
    for gv, decl_line in declared_globals.items():
        if gv not in variables:
            raise SyntaxError(
                f"Global variable '{gv}' declared but not defined.", decl_line
            )

    final_result = " ".join(postfix_tokens)
    final_result = final_result.replace("(", "").replace(")", "")
    if "RESULT!" not in final_result:
        raise SyntaxError("Final result must be assigned to variable 'RESULT!")
    return final_result + " RESULT@"


def validate_static_relative_pixel_indices(
    expr: str, line_num: int, function_name: Optional[str] = None
) -> None:
    """
    Validate that the static relative pixel access indices are numeric constants.
    """
    static_relative_pixel_indices = re.finditer(r"\w+\[(.*?)\]", expr)
    for match in static_relative_pixel_indices:
        indices = match.group(1).split(",")
        for idx in indices:
            if not re.fullmatch(r"-?\d+", idx.strip()):
                raise SyntaxError(
                    f"Static relative pixel access index must be a numeric constant, got '{idx.strip()}'",
                    line_num,
                    function_name,
                )


def check_variable_usage(
    expr: str,
    variables: Set[str],
    line_num: int,
    function_name: Optional[str] = None,
    local_vars: Optional[Set[str]] = None,
) -> None:
    """
    Check that all variables in the expression have been defined.
    In function scope, if local_vars is provided, only local variables are checked.
    """
    expr_no_funcs = re.sub(r"\w+\([^)]*\)", "", expr)
    expr_no_stat_rels = re.sub(r"\[[^\]]*\]", "", expr_no_funcs)
    identifiers = re.finditer(r"\b([a-zA-Z_]\w*)\b", expr_no_stat_rels)
    for match in identifiers:
        var_name = match.group(1)
        if is_constant(var_name) or (
            (local_vars is not None and var_name in local_vars)
            or (local_vars is None and var_name in variables)
        ):
            continue
        if var_name.startswith("__internal_"):
            func_name_extracted, orig_var = extract_function_info(
                var_name, function_name
            )
            if local_vars is not None and orig_var in local_vars:
                continue
            raise SyntaxError(
                f"Variable '{orig_var}' used before definition",
                line_num,
                func_name_extracted,
            )
        if not re.search(rf"{re.escape(var_name)}\s*=", expr):
            raise SyntaxError(
                f"Variable '{var_name}' used before definition", line_num, function_name
            )


def convert_expr(
    expr: str,
    variables: Set[str],
    functions: Dict[str, Tuple[List[str], str, int, Set[str]]],
    line_num: int,
    current_function: Optional[str] = None,
    local_vars: Optional[Set[str]] = None,
) -> str:
    """
    Convert a single infix expression to a postfix expression.
    Supports binary and unary operators, function calls, and custom function definitions.
    """
    expr = expr.strip()
    # Check variable usage and validate static relative pixel indices.
    check_variable_usage(expr, variables, line_num, current_function, local_vars)
    validate_static_relative_pixel_indices(expr, line_num, current_function)

    number_pattern = re.compile(
        r"^("
        r"0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?"
        r"|"
        r"0[0-7]*"
        r"|"
        r"[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)"
        r")$"
    )
    if number_pattern.match(expr):
        return expr

    constants_map = {
        "current_frame_number": "N",
        "current_x": "X",
        "current_y": "Y",
        "current_width": "width",
        "current_height": "height",
    }
    if expr in constants_map:
        return constants_map[expr]

    func_call_full = match_full_function_call(expr)
    if func_call_full:
        func_name, args_str = func_call_full
        if func_name not in functions and not is_builtin_function(func_name):
            raise SyntaxError(
                f"Undefined function '{func_name}'", line_num, current_function
            )
        m_nth = re.match(r"^nth_(\d+)$", func_name)
        if m_nth:
            N_val = int(m_nth.group(1))
            args = parse_args(args_str)
            M = len(args)
            if M < N_val:
                raise SyntaxError(
                    f"nth_{N_val} requires at least {N_val} arguments, but only {M} were provided.",
                    line_num,
                    current_function,
                )
            if N_val < 1:
                raise SyntaxError(
                    f"nth_{N_val} is not supported!",
                    line_num,
                    current_function,
                )
            args_postfix = [
                convert_expr(
                    arg, variables, functions, line_num, current_function, local_vars
                )
                for arg in args
            ]
            sort_token = f"sort{M}"
            drop_before = f"drop{N_val-1}" if (N_val - 1) > 0 else ""
            drop_after = f"drop{M-N_val}" if (M - N_val) > 0 else ""
            tokens = []
            tokens.append(" ".join(args_postfix))
            tokens.append(sort_token)
            if drop_before:
                tokens.append(drop_before)
            tokens.append("__internal_nth_tmp!")
            if drop_after:
                tokens.append(drop_after)
            tokens.append("__internal_nth_tmp@")
            return " ".join(tokens).strip()

        args = parse_args(args_str)
        args_postfix = [
            convert_expr(
                arg, variables, functions, line_num, current_function, local_vars
            )
            for arg in args
        ]
        if func_name in ["min", "max"] and len(args) == 2:
            return f"{args_postfix[0]} {args_postfix[1]} {func_name}"
        elif func_name == "clamp" and len(args) == 3:
            return f"{args_postfix[0]} {args_postfix[1]} {args_postfix[2]} clamp"
        else:
            builtin_unary = [
                "sin",
                "cos",
                "round",
                "floor",
                "abs",
                "sqrt",
                "trunc",
                "bitnot",
                "not",
            ]
            if func_name in builtin_unary and len(args) == 1:
                return f"{args_postfix[0]} {func_name}"
        # Handle custom function calls.
        if func_name in functions:
            params, body, func_line_num, global_vars = functions[func_name]
            if len(args) != len(params):
                raise SyntaxError(
                    f"Function {func_name} requires {len(params)} parameters, but {len(args)} were provided.",
                    line_num,
                    current_function,
                )
            # Rename parameters: __internal_<funcname>_<varname>
            param_map = {p: f"__internal_{func_name}_{p}" for p in params}
            body_lines = [line.strip() for line in body.split("\n")]
            body_lines_strip = [
                line.strip() for line in body.split("\n") if line.strip()
            ]
            return_indices = [
                i
                for i, line in enumerate(body_lines_strip)
                if line.startswith("return")
            ]
            if return_indices and return_indices[0] != len(body_lines_strip) - 1:
                raise SyntaxError(
                    f"Return statement must be the last line in function '{func_name}'",
                    func_line_num,
                    func_name,
                )

            local_map = {}
            for offset, body_line in enumerate(body_lines):
                if body_line.startswith("return"):
                    continue
                m_line = re.match(r"^([a-zA-Z_]\w*)\s*=\s*(.+)$", body_line)
                if m_line:
                    var = m_line.group(1)
                    if var.startswith("__internal_"):
                        raise SyntaxError(
                            f"Variable name '{var}' cannot start with '__internal_' (reserved prefix)",
                            func_line_num + offset,
                            func_name,
                        )
                    if is_constant(var):
                        raise SyntaxError(
                            f"Cannot assign to constant '{var}'.",
                            func_line_num + offset,
                            func_name,
                        )
                    # Only rename & map local variables
                    if (
                        var not in param_map
                        and not var.startswith(f"__internal_{func_name}_")
                        and var not in global_vars
                    ):
                        local_map[var] = f"__internal_{func_name}_{var}"

            rename_map = {}
            rename_map.update(param_map)
            rename_map.update(local_map)
            new_local_vars = (
                set(param_map.keys()).union(set(local_map.keys())).union(global_vars)
            )
            param_assignments = []
            for i, p in enumerate(params):
                arg_orig = args[i].strip()
                if is_constant(arg_orig):
                    rename_map[p] = args_postfix[i]
                else:
                    if p not in global_vars:
                        param_assignments.append(f"{args_postfix[i]} {rename_map[p]}!")
            new_lines = []
            for line_text in body_lines:
                new_line = line_text
                for old, new in rename_map.items():
                    if old not in global_vars:
                        new_line = re.sub(
                            rf"(?<!\w){re.escape(old)}(?!\w)", new, new_line
                        )
                new_lines.append(new_line)
            function_tokens = []
            return_count = 0
            for offset, body_line in enumerate(new_lines):
                effective_line_num = func_line_num + offset
                if body_line.startswith("return"):
                    return_count += 1
                    ret_expr = body_line[len("return") :].strip().rstrip(";")
                    function_tokens.append(
                        convert_expr(
                            ret_expr,
                            variables,
                            functions,
                            effective_line_num,
                            func_name,
                            new_local_vars,
                        )
                    )
                # Process assignment statements with possible semicolons.
                elif "=" in body_line and not re.search(r"[<>!]=|==", body_line):
                    if ";" in body_line:
                        tokens_list = []
                        for sub in body_line.split(";"):
                            sub = sub.strip()
                            if not sub:
                                continue
                            if "=" in sub and not re.search(r"[<>!]=|==", sub):
                                var_name, expr_line = sub.split("=", 1)
                                var_name = var_name.strip()
                                if is_constant(var_name):
                                    raise SyntaxError(
                                        f"Cannot assign to constant '{var_name}'.",
                                        effective_line_num,
                                        func_name,
                                    )
                                if var_name not in new_local_vars and re.search(
                                    r"\b" + re.escape(var_name) + r"\b", expr_line
                                ):
                                    _, orig_var = extract_function_info(
                                        var_name, func_name
                                    )
                                    raise SyntaxError(
                                        f"Variable '{orig_var}' used before definition",
                                        effective_line_num,
                                        func_name,
                                    )
                                if var_name not in new_local_vars:
                                    new_local_vars.add(var_name)
                                tokens_list.append(
                                    f"{convert_expr(expr_line, variables, functions, effective_line_num, func_name, new_local_vars)} {var_name}!"
                                )
                            else:
                                tokens_list.append(
                                    convert_expr(
                                        sub,
                                        variables,
                                        functions,
                                        effective_line_num,
                                        func_name,
                                        new_local_vars,
                                    )
                                )
                        function_tokens.append(" ".join(tokens_list))
                    else:
                        var_name, expr_line = body_line.split("=", 1)
                        var_name = var_name.strip()
                        if is_constant(var_name):
                            raise SyntaxError(
                                f"Cannot assign to constant '{var_name}'.",
                                effective_line_num,
                                func_name,
                            )
                        if var_name not in new_local_vars and re.search(
                            r"\b" + re.escape(var_name) + r"\b", expr_line
                        ):
                            _, orig_var = extract_function_info(var_name, func_name)
                            raise SyntaxError(
                                f"Variable '{orig_var}' used before definition",
                                effective_line_num,
                                func_name,
                            )
                        if var_name not in new_local_vars:
                            new_local_vars.add(var_name)
                        function_tokens.append(
                            f"{convert_expr(expr_line, variables, functions, effective_line_num, func_name, new_local_vars)} {var_name}!"
                        )
                else:
                    function_tokens.append(
                        convert_expr(
                            body_line,
                            variables,
                            functions,
                            effective_line_num,
                            func_name,
                            new_local_vars,
                        )
                    )
            if return_count == 0 or return_count > 1:
                raise SyntaxError(
                    f"Function {func_name} must return exactly one value, got {return_count}",
                    func_line_num,
                    func_name,
                )
            result_expr = " ".join(param_assignments + function_tokens)
            net_effect = compute_stack_effect(result_expr, line_num, func_name)
            if net_effect != 1:
                raise SyntaxError(
                    f"The return value stack of function {func_name} is unbalanced; expected 1 but got {net_effect}.",
                    func_line_num,
                    func_name,
                )
            return result_expr

    stripped = strip_outer_parentheses(expr)
    if stripped != expr:
        return convert_expr(
            stripped, variables, functions, line_num, current_function, local_vars
        )

    m_static = re.match(r"^(\w+)\[(-?\d+),\s*(-?\d+)\](\:\w)?$", expr)
    if m_static:
        clip = m_static.group(1)
        statX = m_static.group(2)
        statY = m_static.group(3)
        suffix = m_static.group(4) or ""
        if not is_clip(clip):
            raise SyntaxError(f"'{clip}' is not a clip!", line_num, current_function)
        try:
            int(statX)
            int(statY)
        except ValueError:
            raise SyntaxError(
                f"Static relative pixel access indices must be integers, got '[{statX},{statY}]'",
                line_num,
                current_function,
            )
        return f"{clip}[{statX},{statY}]{suffix}"

    m_dyn = re.match(r"^(\w+)\.dyn\((.+),\s*(.+)\)$", expr)
    if m_dyn:
        clip = m_dyn.group(1)
        if not is_clip(clip):
            raise SyntaxError(f"'{clip}' is not a clip!", line_num, current_function)
        x_expr = convert_expr(
            m_dyn.group(2), variables, functions, line_num, current_function, local_vars
        )
        y_expr = convert_expr(
            m_dyn.group(3), variables, functions, line_num, current_function, local_vars
        )
        return f"{x_expr} {y_expr} {clip}[]"

    m_ternary = re.search(r"(.+?)\s*\?\s*(.+?)\s*:\s*(.+)", expr)
    if m_ternary:
        cond = convert_expr(
            m_ternary.group(1),
            variables,
            functions,
            line_num,
            current_function,
            local_vars,
        )
        true_expr = convert_expr(
            m_ternary.group(2),
            variables,
            functions,
            line_num,
            current_function,
            local_vars,
        )
        false_expr = convert_expr(
            m_ternary.group(3),
            variables,
            functions,
            line_num,
            current_function,
            local_vars,
        )
        return f"{cond} {true_expr} {false_expr} ?"

    operators = [
        ("||", "or"),
        ("&&", "and"),
        ("==", "="),
        ("!=", "!="),
        ("<=", "<="),
        (">=", ">="),
        ("<", "<"),
        (">", ">"),
        ("+", "+"),
        ("-", "-"),
        ("*", "*"),
        ("/", "/"),
        ("%", "%"),
        ("**", "pow"),
        ("&", "bitand"),
        ("|", "bitor"),
        ("^", "bitxor"),
    ]
    for op_str, postfix_op in operators:
        left, right = find_binary_op(expr, op_str)
        if left is not None and right is not None:
            left_postfix = convert_expr(
                left, variables, functions, line_num, current_function, local_vars
            )
            right_postfix = convert_expr(
                right, variables, functions, line_num, current_function, local_vars
            )
            if re.fullmatch(
                r"[a-zA-Z_]\w*", left.strip()
            ) and not left_postfix.strip().endswith("@"):
                left_postfix = left_postfix.strip() + (
                    "@" if not is_constant(left_postfix) else ""
                )
            if re.fullmatch(
                r"[a-zA-Z_]\w*", right.strip()
            ) and not right_postfix.strip().endswith("@"):
                right_postfix = right_postfix.strip() + (
                    "@" if not is_constant(right_postfix) else ""
                )
            return f"{left_postfix} {right_postfix} {postfix_op}"

    if expr.startswith("!"):
        operand = convert_expr(
            expr[1:], variables, functions, line_num, current_function, local_vars
        )
        return f"{operand} not"

    if is_constant(expr):
        return expr
    if re.fullmatch(r"[a-zA-Z_]\w*", expr):
        return expr if expr.endswith("@") else expr + "@"

    return expr


def find_binary_op(expr: str, op: str):
    """
    Find the last occurrence of the binary operator op at the outer level and return the left and right parts.
    """
    level = 0
    op_index = -1
    for i in range(len(expr)):
        c = expr[i]
        if c == "(":
            level += 1
        elif c == ")":
            level -= 1
        if level == 0 and expr[i : i + len(op)] == op:
            op_index = i
    if op_index == -1:
        return None, None
    left = expr[:op_index].strip()
    right = expr[op_index + len(op) :].strip()
    return left, right


def parse_args(args_str: str) -> List[str]:
    """
    Parse the arguments of a function call.
    """
    args = []
    current = []
    bracket_depth = 0
    square_bracket_depth = 0
    for c in args_str + ",":
        if c == "," and bracket_depth == 0 and square_bracket_depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(c)
            if c == "(":
                bracket_depth += 1
            elif c == ")":
                bracket_depth -= 1
            elif c == "[":
                square_bracket_depth += 1
            elif c == "]":
                square_bracket_depth -= 1
    return [arg for arg in args if arg]


def is_builtin_function(func_name: str) -> bool:
    """
    Check if the function name belongs to a built-in function.
    """
    builtin_unary = [
        "sin",
        "cos",
        "round",
        "floor",
        "abs",
        "sqrt",
        "trunc",
        "bitnot",
        "not",
    ]
    builtin_binary = ["min", "max"]
    builtin_ternary = ["clamp"]
    if any(
        re.match(r, func_name)
        for r in [
            rf"^{prefix}\d+$" for prefix in ["nth_", "sort", "dup", "drop", "swap"] # Only nth_N is literally 'built-in' but we consider stack Ops built-in as well, for no reason. Might be removed.
        ]
    ):
        return True
    return (
        func_name in builtin_unary
        or func_name in builtin_binary
        or func_name in builtin_ternary
    )
