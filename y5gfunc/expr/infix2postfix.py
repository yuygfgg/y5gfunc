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
            super().__init__(f"Line {line_num}: In function '{function_name}', {message}")
        elif line_num is not None:
            super().__init__(f"Line {line_num}: {message}")
        else:
            super().__init__(message)

def is_clip(token: str) -> bool:
    """
    Determine whether token is a source clip.
    (Standard expression: single letter; or akarin.Expr: srcN)
    """
    if re.fullmatch(r'[a-zA-Z]', token):
        return True
    if re.fullmatch(r'src\d+', token):
        return True
    return False

def is_constant(token: str) -> bool:
    """
    Check if token is a built-in constant.
    """
    constants_set = {'N', 'X', 'current_x', 'Y', 'current_y', 'width', 'current_width', 'height', 'current_height'}
    if token in constants_set:
        return True
    if is_clip(token):
        return True
    return False

def strip_outer_parentheses(expr: str) -> str:
    """
    Remove the outer parentheses if the entire expression is enclosed.
    """
    if not expr.startswith('(') or not expr.endswith(')'):
        return expr
    count = 0
    for i, char in enumerate(expr):
        if char == '(':
            count += 1
        elif char == ')':
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
    m = re.match(r'(\w+)\s*\(', expr)
    if not m:
        return None
    func_name = m.group(1)
    start = m.end() - 1  # Position of '('
    depth = 0
    for i in range(start, len(expr)):
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
            if depth == 0:
                if i == len(expr) - 1:
                    args_str = expr[start+1:i]
                    return func_name, args_str
                else:
                    return None
    return None

def extract_function_info(internal_var: str, current_function: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a renamed internal variable (e.g. internalFuncArg),
    extract the original function name and variable name.
    """
    if current_function is not None and internal_var.startswith("internal" + current_function):
        return current_function, internal_var[len("internal" + current_function):]
    match = re.match(r'internal([a-zA-Z_]\w*?)([a-zA-Z_]\w+)$', internal_var)
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
    
    The unrolled fragments are joined with semicolons so that no extra newlines are added.  
    To preserve the original line count, we count the newline characters in the replaced block (the loop statement and matching braces)
    and append that many newline characters as placeholders.
    The base_line parameter indicates the starting line number for the code block.
    """
    pattern = re.compile(r'loop\s*\(\s*(\d+)\s*,\s*([a-zA-Z_]\w*)\s*\)\s*\{')
    while True:
        m = pattern.search(code)
        if not m:
            break
        n = int(m.group(1))
        var = m.group(2)
        loop_start_index = m.start()
        # Compute the line number by counting newlines before this match
        line_num = base_line + code[:loop_start_index].count('\n')
        brace_start = m.end() - 1
        count = 0
        end_index = None
        for i in range(brace_start, len(code)):
            if code[i] == '{':
                count += 1
            elif code[i] == '}':
                count -= 1
                if count == 0:
                    end_index = i
                    break
        if end_index is None:
            raise SyntaxError("Couldn't find matching '}'.", line_num)
        # The block inside the loop braces
        block = code[brace_start+1:end_index]
        block_base_line = base_line + code[:brace_start+1].count('\n')
        expanded_block = expand_loops(block, block_base_line)
        unrolled = []
        for k in range(n):
            # Replace only the standalone occurrences of the iteration variable
            iter_block = re.sub(r'\b' + re.escape(var) + r'\b', str(k), expanded_block)
            unrolled.append(iter_block)
        # Join unrolled fragments using semicolons.
        unrolled_code = ''.join(map(lambda s: s.strip() + ';', unrolled)).strip(';')
        # To preserve line numbers, count the newline characters in the original loop construct.
        original_block = code[m.start():end_index+1]
        newline_placeholder = "\n" * original_block.count("\n")
        replacement = unrolled_code + newline_placeholder
        code = code[:m.start()] + replacement + code[end_index+1:]
    return code

def infix2postfix(infix_code: str) -> LiteralString:
    """
    Convert infix expressions to postfix expressions, supporting function definitions,
    function calls, and built-in functions. Also supports nth_N and loop syntax sugar.
    
    Note: User input is not allowed to contain semicolons; if a semicolon is detected, it is rejected.
    """
    # Reject user input that contains semicolons.
    if ';' in infix_code:
        raise SyntaxError("User input code cannot contain semicolon. Use newlines instead.")
    
    # First, expand loops; the top-level code begins at line 1.
    expanded_code = expand_loops(infix_code, base_line=1)
    
    # Extract function definitions while preserving the line count.
    functions: Dict[str, Tuple[List[str], str, int]] = {}
    function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    # Replace the matched function definitions with the same number of newlines as they occupied.
    def replace_function(match: re.Match) -> str:
        func_name = match.group(1)
        params_str = match.group(2)
        body = match.group(3)
        func_start_index = match.start()
        line_num = 1 + expanded_code[:func_start_index].count('\n')
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        for param in params:
            if param.startswith('internal'):
                raise SyntaxError(
                    f"Parameter name '{param}' cannot start with 'internal' (reserved prefix)",
                    line_num
                )
        functions[func_name] = (params, body, line_num)
        # Return a string that contains the same number of newlines as the original function definition.
        return "\n" * match.group(0).count("\n")
    
    cleaned_code = re.sub(function_pattern, replace_function, expanded_code)
    
    # Process global code by splitting on physical newlines only.
    global_statements = []
    for i, line in enumerate(cleaned_code.splitlines(), start=1):
        if line.strip():
            global_statements.append((line.strip(), i))
    
    variables: Set[str] = set()  # Global variables
    postfix_tokens = []
    
    for stmt, line_num in global_statements:
        # Ignore comment lines.
        if stmt.startswith('#'):
            continue
        # Process assignment statements (ensuring no conflict with '==' or '<=' etc.)
        if '=' in stmt and not re.search(r'[<>!]=|==', stmt):
            var_name, expr = stmt.split('=', 1)
            var_name = var_name.strip()
            if var_name.startswith('internal'):
                raise SyntaxError(
                    f"Variable name '{var_name}' cannot start with 'internal' (reserved prefix)",
                    line_num
                )
            if is_constant(var_name):
                raise SyntaxError(f"Cannot assign to constant '{var_name}'.", line_num)
            expr = expr.strip()
            # I don't know how to handle 'a = a...' in check_variable_usage(), so do it here
            if var_name not in variables and re.search(r'\b' + re.escape(var_name) + r'\b', expr):
                raise SyntaxError(f"Variable '{var_name}' used before definition")
            variables.add(var_name)
            expr_postfix = convert_expr(expr, variables, functions, line_num)
            postfix_tokens.append(f"{expr_postfix} {var_name}!")
        else:
            postfix_tokens.append(convert_expr(stmt, variables, functions, line_num))
    
    final_result = ' '.join(postfix_tokens)
    final_result = final_result.replace('(', '').replace(')', '')
    return final_result

def validate_static_relative_pixel_indices(expr: str, line_num: int, function_name: Optional[str] = None) -> None:
    """
    Validate that static relative pixel access indices are numeric constants.
    """
    static_relative_pixel_indices = re.finditer(r'\w+\[(.*?)\]', expr)
    for match in static_relative_pixel_indices:
        indices = match.group(1).split(',')
        for idx in indices:
            if not re.fullmatch(r'-?\d+', idx.strip()):
                raise SyntaxError(
                    f"Static relative pixel access index must be a numeric constant, got '{idx.strip()}'",
                    line_num,
                    function_name
                )

def check_variable_usage(expr: str, variables: Set[str], line_num: int,
                         function_name: Optional[str] = None, local_vars: Optional[Set[str]] = None) -> None:
    """
    Check if all variables in the expression are defined.
    """
    expr_no_funcs = re.sub(r'\w+\([^)]*\)', '', expr)
    expr_no_stat_rels = re.sub(r'\[[^\]]*\]', '', expr_no_funcs)
    identifiers = re.finditer(r'\b([a-zA-Z_]\w*)\b', expr_no_stat_rels)
    
    for match in identifiers:
        var_name = match.group(1)
        if is_constant(var_name) or var_name in variables or (local_vars is not None and var_name in local_vars):
            continue
        if var_name.startswith('internal'):
            func_name_extracted, orig_var = extract_function_info(var_name, function_name)
            if local_vars is not None and orig_var in local_vars:
                continue
            raise SyntaxError(
                f"Variable '{orig_var}' used before definition",
                line_num,
                func_name_extracted
            )
        if not re.search(fr'{re.escape(var_name)}\s*=', expr):
            raise SyntaxError(f"Variable '{var_name}' used before definition", line_num, function_name)

def convert_expr(expr: str,
                 variables: Set[str],
                 functions: Dict[str, Tuple[List[str], str, int]],
                 line_num: int,
                 current_function: Optional[str] = None,
                 local_vars: Optional[Set[str]] = None) -> str:
    """
    Convert a single infix expression to a postfix expression,
    supporting operators, function calls, and custom function definitions.
    """
    expr = expr.strip()
    # Check variable usage and validate static relative pixel indices.
    check_variable_usage(expr, variables, line_num, current_function, local_vars)
    validate_static_relative_pixel_indices(expr, line_num, current_function)
    
    number_pattern = re.compile(
        r'^('
        r'0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?'
        r'|'
        r'0[0-7]*'
        r'|'
        r'[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?)'
        r')$'
    )
    if number_pattern.match(expr):
        return expr
    
    constants_map = {
        'current_frame_number': 'N',
        'current_x': 'X',
        'current_y': 'Y',
        'current_width': 'width',
        'current_height': 'height'
    }
    if expr in constants_map:
        return constants_map[expr]
    
    func_call_full = match_full_function_call(expr)
    if func_call_full:
        func_name, args_str = func_call_full
        if func_name not in functions and not is_builtin_function(func_name):
            raise SyntaxError(f"Undefined function '{func_name}'", line_num, current_function)
        m_nth = re.match(r'^nth_(\d+)$', func_name)
        if m_nth:
            N_val = int(m_nth.group(1))
            args = parse_args(args_str)
            M = len(args)
            if M < N_val:
                raise SyntaxError(
                    f"nth_{N_val} requires at least {N_val} arguments, but only {M} were provided.",
                    line_num,
                    current_function
                )
            args_postfix = [convert_expr(arg, variables, functions, line_num, current_function, local_vars)
                            for arg in args]
            sort_token = f"sort{M}"
            drop_before = f"drop{N_val-1}" if (N_val-1) > 0 else ""
            drop_after = f"drop{M-N_val}" if (M-N_val) > 0 else ""
            tokens = []
            tokens.append(" ".join(args_postfix))
            tokens.append(sort_token)
            if drop_before:
                tokens.append(drop_before)
            tokens.append("internalnthtmp!")
            if drop_after:
                tokens.append(drop_after)
            tokens.append("internalnthtmp@")
            return " ".join(tokens).strip()
        
        args = parse_args(args_str)
        args_postfix = [convert_expr(arg, variables, functions, line_num, current_function, local_vars)
                        for arg in args]
        if func_name in ['min', 'max'] and len(args) == 2:
            return f"{args_postfix[0]} {args_postfix[1]} {func_name}"
        elif func_name == 'clamp' and len(args) == 3:
            return f"{args_postfix[0]} {args_postfix[1]} {args_postfix[2]} clamp"
        else:
            builtin_unary = ['sin', 'cos', 'round', 'floor', 'abs', 'sqrt', 'trunc', 'bitnot', 'not']
            if func_name in builtin_unary and len(args) == 1:
                return f"{args_postfix[0]} {func_name}"
        # Handling custom function calls.
        if func_name in functions:
            params, body, func_line_num = functions[func_name]
            if len(args) != len(params):
                raise SyntaxError(
                    f"Function {func_name} requires {len(params)} parameters, but {len(args)} were provided.",
                    line_num,
                    current_function
                )
            param_map = {p: f"internal{func_name}{p}" for p in params}
            body_lines = [line.strip() for line in body.split('\n') if line.strip()]
            local_map = {}
            for body_line in body_lines:
                if body_line.startswith('return'):
                    continue
                m_line = re.match(r'^([a-zA-Z_]\w*)\s*=\s*(.+)$', body_line)
                if m_line:
                    var = m_line.group(1)
                    if var.startswith('internal'):
                        raise SyntaxError(
                            f"Variable name '{var}' cannot start with 'internal' (reserved prefix)",
                            func_line_num,
                            func_name
                        )
                    if is_constant(var):
                        raise SyntaxError(f"Cannot assign to constant '{var}'.", func_line_num, func_name)
                    if var not in param_map and not var.startswith(f"internal{func_name}"):
                        if var not in local_map:
                            local_map[var] = f"internal{func_name}{var}"
            rename_map = {}
            rename_map.update(param_map)
            rename_map.update(local_map)
            new_local_vars = set(param_map.keys()).union(set(local_map.keys()))
            param_assignments = []
            for i, p in enumerate(params):
                arg_orig = args[i].strip()
                if is_constant(arg_orig):
                    rename_map[p] = args_postfix[i]
                else:
                    param_assignments.append(f"{args_postfix[i]} {rename_map[p]}!")
            new_lines = []
            for line_text in body_lines:
                new_line = line_text
                for old, new in rename_map.items():
                    new_line = re.sub(rf'\b{re.escape(old)}\b', new, new_line)
                new_lines.append(new_line)
            function_tokens = []
            for offset, body_line in enumerate(new_lines):
                effective_line_num = func_line_num + offset + 1
                if body_line.startswith('return'):
                    ret_expr = body_line[len('return'):].strip().rstrip(';')
                    function_tokens.append(
                        convert_expr(ret_expr, variables, functions, effective_line_num, func_name, new_local_vars)
                    )
                # If a semicolon is present in the assignment statement, split and process separately.
                elif '=' in body_line and not re.search(r'[<>!]=|==', body_line):
                    if ';' in body_line:
                        tokens = []
                        for sub in body_line.split(';'):
                            sub = sub.strip()
                            if not sub:
                                continue
                            if '=' in sub and not re.search(r'[<>!]=|==', sub):
                                var_name, expr_line = sub.split('=', 1)
                                var_name = var_name.strip()
                                if is_constant(var_name):
                                    raise SyntaxError(f"Cannot assign to constant '{var_name}'.", effective_line_num, func_name)
                                expr_line = expr_line.strip()
                                # I don't know how to handle 'a = a...' in check_variable_usage(), so do it here
                                if var_name not in new_local_vars and re.search(r'\b' + re.escape(var_name) + r'\b', expr_line):
                                    _, orig_var = extract_function_info(var_name, func_name)
                                    raise SyntaxError(f"Variable '{orig_var}' used before definition", effective_line_num, func_name)
                                if var_name not in new_local_vars:
                                    new_local_vars.add(var_name)
                                tokens.append(f"{convert_expr(expr_line, variables, functions, effective_line_num, func_name, new_local_vars)} {var_name}!")
                            else:
                                tokens.append(convert_expr(sub, variables, functions, effective_line_num, func_name, new_local_vars))
                        function_tokens.append(' '.join(tokens))
                    else:
                        var_name, expr_line = body_line.split('=', 1)
                        var_name = var_name.strip()
                        if is_constant(var_name):
                            raise SyntaxError(f"Cannot assign to constant '{var_name}'.", effective_line_num, func_name)
                        expr_line = expr_line.strip()
                        # I don't know how to handle 'a = a...' in check_variable_usage(), so do it here
                        if var_name not in new_local_vars and re.search(r'\b' + re.escape(var_name) + r'\b', expr_line):
                            _, orig_var = extract_function_info(var_name, func_name)
                            raise SyntaxError(f"Variable '{orig_var}' used before definition", effective_line_num, func_name)
                        if var_name not in new_local_vars:
                            new_local_vars.add(var_name)
                        function_tokens.append(f"{convert_expr(expr_line, variables, functions, effective_line_num, func_name, new_local_vars)} {var_name}!")
                else:
                    function_tokens.append(
                        convert_expr(body_line, variables, functions, effective_line_num, func_name, new_local_vars)
                    )
            return ' '.join(param_assignments + function_tokens)
    
    stripped = strip_outer_parentheses(expr)
    if stripped != expr:
        return convert_expr(stripped, variables, functions, line_num, current_function, local_vars)
    
    m_static = re.match(r'^(\w+)\[(-?\d+),\s*(-?\d+)\](\:\w)?$', expr)
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
                current_function
            )
        return f"{clip}[{statX},{statY}]{suffix}"
    
    m_dyn = re.match(r'^(\w+)\.dyn\((.+),\s*(.+)\)$', expr)
    if m_dyn:
        clip = m_dyn.group(1)
        if not is_clip(clip):
            raise SyntaxError(f"'{clip}' is not a clip!", line_num, current_function)
        x_expr = convert_expr(m_dyn.group(2), variables, functions, line_num, current_function, local_vars)
        y_expr = convert_expr(m_dyn.group(3), variables, functions, line_num, current_function, local_vars)
        return f"{x_expr} {y_expr} {clip}[]"
    
    m_ternary = re.search(r'(.+?)\s*\?\s*(.+?)\s*:\s*(.+)', expr)
    if m_ternary:
        cond = convert_expr(m_ternary.group(1), variables, functions, line_num, current_function, local_vars)
        true_expr = convert_expr(m_ternary.group(2), variables, functions, line_num, current_function, local_vars)
        false_expr = convert_expr(m_ternary.group(3), variables, functions, line_num, current_function, local_vars)
        return f"{cond} {true_expr} {false_expr} ?"
    
    operators = [
        ('||', 'or'), ('&&', 'and'),
        ('==', '='), ('!=', '!='), 
        ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'),
        ('+', '+'), ('-', '-'),
        ('*', '*'), ('/', '/'), ('%', '%'),
        ('**', 'pow'),
        ('&', 'bitand'), ('|', 'bitor'), ('^', 'bitxor')
    ]
    for op_str, postfix_op in operators:
        left, right = find_binary_op(expr, op_str)
        if left is not None and right is not None:
            left_postfix = convert_expr(left, variables, functions, line_num, current_function, local_vars)
            right_postfix = convert_expr(right, variables, functions, line_num, current_function, local_vars)
            if re.fullmatch(r'[a-zA-Z_]\w*', left.strip()) and not left_postfix.strip().endswith('@'):
                left_postfix = left_postfix.strip() + ('@' if not is_constant(left_postfix) else '')
            if re.fullmatch(r'[a-zA-Z_]\w*', right.strip()) and not right_postfix.strip().endswith('@'):
                right_postfix = right_postfix.strip() + ('@' if not is_constant(right_postfix) else '')
            return f"{left_postfix} {right_postfix} {postfix_op}"
    
    if expr.startswith('!'):
        operand = convert_expr(expr[1:], variables, functions, line_num, current_function, local_vars)
        return f"{operand} not"
    
    if is_constant(expr):
        return expr
    if re.fullmatch(r'[a-zA-Z_]\w*', expr):
        return expr if expr.endswith('@') else expr + '@'
    
    return expr

def find_binary_op(expr: str, op: str):
    """
    Find the last occurrence of the binary operator op at the outer level
    and return the left and right parts of the expression.
    """
    level = 0
    op_index = -1
    for i in range(len(expr)):
        c = expr[i]
        if c == '(':
            level += 1
        elif c == ')':
            level -= 1
        if level == 0 and expr[i:i+len(op)] == op:
            op_index = i
    if op_index == -1:
        return None, None
    left = expr[:op_index].strip()
    right = expr[op_index+len(op):].strip()
    return left, right

def parse_args(args_str: str) -> List[str]:
    """
    Parse function call arguments.
    """
    args = []
    current = []
    bracket_depth = 0
    square_bracket_depth = 0
    for c in args_str + ',':
        if c == ',' and bracket_depth == 0 and square_bracket_depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
            if c == '(':
                bracket_depth += 1
            elif c == ')':
                bracket_depth -= 1
            elif c == '[':
                square_bracket_depth += 1
            elif c == ']':
                square_bracket_depth -= 1
    return [arg for arg in args if arg]

def is_builtin_function(func_name: str) -> bool:
    """
    Check if the function name belongs to a built-in function.
    """
    builtin_unary = ['sin', 'cos', 'round', 'floor', 'abs', 'sqrt', 'trunc', 'bitnot', 'not']
    builtin_binary = ['min', 'max']
    builtin_ternary = ['clamp']
    if re.match(r'^nth_\d+$', func_name):
        return True
    return (func_name in builtin_unary or 
            func_name in builtin_binary or 
            func_name in builtin_ternary)