import re
import sys
if sys.version_info >= (3, 11):
    from typing import LiteralString, Dict, List, Tuple, Optional
else:
    from typing import Dict, List, Tuple, Optional
    LiteralString = str

def is_constant(token: str) -> bool:
    """
    Determine whether token is a built-in constant or source pixel.
    Built-in constants include N, X, current_x, Y, current_y, width, current_width, height, current_height.
    Single letters and identifiers like srcN are also considered constants.
    """
    constants_set = {'N', 'X', 'current_x', 'Y', 'current_y', 'width', 'current_width', 'height', 'current_height'}
    if token in constants_set:
        return True
    if re.fullmatch(r'[a-zA-Z]', token):
        return True
    if re.fullmatch(r'src\d+', token):
        return True
    return False

def strip_outer_parentheses(expr: str) -> str:
    """
    If the entire expression is surrounded by a matching pair of parentheses,
    remove the outer parentheses; otherwise return the original string.
    """
    if not expr.startswith('(') or not expr.endswith(')'):
        return expr
    count = 0
    for i, char in enumerate(expr):
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        # If the count becomes zero before the end, the parentheses don't fully enclose the expression
        if count == 0 and i < len(expr) - 1:
            return expr
    return expr[1:-1]

def match_full_function_call(expr: str) -> Optional[Tuple[str, str]]:
    """
    Attempt to match whether the entire expr is a complete function call (supports nested parentheses).
    Returns (func_name, args_str) if the match is successful; otherwise returns None.
    """
    expr = expr.strip()
    m = re.match(r'(\w+)\s*\(', expr)
    if not m:
        return None
    func_name = m.group(1)
    start = m.end() - 1  # expr[start] should be '('
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

def expand_loops(code: str) -> str:
    """
    Expands the loop syntax sugar for fixed number of iterations.
    For example:
        loop(2, i) { ... }
    Will be expanded into two copies of the loop body, with i replaced by 0 and 1 respectively.
    """
    pattern = re.compile(r'loop\s*\(\s*(\d+)\s*,\s*([a-zA-Z_]\w*)\s*\)\s*\{')
    while True:
        m = pattern.search(code)
        if not m:
            break
        n = int(m.group(1))
        var = m.group(2)
        loop_start = m.start()
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
            raise SyntaxError("infix2postfix: Couldn't find matching '}'.")
        block = code[brace_start+1:end_index]
        # Recursively expand inner loops
        expanded_block = expand_loops(block)
        unrolled = []
        for k in range(n):
            # Only replace standalone variable identifiers
            iter_block = re.sub(r'\b' + re.escape(var) + r'\b', str(k), expanded_block)
            unrolled.append(iter_block)
        unrolled_code = ' '.join(unrolled)
        code = code[:loop_start] + unrolled_code + code[end_index+1:]
    return code

def infix2postfix(infix_code: str) -> LiteralString:
    """
    Convert infix expressions to postfix expressions, supporting function definitions,
    function calls, and built-in functions. Also supports nth_N and loop syntax sugar.
    """
    # Preprocessing: expand loop syntax sugar
    infix_code = expand_loops(infix_code)
    
    # Preprocessing: extract function definitions
    functions: Dict[str, Tuple[List[str], str]] = {}
    function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*((?:\{[^{}]*\})[^{}]*)*)\}'
    cleaned_code = infix_code
    for match in re.finditer(function_pattern, infix_code):
        func_name = match.group(1)
        params_str = match.group(2)
        body = match.group(3)
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        functions[func_name] = (params, body)
        cleaned_code = cleaned_code.replace(match.group(0), '')
        
    # Process global code
    lines = re.split(r'[\n;]', cleaned_code.strip())
    variables = set()  # Store global variables
    postfix_tokens = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # If it's an assignment statement, convert to "expression variable!" format
        if '=' in line and not re.search(r'[<>!]=|==', line):
            var_name, expr = line.split('=', 1)
            var_name = var_name.strip()
            if is_constant(var_name):
                raise SyntaxError(f"Error: Cannot assign to constant '{var_name}'.")
            expr = expr.strip()
            variables.add(var_name)
            expr_postfix = convert_expr(expr, variables, functions)
            postfix_tokens.append(f"{expr_postfix} {var_name}!")
        else:
            postfix_tokens.append(convert_expr(line, variables, functions))
    
    final_result = ' '.join(postfix_tokens)
    final_result = final_result.replace('(', '').replace(')', '')
    return final_result

def convert_expr(expr: str, variables: set, functions: Dict[str, Tuple[List[str], str]]) -> str:
    """
    Convert a single infix expression to postfix expression.
    
    Conversion rules:
        - Numbers and constants are returned directly;
        - Built-in functions are converted directly;
        - Custom function calls rename all parameters and local variables, then convert to postfix assignment sequences;
        - Variable references append '@' (but single letters and srcN are treated as constants, with no appending);
        - Assignment statements are converted to "expression variable!" format;
        - nth_N syntax sugar: e.g., nth_3(a, b, c, d) is converted to:
            a_postfix b_postfix c_postfix d_postfix sort4 [drop2] nth! [drop1] nth@
        (where drop2 and drop1 are added only if needed).
    """
    expr = expr.strip()
    
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
        
        # Handle nth_N syntax sugar
        m_nth = re.match(r'^nth_(\d+)$', func_name)
        if m_nth:
            N_val = int(m_nth.group(1))
            args = parse_args(args_str)
            M = len(args)
            if M < N_val:
                raise ValueError(f"nth_{N_val} requires at least {N_val} arguments, but only {M} were provided.")
            args_postfix = [convert_expr(arg, variables, functions) for arg in args]
            sort_token = f"sort{M}"
            drop_before = f"drop{N_val-1}" if (N_val-1) > 0 else ""
            drop_after = f"drop{M-N_val}" if (M-N_val) > 0 else ""
            tokens = []
            tokens.append(" ".join(args_postfix))
            tokens.append(sort_token)
            if drop_before:
                tokens.append(drop_before)
            tokens.append("nth!")
            if drop_after:
                tokens.append(drop_after)
            tokens.append("nth@")
            return " ".join(tokens).strip()
        
        args = parse_args(args_str)
        args_postfix = [convert_expr(arg, variables, functions) for arg in args]
        # Handle built-in binary functions: min, max
        if func_name in ['min', 'max'] and len(args) == 2:
            return f"{args_postfix[0]} {args_postfix[1]} {func_name}"
        # Handle built-in ternary function: clamp
        elif func_name == 'clamp' and len(args) == 3:
            return f"{args_postfix[0]} {args_postfix[1]} {args_postfix[2]} clamp"
        else:
            builtin_unary = ['sin', 'cos', 'round', 'floor', 'abs', 'sqrt', 'trunc', 'bitnot', 'not']
            if func_name in builtin_unary and len(args) == 1:
                return f"{args_postfix[0]} {func_name}"
        # Handle custom function calls
        if func_name in functions:
            params, body = functions[func_name]
            if len(args) != len(params):
                raise ValueError(f"Function {func_name} requires {len(params)} parameters, but {len(args)} were provided.")
            # Create rename mapping for parameters (e.g., a -> internalfuncnamea)
            param_map = { p: f"internal{func_name}{p}" for p in params }
            lines = [line.strip() for line in body.split('\n') if line.strip()]
            local_map = {}
            # Scan body for assignments to collect local variables
            for line in lines:
                if line.startswith('return'):
                    continue
                m_line = re.match(r'^([a-zA-Z_]\w*)\s*=\s*(.+)$', line)
                if m_line:
                    var = m_line.group(1)
                    if is_constant(var):
                        raise SyntaxError(f"Error: Cannot assign to constant '{var}'.")
                    if var not in param_map and not var.startswith(f"internal{func_name}"):
                        if var not in local_map:
                            local_map[var] = f"internal{func_name}{var}"
            # Merge rename mappings
            rename_map = {}
            rename_map.update(param_map)
            rename_map.update(local_map)
            # Generate parameter assignments
            param_assignments = []
            for i, p in enumerate(params):
                arg_orig = args[i].strip()
                if is_constant(arg_orig):
                    # For constants, directly replace in the rename map without generating assignment
                    rename_map[p] = args_postfix[i]
                else:
                    param_assignments.append(f"{args_postfix[i]} {rename_map[p]}!")
            # Apply rename substitution to each line in function body
            new_lines = []
            for line in lines:
                new_line = line
                for old, new in rename_map.items():
                    new_line = re.sub(rf'\b{re.escape(old)}\b', new, new_line)
                new_lines.append(new_line)
            # Generate tokens for function body
            function_tokens = []
            for line in new_lines:
                if line.startswith('return'):
                    ret_expr = line[len('return'):].strip().rstrip(';')
                    function_tokens.append(convert_expr(ret_expr, variables, functions))
                elif '=' in line and not re.search(r'[<>!]=|==', line):
                    var_name, expr_line = line.split('=', 1)
                    var_name = var_name.strip()
                    if is_constant(var_name):
                        raise SyntaxError(f"Error: Cannot assign to constant '{var_name}'.")
                    expr_line = expr_line.strip()
                    function_tokens.append(f"{convert_expr(expr_line, variables, functions)} {var_name}!")
                else:
                    function_tokens.append(convert_expr(line, variables, functions))
            return ' '.join(param_assignments + function_tokens)
    
    # Try stripping outer parentheses
    stripped = strip_outer_parentheses(expr)
    if stripped != expr:
        return convert_expr(stripped, variables, functions)
    
    # Handle static relative pixel access, like x[1,2]
    m_static = re.match(r'^(\w+)\[(-?\d+),\s*(-?\d+)\](\:\w)?$', expr)
    if m_static:
        clip = m_static.group(1)
        statX = m_static.group(2)
        statY = m_static.group(3)
        suffix = m_static.group(4) or ""
        return f"{clip}[{statX},{statY}]{suffix}"
    
    # Handle dynamic pixel access, like clip.dyn(expr, expr)
    m_dyn = re.match(r'^(\w+)\.dyn\((.+),\s*(.+)\)$', expr)
    if m_dyn:
        clip = m_dyn.group(1)
        x_expr = convert_expr(m_dyn.group(2), variables, functions)
        y_expr = convert_expr(m_dyn.group(3), variables, functions)
        return f"{x_expr} {y_expr} {clip}[]"
    
    # Handle ternary operator: condition ? true_expr : false_expr
    m_ternary = re.search(r'(.+?)\s*\?\s*(.+?)\s*:\s*(.+)', expr)
    if m_ternary:
        cond = convert_expr(m_ternary.group(1), variables, functions)
        true_expr = convert_expr(m_ternary.group(2), variables, functions)
        false_expr = convert_expr(m_ternary.group(3), variables, functions)
        return f"{cond} {true_expr} {false_expr} ?"
    
    # Handle binary operators (from lowest to highest precedence)
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
            left_postfix = convert_expr(left, variables, functions)
            right_postfix = convert_expr(right, variables, functions)
            # For single identifiers, add '@' if not already ending with it (unless it's a constant)
            if re.fullmatch(r'[a-zA-Z_]\w*', left.strip()) and not left_postfix.strip().endswith('@'):
                left_postfix = left_postfix.strip() + ('@' if not is_constant(left_postfix) else '')
            if re.fullmatch(r'[a-zA-Z_]\w*', right.strip()) and not right_postfix.strip().endswith('@'):
                right_postfix = right_postfix.strip() + ('@' if not is_constant(right_postfix) else '')
            return f"{left_postfix} {right_postfix} {postfix_op}"
    
    # Handle unary operator '!'
    if expr.startswith('!'):
        operand = convert_expr(expr[1:], variables, functions)
        return f"{operand} not"
    
    # For identifiers: return directly if constant, otherwise append '@'
    if is_constant(expr):
        return expr
    if re.fullmatch(r'[a-zA-Z_]\w*', expr):
        return expr if expr.endswith('@') else expr + '@'
    
    return expr

def find_binary_op(expr: str, op: str):
    """
    Find the last occurrence of the binary operator op at the outermost level (parentheses level 0) in expr,
    and return the left and right subexpressions (with whitespace removed).
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
    Argument parser that supports commas inside array indices.
    """
    args = []
    current = []
    bracket_depth = 0  # Parentheses depth
    square_bracket_depth = 0  # Square brackets depth
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