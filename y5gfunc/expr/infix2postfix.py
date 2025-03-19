import re
import sys
from typing import Optional
if sys.version_info >= (3, 11):
    from typing import LiteralString, Dict, List, Tuple
else:
    from typing import Dict, List, Tuple
    LiteralString = str

def is_constant(token: str) -> bool:
    """
    Determine if the identifier token is built-in constants and source pixels.
    """
    constants_set = {'N', 'X', 'current_x', 'Y', 'current_y', 'width', 'current_width', 'height', 'current_height'}
    print(token)
    if token in constants_set:
        return True
    if re.fullmatch(r'[a-zA-Z]', token):
        return True
    if re.fullmatch(r'src\d+', token):
        return True
    print(0)
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
        # If count becomes zero before the end, the outer parentheses don't fully enclose the expression
        if count == 0 and i < len(expr) - 1:
            return expr
    return expr[1:-1]

def match_full_function_call(expr: str) -> Optional[Tuple[str, str]]:
    """
    Attempts to match whether the entire expr is a complete function call (nested parentheses are supported).
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
                # If the matching right bracket is exactly at the end, the whole expression is considered to be a single function call.
                if i == len(expr) - 1:
                    args_str = expr[start+1:i]
                    return func_name, args_str
                else:
                    return None
    return None

def infix2postfix(infix_code: str) -> LiteralString:
    """
    Convert infix expressions to postfix expressions, supporting function definitions and calls.
    """
    # Preprocessing: extract function definitions
    functions = {}  # Format: {function_name: (parameter_list, function_body)}
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
        # Global assignment: directly convert to "expression variable!" form
        if '=' in line and not re.search(r'[<>!]=|==', line):
            var_name, expr = line.split('=', 1)
            var_name = var_name.strip()
            # Basic syntax check: cannot assign to constants
            if is_constant(var_name):
                raise SyntaxError(f"Error: Cannot assign to constant '{var_name}'.")
            expr = expr.strip()
            variables.add(var_name)
            expr_postfix = convert_expr(expr, variables, functions)
            postfix_tokens.append(f"{expr_postfix} {var_name}!")
        else:
            postfix_tokens.append(convert_expr(line, variables, functions))
    
    # Remove all parentheses from the final result (no parentheses should appear in global postfix expressions)
    final_result = ' '.join(postfix_tokens)
    final_result = final_result.replace('(', '').replace(')', '')
    return final_result

def convert_expr(expr: str, variables: set, functions: Dict[str, Tuple[List[str], str]]) -> str:
    """
    Convert a single infix expression to postfix expression, supporting function calls, 
    binary/ternary operations, etc.
    
    Conversion rules:
        - Numbers and constants are returned directly;
        - Built-in functions are converted directly: built-in binary functions support 'min', 'max', and the ternary function 'clamp';
        - Built-in unary functions support ['sin', 'cos', 'round', 'floor', 'abs', 'sqrt', 'trunc', 'bitnot', 'not'];
        - Custom function calls rename all parameters and local variables (e.g., testinternala, testinternalb), and convert everything to postfix assignment sequences;
        - Variable references (including renamed ones) are appended with '@' (but single letters and srcN are treated as constants, with no appending);
        - Assignment statements are converted to "expression variable!" format, with syntax checking to prevent assignment to constants.
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
    # Return numbers directly
    if number_pattern.match(expr):
        return expr
    
    # Constant mapping
    constants = {
        'current_frame_number': 'N',
        'current_x': 'X',
        'current_y': 'Y',
        'current_width': 'width',
        'current_height': 'height'
    }
    if expr in constants:
        return constants[expr]
    
    # Determine if it's a function call form
    func_call_full = match_full_function_call(expr)
    if func_call_full:
        func_name, args_str = func_call_full
        args = parse_args(args_str)
        args_postfix = [convert_expr(arg, variables, functions) for arg in args]
        # Built-in function processing
        if func_name in ['min', 'max'] and len(args) == 2:
            return f"{args_postfix[0]} {args_postfix[1]} {func_name}"
        elif func_name == 'clamp' and len(args) == 3:
            return f"{args_postfix[0]} {args_postfix[1]} {args_postfix[2]} clamp"
        else:
            builtin_unary = ['sin', 'cos', 'round', 'floor', 'abs', 'sqrt', 'trunc', 'bitnot', 'not']
            if func_name in builtin_unary and len(args) == 1:
                return f"{args_postfix[0]} {func_name}"

        if func_name in functions:
            params, body = functions[func_name]
            if len(args) != len(params):
                raise ValueError(f"Function {func_name} requires {len(params)} parameters, but {len(args)} were provided.")
            # ① Create rename mapping for all parameters, e.g., a -> {func_name}internala, b -> {func_name}internalb
            param_map = { p: f"{func_name}internal{p}" for p in params }
            # ② Scan assignment statements in the function body (except return), collect local variables (non-parameters), 
            # rename them to {func_name}internal variable form
            lines = [line.strip() for line in body.split('\n') if line.strip()]
            local_map = {}
            for line in lines:
                if line.startswith('return'):
                    continue
                m = re.match(r'^([a-zA-Z_]\w*)\s*=\s*(.+)$', line)
                if m:
                    var = m.group(1)
                    # Syntax check: cannot assign to constants
                    if is_constant(var):
                        raise SyntaxError(f"Error: Cannot assign to constant '{var}'.")
                    if var not in param_map and not var.startswith(f"{func_name}internal"):
                        if var not in local_map:
                            local_map[var] = f"{func_name}internal{var}"
            # ③ Merge rename mappings
            rename_map = {}
            rename_map.update(param_map)
            rename_map.update(local_map)
            # ④ Perform rename substitution for each line in the function body
            new_lines = []
            for line in lines:
                new_line = line
                for old, new in rename_map.items():
                    new_line = re.sub(rf'\b{re.escape(old)}\b', new, new_line)
                new_lines.append(new_line)
            # ⑤ Generate parameter assignment tokens, e.g., "x@ testinternala!"
            # Recalculate postfix expressions for each parameter
            args_postfix = [convert_expr(arg, variables, functions) for arg in args]
            param_assignments = [f"{args_postfix[i]} {rename_map[params[i]]}!" for i in range(len(params))]
            # ⑥ Convert each line in the function body to postfix tokens in sequence
            function_tokens = []
            for line in new_lines:
                if line.startswith('return'):
                    ret_expr = line[len('return'):].strip().rstrip(';')
                    function_tokens.append(convert_expr(ret_expr, variables, functions))
                elif '=' in line and not re.search(r'[<>!]=|==', line):
                    var_name, expr_line = line.split('=', 1)
                    var_name = var_name.strip()
                    # Check function internal assignment: cannot assign to constants
                    if is_constant(var_name):
                        raise SyntaxError(f"Error: Cannot assign to constant '{var_name}'.")
                    expr_line = expr_line.strip()
                    function_tokens.append(f"{convert_expr(expr_line, variables, functions)} {var_name}!")
                else:
                    function_tokens.append(convert_expr(line, variables, functions))
            return ' '.join(param_assignments + function_tokens)
        
    # Strip outer parentheses
    stripped = strip_outer_parentheses(expr)
    if stripped != expr:
        return convert_expr(stripped, variables, functions)
    
    # Process static relative pixel access: y[n,n]
    static_relative_pixel_match = re.match(r'(\w+)\[(-?\d+),\s*(-?\d+)\](:m)?', expr)
    if static_relative_pixel_match:
        clip = static_relative_pixel_match.group(1)
        x_val = static_relative_pixel_match.group(2)
        y_val = static_relative_pixel_match.group(3)
        suffix = static_relative_pixel_match.group(4) or ""
        return f"{clip}[{x_val},{y_val}]{suffix}"
    
    # Process dynamic pixel access: clip.dyn(expr, expr)
    dyn_match = re.match(r'(\w+)\.dyn\((.+),\s*(.+)\)', expr)
    if dyn_match:
        clip = dyn_match.group(1)
        x_expr = convert_expr(dyn_match.group(2), variables, functions)
        y_expr = convert_expr(dyn_match.group(3), variables, functions)
        return f"{x_expr} {y_expr} {clip}[]"
    
    # Process ternary operator: condition ? true_expr : false_expr
    ternary_match = re.search(r'(.+?)\s*\?\s*(.+?)\s*:\s*(.+)', expr)
    if ternary_match:
        condition = convert_expr(ternary_match.group(1), variables, functions)
        true_expr = convert_expr(ternary_match.group(2), variables, functions)
        false_expr = convert_expr(ternary_match.group(3), variables, functions)
        return f"{condition} {true_expr} {false_expr} ?"
    
    # Process binary operators (from lowest to highest precedence)
    operators = [
        ('||', 'or'), ('&&', 'and'),
        ('==', '='), ('!=', '!='), 
        ('<=', '<='), ('>=', '>='), ('<', '<'), ('>', '>'),
        ('+', '+'), ('-', '-'),
        ('*', '*'), ('/', '/'), ('%', '%'),
        ('**', 'pow')
    ]
    for op_str, postfix_op in operators:
        left, right = find_binary_op(expr, op_str)
        if left is not None and right is not None:
            left_postfix = convert_expr(left, variables, functions)
            right_postfix = convert_expr(right, variables, functions)
            # If left and right parts are single identifiers but don't end with '@', add it if it's not a constant.
            if re.fullmatch(r'[a-zA-Z_]\w*', left.strip()) and not left_postfix.strip().endswith('@'):
                left_postfix = left_postfix.strip() + '@' if not is_constant(left_postfix) else left_postfix.strip()
            if re.fullmatch(r'[a-zA-Z_]\w*', right.strip()) and not right_postfix.strip().endswith('@'):
                right_postfix = right_postfix.strip() + '@' if not is_constant(right_postfix) else right_postfix.strip()
            return f"{left_postfix} {right_postfix} {postfix_op}"
    
    # Process unary operator '!'
    if expr.startswith('!'):
        operand = convert_expr(expr[1:], variables, functions)
        return f"{operand} not"
    
    # Processing for identifiers:
    # If it's a constant, return directly (no '@' appended)
    if is_constant(expr):
        return expr
    # Otherwise, if expr fully matches an identifier, append '@' (if not already appended)
    if re.fullmatch(r'[a-zA-Z_]\w*', expr):
        return expr if expr.endswith('@') else expr + '@'
    
    return expr

def find_binary_op(expr: str, op: str):
    """
    Find the outermost binary operator op in expr, and split it into left and right expressions.
    """
    if op not in expr:
        return None, None
    paren_level = 0
    # Scan from right to left
    for i in range(len(expr) - len(op), -1, -1):
        if expr[i:i+len(op)] == op and paren_level == 0:
            is_valid = True
            if i > 0 and expr[i-1].isalnum():
                is_valid = False
            if i+len(op) < len(expr) and expr[i+len(op)].isalnum():
                is_valid = False
            if is_valid:
                return expr[:i].strip(), expr[i+len(op):].strip()
        if expr[i] == ')':
            paren_level += 1
        elif expr[i] == '(':
            paren_level -= 1
    return None, None

def parse_args(args_str: str) -> List[str]:
    """
    Argument parser.
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