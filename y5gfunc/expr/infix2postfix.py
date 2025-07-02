import regex as re
from functools import reduce, lru_cache
from .optimize import optimize_akarin_expr
from typing import Optional
import sys
from enum import StrEnum

sys.setrecursionlimit(5000)


class GlobalMode(StrEnum):
    ALL = "all"
    NONE = "none"
    SPECIFIC = "specific"


def infix2postfix(infix_code: str) -> str:
    R"""
    Convert infix expressions to postfix expressions.

    [muvs](https://github.com/WolframRhodium/muvsfunc/blob/master/muvs.py) is a better alternative of this function.

    Args:
        infix_code: Input infix code.

    Returns:
        Converted postfix expr.

    Raises:
        SyntaxError: If infix code failed to convert to postfix expr.

    Refer to ..vfx/ and .expr_utils.py for examples.

    ## General Format

    - **Input Structure:**
    The source code is written as plain text with one statement per line. User input must not contain semicolons.

    - **Whitespace:**
    Whitespace (spaces and newlines) is used to separate tokens and statements. Extra spaces are generally ignored.

    ---

    ## Lexical Elements

    ### Identifiers and Literals

    - **Identifiers (Variable and Function Names):**
    - Must start with a letter or an underscore and can be composed of letters, digits, and underscores (matching `[a-zA-Z_]\w*`).
    - Identifiers starting with the reserved prefix `__internal_` are not allowed in user code (they are used internally for parameter renaming and temporary variables).
    - Some names are "built-in constants" (see below) and cannot be reassigned.

    - **Built-in Constants:**
    The language defines the following reserved identifiers: (Refer to std.Expr and akarin.Expr documents for more information)
    - `N`, `X`, `current_x`, `Y`, `current_y`, `width`, `current_width`, `height`, `current_height`, `pi`
    In addition, any token that is a single letter (e.g. `a`, `b`) or a string matching `src` followed by digits (e.g. `src1`, `src42`) is considered a source clip.

    - **Numeric Literals:**
    The language supports:  (Refer to akarin.Expr documents for more information)
    - **Decimal numbers:** Integers (e.g. `123`, `-42`) and floating‑point numbers (e.g. `3.14`, `-0.5` with optional scientific notation).
    - **Hexadecimal numbers:** Starting with `0x` followed by hexadecimal digits (optionally with a fractional part and a "p‑exponent").
    - **Octal numbers:** A leading zero followed by octal digits (e.g. `0755`).

    ---

    ## Operators

    ### Binary Operators

    Expressions may contain binary operators that, when converted, yield corresponding postfix tokens. The supported binary operators include:

    - **Logical Operators:**
    - `||` (logical OR; converted to `or`)
    - `&&` (logical AND; converted to `and`)

    - **Bitwise Operators:**
        - `&` (bitwise AND; converted to `bitand`)
        - `|` (bitwise OR; converted to `bitor`)
        - `^` (bitwise XOR; converted to `bitxor`)

    - **Equality and Relational Operators:**
    - `==` (equality, converted to `=` in postfix)
    - `!=` (inequality)
    - Relational: `<`, `>`, `<=`, `>=`

    - **Arithmetic Operators:**
    - `+` (addition)
    - `-` (subtraction)
    - `*` (multiplication)
    - `/` (division)
    - `%` (modulus)
    - `**` (exponentiation; converted to `pow`)

    When parsing an infix expression, the algorithm searches for a binary operator at the outer level (i.e. not inside any nested parentheses) and—depending on its position—splits the expression into left and right operands. The order in which the operator candidates are considered effectively defines the operator precedence.

    ### Unary Operators

    - **Negation:**
    A minus sign (`-`) may be used to denote negative numeric literals; if used before an expression that is not a literal number, it is interpreted as multiplying the operand by -1.

    - **Logical NOT:**
    An exclamation mark (`!`) is used as a unary operator for logical NOT. For example, `!expr` is converted to postfix by appending the operator `not` to the processed operand.

    ### Ternary Operator

    - **Conditional Expression:**
    The language supports a ternary operator with the syntax:
    ```
    condition ? true_expression : false_expression
    ```
    The operator first evaluates the condition; then based on its result, it selects either the true or false branch. In the conversion process, the three expressions are translated to a corresponding postfix form followed by a `?` token.

    ---

    ## Grouping

    - **Parentheses:**
    Parentheses `(` and `)` can be used to override the default evaluation order. If an entire expression is wrapped in parentheses, the outer pair is stripped prior to further conversion.

    ---

    ## Function Calls and Built-in Functions

    ### General Function Call Syntax

    - **Invocation:**
    Functions are called using the usual form:
    ```
    functionName(arg1, arg2, …)
    ```
    The argument list must be properly parenthesized and can contain nested expressions. The arguments are parsed by taking into account nested parentheses and brackets.

    - **Builtin Function Examples:**
    Some functions are specially handled and have fixed argument counts:

    - **Unary Functions:**
        `sin(x)`, `cos(x)`, `round(x)`, `floor(x)`, `abs(x)`, `sqrt(x)`, `trunc(x)`, `bitnot(x)`, `not(x)`

    - **Binary Functions:**
        `min(a, b)`, `max(a, b)`

    - **Ternary Functions:**
        `clamp(a, b, c)`, `dyn(a, b, c)`
        Again, in the case of `dyn` the first argument must be a valid source clip.
        In addition, an extra optimizing will be performed to convert dynamic access (`dyn`) to static access (see below) if possible for potentially higher performance.

    - **Special Pattern – nth_N Functions:**
        Function names matching the pattern `nth_<number>` (for example, `nth_2`) are supported. They require at least N arguments and returns the Nth smallest of the arguments (e.g. `nth_1(a, b, c, d)` returns the smallest one of `a`, `b`, `c` and `d`).

    ### Custom Function Definitions

    - **Syntax:**
    Functions are defined as follows:
    ```
    function functionName(param1, param2, …) {
        // function body
        return expression // optional
    }
    ```
    - **Requirements and Checks:**
    - The parameter names must be unique, and none may begin with the reserved prefix `__internal_`.
    - The function body may span several lines. It can contain assignment statements and expression evaluations.
    - There must be at most one return statement, and if it exists, it must be the last (non-empty) statement in the function body.
    - Defining functions inside another function is not currently supported.

    ---

    ## Global Declarations and Assignments

    ### Global Declarations

    - **Syntax:**
    Global variables may be declared on a dedicated line immediately preceding a function definition. Three formats are supported:

    1.  **Specific Globals:** To declare a specific set of global variables for a function:
        ```
        <global<var1><var2>…>
        ```
        The declared names (`var1`, `var2`, etc.) are recorded as the only global variables accessible by that function.

    2.  **All Globals:** To allow a function to access all currently defined global variables:
        ```
        <global.all>
        ```
        This makes any global variable defined in the top-level scope available within the function.

    3.  **No Globals:** To explicitly prevent a function from accessing any global variables:
        ```
        <global.none>
        ```
        This is the **default behavior** if no global declaration is provided for a function.

    - **Placement:**
    A global declaration must immediately precede the `function` definition it applies to.

    ### Assignment Statements

    - **Global Assignments:**
    A top-level assignment statement uses the syntax:
    ```
    variable = expression
    ```
    The left-hand side (`variable`) must not be a built-in constant or otherwise reserved. The expression on the right-hand side is converted to its postfix form. Internally, the assignment is marked by appending an exclamation mark (`!`) to indicate that the result is stored in that variable.

    - **Variable Usage Rules:**
    Variables must be defined (assigned) before they are referenced in expressions. Otherwise, a syntax error will be raised.

    - **Naming Restrictions:**
    Variables, function names, and parameters must not use reserved names (such as built-in constants or names beginning with `__internal_`).

    - **Global Dependencies:**
    For functions that use global variables (declared using `<global<...>>` or `<global.all>`), the referenced global variables must be defined in the global scope before any call to that function.

    - **Function Return:**
    Each function definition must have exactly one return statement, and that return must be the last statement in the function.

    - **Argument Counts:**
    Function calls (both built-in and custom) check that the exact number of required arguments is provided; otherwise, a syntax error is raised.

    ---

    ## Special Constructs

    ### Frame Property Access

    - **Syntax:**
    To access a frame property from a source clip, use dot notation:
    ```
    clip.propname
    ```
    - `clip` must be a valid source clip.
    - `propname` is the name of the property.

    ### Static Relative Pixel Access

    - **Syntax:**
    For accessing a pixel value relative to a source clip, the expression can take the following form:
    ```
    clip[statX, statY]
    ```
    where:
    - `clip` is a valid source clip,
    - `statX` and `statY` are integer literals specifying the x and y offsets, and
    - An optional suffix (`:m` or `:c`) may follow the closing bracket.

    - **Validation:**
    The indices must be numeric constants (no expressions allowed inside the brackets).

    ---

    ## Operator Precedence and Expression Parsing

    - **Precedence Determination:**
    When converting infix to postfix, the parser searches the expression (tracking parenthesis nesting) for a binary operator at the outer level. The operators are considered in the following order (which effectively sets their precedence):

    1. Logical OR: `||`
    2. Logical AND: `&&`
    3. Bitwise Operators: `&`, `|`, `^`
    4. Relational: `<`, `<=`, `>`, `>=`
    5. Equality: `==`
    6. Inequality: `!=`
    7. Addition and Subtraction: `+`, `-`
    8. Multiplication, Division, and Modulus: `*`, `/`, `%`
    9. Exponentiation: `**`

    The conversion function finds the **last occurrence** of an operator at the outer level for splitting the expression. Parentheses can be used to override this behavior.

    - **Ternary Operator:**
    The ternary operator (`? :`) is detected after other operators have been processed.

    ---

    ## Error Checks and Restrictions

    - **Semicolon Usage:**
    The input may not contain semicolons.

    - **Naming Restrictions:**
    Variables, function names, and parameters must not use reserved names (such as built-in constants or names beginning with `__internal_`).

    - **Global Dependencies:**
    For functions that use global variables (declared using `<global<...>>` or `<global.all>`), the referenced global variables must be defined in the global scope before any call to that function.

    - **Function Return:**
    Each function definition must have exactly one return statement, and that return must be the last statement in the function.

    - **Argument Counts:**
    Function calls (both built-in and custom) check that the exact number of required arguments is provided; otherwise, a syntax error is raised.
    """

    # Reject user input that contains semicolons.
    if ";" in infix_code:
        raise SyntaxError(
            "User input code cannot contain semicolons. Use newlines instead."
        )

    # Remove comments
    infix_code = "\n".join(
        line.split("#", 1)[0].rstrip() for line in infix_code.split("\n")
    )

    # Check for duplicate function definitions.
    find_duplicate_functions(infix_code)

    # Process global declarations.
    # global declaration syntax: <global<var1><var2>...>
    # Also record for functions if global declaration appears immediately before function definition.
    declared_globals: dict[str, int] = (
        {}
    )  # mapping: global variable -> declaration line number
    global_vars_for_functions: dict[str, set[str]] = {}
    global_mode_for_functions: dict[str, GlobalMode] = {}  # "all", "none", "specific"
    lines = infix_code.split("\n")
    modified_lines: list[str] = []

    # Build a mapping from line number to function name for function declarations.
    function_lines = {}
    for i, line in enumerate(lines):
        func_match = func_pattern.match(line.strip())
        if func_match:
            func_name = func_match.group(1)
            function_lines[i] = func_name

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        global_decl_match = global_decl_pattern.match(line)
        if global_decl_match:
            content = global_decl_match.group(1).strip()
            # Extract all global variable names from this line.
            j = i + 1
            # If the next non-global line is a function definition, apply these globals for that function.
            if j < len(lines):
                function_match = function_def_pattern.match(lines[j])
                if function_match:
                    func_name = function_match.group(1)
                    args = function_match.group(2)
                    if content == ".all":
                        global_mode_for_functions[func_name] = GlobalMode.ALL
                    elif content == ".none":
                        global_mode_for_functions[func_name] = GlobalMode.NONE
                    else:
                        global_mode_for_functions[func_name] = GlobalMode.SPECIFIC
                        globals_list = global_match_pattern.findall(content)

                        if not globals_list and content:
                            raise SyntaxError(
                                f"Invalid global declaration syntax: <global{content}>",
                                i + 1,
                            )
                        for gv in globals_list:
                            if gv not in declared_globals:
                                declared_globals[gv] = (
                                    i + 1
                                )  # store declaration line number (1-indexed)
                        if func_name not in global_vars_for_functions:
                            global_vars_for_functions[func_name] = set()
                        global_vars_for_functions[func_name].update(globals_list)
                        function_params = parse_args(args)
                        if any(
                            global_var in function_params for global_var in globals_list
                        ):
                            raise SyntaxError(
                                "Function param must not duplicate with global declarations.",
                                j,
                            )
                else:
                    raise SyntaxError(
                        "Global declaration must be followed by a function definition.",
                        j,
                    )
            else:
                raise SyntaxError(
                    "Global declaration must not be at the last line of code.",
                    j,
                )
            # Replace the global declaration lines with empty line.
            modified_lines.append("\n")
            i += 1
        else:
            modified_lines.append(lines[i])
            i += 1

    expanded_code = "\n".join(modified_lines)

    # Replace function definitions with newlines to preserve line numbers.
    functions: dict[str, tuple[list[str], str, int, set[str]]] = {}

    def replace_function(match: re.Match[str]) -> str:
        func_name = match.group(1)
        params_str = match.group(2)
        body = match.group(3)
        func_start_index = match.start()
        line_num = 1 + expanded_code[:func_start_index].count("\n")
        params = [p.strip() for p in params_str.split(",") if p.strip()]

        try:
            dup1, dup2, dupc = reduce(  # type: ignore
                lambda a, i: (
                    a  # type: ignore
                    if a[1] is not None  # type: ignore
                    else (
                        {**a[0], i[1]: i[0]} if i[1] not in a[0] else a[0],  # type: ignore
                        (a[0][i[1]], i[0], i[1]) if i[1] in a[0] else None,  # type: ignore
                    )
                ),
                enumerate(params),
                ({}, None),  # type: ignore
            )[1]
            if any([dup1, dup2, dupc]):  # type: ignore
                raise SyntaxError(
                    f"{dup1}th argument and {dup2}th argument '{dupc}' duplicated in function definition.",
                    line_num,
                )
        except (TypeError, UnboundLocalError):
            pass

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

    cleaned_code = function_pattern.sub(replace_function, expanded_code)

    # Process global code by splitting on physical newlines.
    global_statements: list[tuple[str, int]] = []
    for i, line in enumerate(cleaned_code.splitlines(), start=1):
        if line.strip():
            global_statements.append((line.strip(), i))

    # Record global assignments with line numbers.
    global_assignments: dict[str, int] = {}
    # current_globals holds the set of global variables defined so far (by assignment).
    current_globals: set[str] = set()
    postfix_tokens: list[str] = []

    # Process global statements in order and check function call global dependencies.
    for stmt, line_num in global_statements:
        # Process assignment statements.
        if assign_pattern.search(stmt):
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
            # If the right-hand side is a function call, check that its global dependencies are defined.
            m_call = m_call_pattern.match(expr)
            if m_call:
                func_name = m_call.group(1)
                if func_name in functions:
                    func_global_mode = global_mode_for_functions.get(
                        func_name, GlobalMode.NONE
                    )
                    if func_global_mode == GlobalMode.SPECIFIC:
                        for gv in functions[func_name][3]:
                            if gv not in current_globals:
                                raise SyntaxError(
                                    f"Global variable '{gv}' used in function '{func_name}' is not defined before its first call.",
                                    line_num,
                                    func_name,
                                )
                    elif func_global_mode == GlobalMode.ALL:
                        pass  # All globals are implicitly available
            # Check self-reference in definition.
            if var_name not in current_globals and re.search(
                r"\b" + re.escape(var_name) + r"\b", expr
            ):
                raise SyntaxError(
                    f"Variable '{var_name}' used before definition", line_num
                )
            # Record the assignment (only record the first assignment).
            if var_name not in global_assignments:
                global_assignments[var_name] = line_num
            current_globals.add(var_name)
            postfix_expr = convert_expr(
                expr, current_globals, functions, line_num, global_mode_for_functions
            )
            postfix_tokens.append(f"{postfix_expr} {var_name}!")
        else:
            # For standalone expression statements, check if they directly call a function.
            m_call = m_call_pattern.match(stmt)
            if m_call:
                func_name = m_call.group(1)
                if func_name in functions:
                    func_global_mode = global_mode_for_functions.get(
                        func_name, GlobalMode.NONE
                    )
                    if func_global_mode == GlobalMode.SPECIFIC:
                        for gv in functions[func_name][3]:
                            if gv not in current_globals:
                                raise SyntaxError(
                                    f"Global variable '{gv}' used in function '{func_name}' is not defined before its first call.",
                                    line_num,
                                    func_name,
                                )
                    elif func_global_mode == GlobalMode.ALL:
                        pass
            postfix_expr = convert_expr(
                stmt, current_globals, functions, line_num, global_mode_for_functions
            )
            if compute_stack_effect(postfix_expr, line_num) != 0:
                raise SyntaxError(f"Unused global expression: {stmt}", line_num)
            postfix_tokens.append(postfix_expr)

    # Check that all declared global variables are defined.
    for gv, decl_line in declared_globals.items():
        if gv not in global_assignments:
            raise SyntaxError(
                f"Global variable '{gv}' declared but not defined.", decl_line
            )

    final_result = " ".join(postfix_tokens)
    final_result = final_result.replace("(", "").replace(")", "")
    if "RESULT!" not in final_result:
        raise SyntaxError("Final result must be assigned to variable 'RESULT!'")

    optimized = optimize_akarin_expr(final_result + " RESULT@")
    return optimized


func_call_pattern = re.compile(r"(\w+)\s*\(")
clip_pattern = re.compile(r"(?:[a-zA-Z]|src\d+)$")
func_info_pattern = re.compile(r"__internal_([a-zA-Z_]\w*)_([a-zA-Z_]\w+)$")
func_pattern = re.compile(r"function\s+(\w+)")
global_decl_pattern = re.compile(r"^<global(.*)>$")
function_def_pattern = re.compile(r"^\s*function\s+(\w+)\s*\(([^)]*)\)\s*\{")
m_call_pattern = re.compile(r"^(\w+)\s*\(")
prop_access_pattern = re.compile(r"^(\w+)\.([a-zA-Z_]\w*)$")
prop_access_generic_pattern = re.compile(r"([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)")
number_pattern = re.compile(
    r"^(0x[0-9A-Fa-f]+(\.[0-9A-Fa-f]+(p[+\-]?\d+)?)?|0[0-7]*|[+\-]?(\d+(\.\d+)?([eE][+\-]?\d+)?))$"
)
nth_pattern = re.compile(r"^nth_(\d+)$")
m_line_pattern = re.compile(r"^([a-zA-Z_]\w*)\s*=\s*(.+)$")
m_static_pattern = re.compile(r"^(\w+)\[(-?\d+),\s*(-?\d+)\](\:\w)?$")
find_duplicate_functions_pattern = re.compile(r"\bfunction\s+(\w+)\s*\(.*?\)")
drop_pattern = re.compile(r"drop(\d*)")
sort_pattern = re.compile(r"sort(\d+)")
global_match_pattern = re.compile(r"<([a-zA-Z_]\w*)>")
assign_pattern = re.compile(r"(?<![<>!])=(?![=])")
rel_pattern = re.compile(r"\w+\[(.*?)\]")
func_sub_pattern = re.compile(r"\w+\([^)]*\)|\[[^\]]*\]")
identifier_pattern = re.compile(r"\b([a-zA-Z_]\w*)\b(?!\s*\()")
letter_pattern = re.compile(r"[a-zA-Z_]\w*")
function_pattern = re.compile(
    r"function\s+(\w+)\s*\(([^)]*)\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
)
build_in_func_patterns = [
    re.compile(r)
    for r in [rf"^{prefix}\d+$" for prefix in ["nth_", "sort", "dup", "drop", "swap"]]
]


class SyntaxError(Exception):
    """Custom syntax error class with line information"""

    def __init__(
        self,
        message: str,
        line_num: Optional[int] = None,
        function_name: Optional[str] = None,
    ):
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


@lru_cache
def is_clip(token: str) -> bool:
    """
    Determine whether token is a source clip.
    (std.Expr: single letter; or akarin.Expr: srcN)
    """
    return bool(clip_pattern.fullmatch(token))


@lru_cache
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
        "pi",
    }
    if token in constants_set:
        return True
    if is_clip(token):
        return True
    return False


@lru_cache
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


@lru_cache
def match_full_function_call(expr: str) -> Optional[tuple[str, str]]:
    """
    Try to match a complete function call with proper nesting.
    Returns (func_name, args_str) if matched; otherwise returns None.
    """
    expr = expr.strip()
    m = func_call_pattern.match(expr)
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


@lru_cache
def extract_function_info(
    internal_var: str, current_function: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Given a renamed internal variable (i.e. __internal_funcname_varname),
    extract the original function name and variable name.
    """
    if current_function is not None and internal_var.startswith(
        f"__internal_{current_function}_"
    ):
        return current_function, internal_var[len(f"__internal_{current_function}_") :]
    match = func_info_pattern.match(internal_var)
    if match:
        return match.group(1), match.group(2)
    return None, None


def find_duplicate_functions(code: str):
    """
    Check if any duplicate function is defined.
    """
    function_lines = {}
    lines = code.split("\n")
    for line_num, line in enumerate(lines, start=1):
        match = find_duplicate_functions_pattern.search(line)
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

    stack: list[int] = []
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

        # dropN operator (default is drop1)
        m_drop = drop_pattern.fullmatch(token)
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

        # sortN operator: reorder top N items without changing the stack count.
        m_sort = sort_pattern.fullmatch(token)
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


def parse_ternary(
    expr: str, line_num: Optional[int] = None, func_name: Optional[str] = None
) -> Optional[tuple[str, str, str]]:
    """
    For a given expression string without delimiters, detects the presence of a ternary operator in the outer layer.
    If present, returns a triple (condition, true_expr, false_expr); otherwise returns None.
    """
    expr = expr.strip()
    n = len(expr)
    depth = 0
    q_index = -1
    for i, c in enumerate(expr):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "?" and depth == 0:
            q_index = i
            break
    if q_index == -1:
        return None

    condition = expr[:q_index].strip()

    ternary_level = 1
    depth = 0
    colon_index = -1
    for i in range(q_index + 1, n):
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "?" and depth == 0:
            ternary_level += 1
        elif c == ":" and depth == 0:
            ternary_level -= 1
            if ternary_level == 0:
                colon_index = i
                break
    if colon_index == -1:
        raise SyntaxError("Ternary operator missing ':'", line_num, func_name)
    true_expr = expr[q_index + 1 : colon_index].strip()
    false_expr = expr[colon_index + 1 :].strip()
    return (condition, true_expr, false_expr)


def validate_static_relative_pixel_indices(
    expr: str, line_num: int, function_name: Optional[str] = None
) -> None:
    """
    Validate that the static relative pixel access indices are numeric constants.
    """
    static_relative_pixel_indices = rel_pattern.finditer(expr)
    for match in static_relative_pixel_indices:
        indices = match.group(1).split(",")
        for idx in indices:
            if not number_pattern.fullmatch(idx.strip()):
                raise SyntaxError(
                    f"Static relative pixel access index must be a numeric constant, got '{idx.strip()}'",
                    line_num,
                    function_name,
                )


def check_variable_usage(
    expr: str,
    variables: set[str],
    line_num: int,
    function_name: Optional[str] = None,
    local_vars: Optional[set[str]] = None,
) -> None:
    """
    Check that all variables in the expression have been defined.
    In function scope, if local_vars is provided, only local variables are checked.
    """

    def prop_repl(m: re.Match[str]) -> str:
        clip_candidate = m.group(1)
        if is_clip(clip_candidate):
            return " "
        return m.group(0)

    expr = prop_access_generic_pattern.sub(prop_repl, expr)
    expr = func_sub_pattern.sub("", expr)
    identifiers = identifier_pattern.finditer(expr)
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
    variables: set[str],
    functions: dict[str, tuple[list[str], str, int, set[str]]],
    line_num: int,
    global_mode_for_functions: dict[str, GlobalMode],
    current_function: Optional[str] = None,
    local_vars: Optional[set[str]] = None,
) -> str:
    """
    Convert a single infix expression to a postfix expression.
    Supports binary and unary operators, function calls, and custom function definitions.
    """
    expr = expr.strip()

    prop_access_match = prop_access_pattern.match(expr)
    if prop_access_match:
        clip_name = prop_access_match.group(1)
        if is_clip(clip_name):
            return expr

    # Check variable usage and validate static relative pixel indices.
    check_variable_usage(expr, variables, line_num, current_function, local_vars)
    validate_static_relative_pixel_indices(expr, line_num, current_function)

    if number_pattern.match(expr):
        return expr

    constants_map = {
        "current_frame_number": "N",
        "current_x": "X",
        "current_y": "Y",
        "current_width": "width",
        "current_height": "height",
        "pi": "pi",
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
        m_nth = nth_pattern.match(func_name)
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
                    arg,
                    variables,
                    functions,
                    line_num,
                    global_mode_for_functions,
                    current_function,
                    local_vars,
                )
                for arg in args
            ]
            sort_token = f"sort{M}"
            drop_before = f"drop{N_val-1}" if (N_val - 1) > 0 else ""
            drop_after = f"drop{M-N_val}" if (M - N_val) > 0 else ""
            tokens: list[str] = []
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
                arg,
                variables,
                functions,
                line_num,
                global_mode_for_functions,
                current_function,
                local_vars,
            )
            for arg in args
        ]

        if func_name == "dyn":
            if len(args) != 3:
                raise SyntaxError(
                    f"Built-in function {func_name} requires 3 arguments, but {len(args)} were provided.",
                    line_num,
                    current_function,
                )
            if not is_clip(args[0]):
                raise SyntaxError(
                    f"{args[0]} is not a source clip.", line_num, current_function
                )
            return f"{args_postfix[1]} {args_postfix[2]} {args[0]}[]"

        if func_name in ["min", "max"]:
            if len(args) != 2:
                raise SyntaxError(
                    f"Built-in function {func_name} requires 2 arguments, but {len(args)} were provided.",
                    line_num,
                    current_function,
                )
            return f"{args_postfix[0]} {args_postfix[1]} {func_name}"
        elif func_name == "clamp":
            if len(args) != 3:
                raise SyntaxError(
                    f"Built-in function {func_name} requires 3 arguments, but {len(args)} were provided.",
                    line_num,
                    current_function,
                )
            return f"{args_postfix[0]} {args_postfix[1]} {args_postfix[2]} clamp"
        else:
            builtin_unary = [
                "sin",
                "cos",
                "log",
                "exp",
                "round",
                "floor",
                "abs",
                "sqrt",
                "trunc",
                "bitnot",
                "not",
            ]
            if func_name in builtin_unary:
                if len(args) != 1:
                    raise SyntaxError(
                        f"Built-in function {func_name} requires 1 argument, but {len(args)} were provided.",
                        line_num,
                        current_function,
                    )
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

            local_map: dict[str, str] = {}
            func_global_mode = global_mode_for_functions.get(func_name, GlobalMode.NONE)
            effective_globals = set[str]()
            if func_global_mode == GlobalMode.ALL:
                effective_globals = variables
            elif func_global_mode == GlobalMode.SPECIFIC:
                effective_globals = global_vars

            for offset, body_line in enumerate(body_lines):
                if body_line.startswith("return"):
                    continue
                m_line = m_line_pattern.match(body_line)
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
                        and var not in effective_globals
                    ):
                        local_map[var] = f"__internal_{func_name}_{var}"

            rename_map: dict[str, str] = {}
            rename_map.update(param_map)
            rename_map.update(local_map)
            new_local_vars = (
                set(param_map.keys())
                .union(set(local_map.keys()))
                .union(effective_globals)
            )
            param_assignments: list[str] = []
            for i, p in enumerate(params):
                arg_orig = args[i].strip()
                if is_constant(arg_orig):
                    rename_map[p] = args_postfix[i]
                else:
                    if p not in effective_globals:
                        param_assignments.append(f"{args_postfix[i]} {rename_map[p]}!")
            new_lines: list[str] = []
            for line_text in body_lines:
                new_line = line_text
                for old, new in rename_map.items():
                    if old not in effective_globals:
                        new_line = re.sub(
                            rf"(?<!\w){re.escape(old)}(?!\w)", new, new_line
                        )
                new_lines.append(new_line)
            function_tokens: list[str] = []
            return_count = 0
            for offset, body_line in enumerate(new_lines):
                effective_line_num = func_line_num + offset
                if body_line.startswith(
                    "return"
                ):  # Return does nothing, but it looks better to have one.
                    return_count += 1
                    ret_expr = body_line[len("return") :].strip()
                    function_tokens.append(
                        convert_expr(
                            ret_expr,
                            variables,
                            functions,
                            effective_line_num,
                            global_mode_for_functions,
                            func_name,
                            new_local_vars,
                        )
                    )
                # Process assignment statements.
                elif assign_pattern.search(body_line):
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
                        f"{convert_expr(expr_line, variables, functions, effective_line_num, global_mode_for_functions, func_name, new_local_vars)} {var_name}!"
                    )
                else:
                    function_tokens.append(
                        convert_expr(
                            body_line,
                            variables,
                            functions,
                            effective_line_num,
                            global_mode_for_functions,
                            func_name,
                            new_local_vars,
                        )
                    )
            if return_count > 1:
                raise SyntaxError(
                    f"Function {func_name} must return at most one value, got {return_count}",
                    func_line_num,
                    func_name,
                )
            result_expr = " ".join(param_assignments + function_tokens)
            net_effect = compute_stack_effect(result_expr, line_num, func_name)
            if return_count == 1:
                if net_effect != 1:
                    raise SyntaxError(
                        f"The return value stack of function {func_name} is unbalanced; expected 1 but got {net_effect}.",
                        func_line_num,
                        func_name,
                    )
            elif net_effect != 0:
                raise SyntaxError(
                    f"The function {func_name} should not return a value, but stack is not empty. Stack effect: {net_effect}.",
                    func_line_num,
                    func_name,
                )
            return result_expr

    stripped = strip_outer_parentheses(expr)
    if stripped != expr:
        return convert_expr(
            stripped,
            variables,
            functions,
            line_num,
            global_mode_for_functions,
            current_function,
            local_vars,
        )

    m_static = m_static_pattern.match(expr)
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

    ternary_parts = parse_ternary(expr, line_num, current_function)
    if ternary_parts is not None:
        cond, true_expr, false_expr = ternary_parts
        cond_conv = convert_expr(
            cond,
            variables,
            functions,
            line_num,
            global_mode_for_functions,
            current_function,
            local_vars,
        )
        true_conv = convert_expr(
            true_expr,
            variables,
            functions,
            line_num,
            global_mode_for_functions,
            current_function,
            local_vars,
        )
        false_conv = convert_expr(
            false_expr,
            variables,
            functions,
            line_num,
            global_mode_for_functions,
            current_function,
            local_vars,
        )
        return f"{cond_conv} {true_conv} {false_conv} ?"

    operators = [
        ("||", "or"),
        ("&&", "and"),
        ("|", "bitor"),
        ("^", "bitxor"),
        ("&", "bitand"),
        ("<", "<"),
        ("<=", "<="),
        (">", ">"),
        (">=", ">="),
        ("==", "="),
        ("!=", "!="),
        ("+", "+"),
        ("-", "-"),
        ("*", "*"),
        ("/", "/"),
        ("%", "%"),
        ("**", "pow"),
    ]
    for op_str, postfix_op in operators:
        left, right = find_binary_op(expr, op_str)
        if left is not None and right is not None:
            left_postfix = convert_expr(
                left,
                variables,
                functions,
                line_num,
                global_mode_for_functions,
                current_function,
                local_vars,
            )
            right_postfix = convert_expr(
                right,
                variables,
                functions,
                line_num,
                global_mode_for_functions,
                current_function,
                local_vars,
            )
            if letter_pattern.fullmatch(
                left.strip()
            ) and not left_postfix.strip().endswith("@"):
                left_postfix = left_postfix.strip() + (
                    "@" if not is_constant(left_postfix) else ""
                )
            if letter_pattern.fullmatch(
                right.strip()
            ) and not right_postfix.strip().endswith("@"):
                right_postfix = right_postfix.strip() + (
                    "@" if not is_constant(right_postfix) else ""
                )
            return f"{left_postfix} {right_postfix} {postfix_op}"

    if expr.startswith("!"):
        operand = convert_expr(
            expr[1:],
            variables,
            functions,
            line_num,
            global_mode_for_functions,
            current_function,
            local_vars,
        )
        return f"{operand} not"

    if expr.startswith("-"):
        operand = convert_expr(
            expr[1:],
            variables,
            functions,
            line_num,
            global_mode_for_functions,
            current_function,
            local_vars,
        )
        return f"{operand} -1 *"

    if is_constant(expr):
        return expr
    if letter_pattern.fullmatch(expr):
        return expr if expr.endswith("@") else expr + "@"

    return expr


def find_binary_op(expr: str, op: str):
    """
    Find the last occurrence of the binary operator op at the outer level and return the left and right parts.
    """
    if not op:
        return None, None

    prefix = [0] * len(op)
    j = 0
    for i in range(1, len(op)):
        while j > 0 and op[i] != op[j]:
            j = prefix[j - 1]
        if op[i] == op[j]:
            j += 1
            prefix[i] = j

    candidate_index = -1
    level = 0
    kmp_state = 0
    levels = [0] * len(expr)

    for i, c in enumerate(expr):
        if c == "(":
            level += 1
        elif c == ")":
            level -= 1
        levels[i] = level

        while kmp_state > 0 and c != op[kmp_state]:
            kmp_state = prefix[kmp_state - 1]
        if c == op[kmp_state]:
            kmp_state += 1

        if kmp_state == len(op):
            candidate = i - len(op) + 1
            if levels[candidate] == 0:
                left_valid = (candidate == 0) or (expr[candidate - 1] in " (,\t")
                after = candidate + len(op)
                right_valid = (after == len(expr)) or (expr[after] in " )\t,")
                if left_valid and right_valid:
                    candidate_index = candidate
            kmp_state = prefix[kmp_state - 1]

    if candidate_index == -1:
        return None, None

    left = expr[:candidate_index].strip()
    right = expr[candidate_index + len(op) :].strip()
    return left, right


def parse_args(args_str: str) -> list[str]:
    """
    Parse the arguments of a function call.
    """
    args: list[str] = []
    current: list[str] = []
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
        "log",
        "exp",
        "round",
        "floor",
        "abs",
        "sqrt",
        "trunc",
        "bitnot",
        "not",
    ]
    builtin_binary = ["min", "max"]
    builtin_ternary = ["clamp", "dyn"]
    if any(r.match(func_name) for r in build_in_func_patterns):
        return True
    return (
        func_name in builtin_unary
        or func_name in builtin_binary
        or func_name in builtin_ternary
    )
