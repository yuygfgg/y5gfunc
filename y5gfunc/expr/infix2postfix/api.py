from enum import StrEnum
from .front_end import parse_infix_to_postfix
from .middle_end import optimize_akarin_expr
from .back_end import verify_akarin_expr, verify_std_expr, to_std_expr


class BackEnd(StrEnum):
    AKARIN = "akarin"
    STD = "std"


def compile(expr: str, back_end: BackEnd = BackEnd.AKARIN) -> str:
    R"""
    Convert infix expressions to postfix expressions.

    Args:
        expr: Input infix code.
        back_end: Target backend to run the generated expression.

    Returns:
        Converted postfix expr.

    Raises:
        SyntaxError: If infix code failed to convert to postfix expr.
        ValueError: If the expression is invalid (failed to generate for specified backend).

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
    - `N`, `X`, `Y`, `width`, `height`, `pi`
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
    Each function definition must have at most one return statement, and if it exists, it must be the last statement in the function.

    - **Argument Counts:**
    Function calls (both built-in and custom) check that the exact number of required arguments is provided; otherwise, a syntax error is raised.
    """
    postfix = parse_infix_to_postfix(expr)
    optimized = optimize_akarin_expr(postfix)

    if back_end == BackEnd.STD:
        optimized = to_std_expr(optimized)
        if not verify_std_expr(optimized):
            raise ValueError("Invalid expression")
    elif back_end == BackEnd.AKARIN:
        if not verify_akarin_expr(optimized):
            raise ValueError("Invalid expression")

    return optimize_akarin_expr(optimized)
