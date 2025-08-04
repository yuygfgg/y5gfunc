## 1. Introduction

This function implements a simple domain-specific language designed to be transpiled into postfix notation (Reverse Polish Notation) for use with VapourSynth's `std.Expr` and the enhanced `akarin.Expr`.

The language provides a familiar C-style infix syntax that supports variables, operators, user-defined functions, and special constructs for video processing.
It is designed to be expressive and readable, abstracting away the complexities of writing raw postfix expressions.

### Modes of Operation

The transpiler can target two different backends, which affects the set of available features:

- **Standard Mode (`force_std=True`):** Generates expressions compatible with VapourSynth's built-in `core.std.Expr`. This mode has a restricted feature set.
- **Akarin Mode (`force_std=False`):** (Default) Generates expressions for the more powerful `akarin.Expr`, enabling numerous advanced features.

## 2. Lexical Structure

### 2.1. Comments

Comments begin with a `#` character and extend to the end of the line. They are ignored by the parser.

```
# This is a comment.
a = 1 # This is an inline comment.
```

### 2.2. Whitespace

Whitespace characters (spaces, tabs, newlines) are used to separate tokens. Multiple whitespace characters are treated as a single separator. Newlines are significant as they terminate statements.

### 2.3. Identifiers

Identifiers are used for naming variables and functions.

- **Rules:** Must begin with a letter (`a-z`, `A-Z`) or an underscore (`_`) and can be followed by any number of letters, digits, or underscores. The pattern is `[a-zA-Z_]\w*`.
- **Case-Sensitivity:** Identifiers are case-sensitive. `myVar` and `myvar` are different.
- **Reserved Prefix:** Identifiers starting with `__internal_` are reserved for internal use by the transpiler and must not be used in user code.

### 2.4. Literals

#### Numeric Literals

The language supports several formats for numeric constants:

- **Decimal:** Standard integers (`100`, `-42`) and floating-point numbers (`3.14`, `-0.5`, `1.2e-5`).
- **Hexadecimal:** Prefixed with `0x`. Can include a fractional part and a binary exponent (`p-exponent`). Examples: `0xFF`, `0x1.9p-2`.
- **Octal:** Prefixed with a leading `0`. Example: `0755`.

## 3. Program Structure

A program is a script composed of one or more statements.

### 3.1. Statements

- Statements are written one per line.
- The use of semicolons (`;`) to terminate or separate statements is forbidden and will result in a syntax error.

#### Assignment Statements

An assignment statement computes the value of an expression and stores it in a variable.

- **Syntax:** `variable = expression`
- The stack must be balanced after an assignment (i.e., the expression must result in exactly one value).

#### Expression Statements

A statement can consist of a single expression, such as a function call that does not return a value. The expression is evaluated, and its result is discarded. The net stack effect of such a statement must be zero.

```
# Assignment statement
my_var = 10 * 2

# Expression statement (assuming 'my_func' has no return value)
my_func(my_var)
```

### 3.2. Final Result

To produce a final output from the script, a value must be assigned to the special variable `RESULT`. This assignment is typically the last statement in the global scope.

```
# ... calculations ...
RESULT = final_value
```

## 4. Variables and Constants

### 4.1. Variables

- **Declaration:** Variables are declared implicitly upon their first assignment.
- **Usage:** A variable must be assigned a value before it is used in an expression. Using an undefined variable will result in a syntax error.

### 4.2. Constants

Constants represent fixed values and are **always** identified by a `$` prefix. They are treated as literal values and do not require prior assignment.

### 4.3. Built-in Constants

The language provides several built-in constants.

| Constant  | Description                                                 | Availability |
| :-------- | :---------------------------------------------------------- | :----------- |
| `$pi`     | The value of π.                                             | All modes    |
| `$N`      | The current frame number.                                   | Akarin Only  |
| `$X`      | The current column coordinate (chroma-subsampling counted). | Akarin Only  |
| `$Y`      | The current row coordinate (chroma-subsampling counted).    | Akarin Only  |
| `$width`  | The width of the video plane (chroma-subsampling counted).  | Akarin Only  |
| `$height` | The height of the video plane (chroma-subsampling counted). | Akarin Only  |

### 4.4. Source Clips

Source clips are special constants used to reference input video clips. They must be prefixed with a `$`.

Source clips can be referenced in two equivalent ways:

1.  **Single Letters:** The lowercase letters `xyzabcdefghijklmnopqrstuvw` are aliases for `src0` through `src25` respectively:

- `$x` is equivalent to `$src0`
- `$y` is equivalent to `$src1`
- `$z` is equivalent to `$src2`
- `$a` is equivalent to `$src3`
- And so on through `$w` which is equivalent to `$src25`

2.  **`src` Prefixed:** The identifier `src` followed by one or more digits, prefixed with `$` (e.g., `$src0`, `$src1`).

- In **Standard Mode**, only `$src0` through `$src25` (or their letter aliases) are available.
- In **Akarin Mode**, this range is extended beyond `$src25`.

## 5. Operators

Operators are left-associative, except for the unary and ternary operators. The following table lists operators in order of precedence, from lowest to highest.

| Precedence | Operator            | Description              | Arity      | Associativity | Availability |
| :--------- | :------------------ | :----------------------- | :--------- | :------------ | :----------- |
| 1          | `|| `               | Logical OR               | Binary     | Left          | All modes    |
| 2          | `&&`                | Logical AND              | Binary     | Left          | All modes    |
| 3          | `|`                 | Bitwise OR               | Binary     | Left          | Akarin Only  |
|            | `^`                 | Bitwise XOR              | Binary     | Left          | Akarin Only  |
|            | `&`                 | Bitwise AND              | Binary     | Left          | Akarin Only  |
|            | `~`                 | Bitwise NOT              | Unary      | Right         | Akarin Only  |
| 4          | `<`, `<=`, `>` `>=` | Relational               | Binary     | Left          | All modes    |
| 5          | `==`, `!=`          | Equality                 | Binary     | Left          | All modes    |
| 6          | `+`, `-`            | Addition, Subtraction    | Binary     | Left          | All modes    |
| 7          | `*`, `/`            | Multiplication, Division | Binary     | Left          | All modes    |
|            | `%`                 | Modulus                  | Binary     | Left          | Akarin Only  |
| 8          | `**`                | Exponentiation (Power)   | Binary     | Left          | All modes    |
| 9          | `-`                 | Negation                 | Unary      | Right         | All modes    |
|            | `!`                 | Logical NOT              | Unary      | Right         | All modes    |
| 10         | `? :`               | Ternary Conditional      | Ternary    | Right         | All modes    |

Note: Bitwise operators operates on integer values. When operating on floating-point values, operands are first rounded to the nearest integer.

## 6. Functions

### 6.1. Function Calls

Functions are called using standard syntax: `functionName(argument1, argument2, ...)`

### 6.2. Built-in Functions

| Function                   | Arity           | Description                                       | Availability |
| :------------------------- | :-------------- | :------------------------------------------------ | :----------- |
| `sin`, `cos`, `log`, `exp` | 1               | Standard trigonometric and logarithmic functions. | All modes    |
| `sqrt`, `abs`              | 1               | Square root, absolute value.                      | All modes    |
| `trunc`                    | 1               | Truncation.                                       | Akarin Only  |
| `min`, `max`               | 2               | Returns the minimum or maximum of two values.     | All modes    |
| `clamp`                    | 3               | `clamp(x, min_val, max_val)`                      | All modes    |
| `round`, `floor`           | 1               | Round to nearest integer, round down.             | Akarin Only  |
| `dyn`                      | 3               | Dynamic static pixel access. See Section 8.       | Akarin Only  |
| `nth_N`                    | `M` (where M≥N) | `nth_3(a, b, c, d)` finds the 3rd smallest value. | All modes    |

### 6.3. User-Defined Functions

- **Definition:**

```
function functionName(param1, param2) {
    // Body of the function
    local_var = param1 * param2
    return local_var + 10
}
```

- **Return Statement:** A function can have at most one `return` statement, which must be the last non-empty statement in its body. If a function has no `return` statement, it produces no value.
- **Parameters:** Function parameters are read-only and cannot be reassigned within the function body.
- **Inlining:** Function calls are effectively inlined at compile time. Recursion is not supported.
- **Nesting:** Function definitions cannot be nested.

### 6.4. Standard Library

A set of additional mathematical functions are available through the provided `math_functions` utility. These are implemented as user-defined functions and include `atan`, `atan2`, `sinh`, `cosh`, `tgamma`, `erf`, `cbrt`, and more. They can be prepended to user code to be made available.

## 7. Scope and Globals

### 7.1. Scopes

- **Global Scope:** Variables defined at the top level of the script.
- **Function Scope:** Each function has its own local scope. This includes its parameters and any variables assigned within its body.

### 7.2. Global Variable Access

By default, functions operate in an isolated scope and cannot access global variables. This behavior can be modified with a global declaration placed immediately before a function definition.

- **`<global.none>`:** (Default) The function cannot access any global variables.
- **`<global.all>`:** The function can access any global variable defined in the script at the time of the function's call.
- **`<global<var1><var2>...>`:** The function can access only the specified global variables.

**Example:**

```
<global<my_global>>
function useGlobal(x) {
    return x + my_global # Accesses global 'my_global'
}

my_global = 100
RESULT = useGlobal(5) # Evaluates to 105
```

- Any global variable a function depends on must be defined before that function is called.

## 8. Special Constructs (Akarin Mode Only)

### 8.1. Frame Property Access

Read a property from a clip's frame properties.

- **Syntax:** `clip.propname`
- `clip` must be a valid source clip identifier (e.g., `$a`, `$src1`), but the `$` prefix is optional. For example: `$a.propname` or `src1._Matrix`.

### 8.2. Static Relative Pixel Access

Access a pixel from a source clip at a fixed, constant offset from the current coordinate (`X`, `Y`).

- **Syntax:** `$clip[offsetX, offsetY]`
- `$clip` must be a source clip constant.
- `offsetX` and `offsetY` must be integer literals.

### 8.3. Dynamic Absolute Pixel Access

Access a pixel from a source clip at a dynamically calculated coordinate. This is achieved via the `dyn` built-in function.

- **Syntax:** `dyn($clip, x_expr, y_expr)`
- `$clip` must be a source clip constant.
- `x_expr` and `y_expr` can be any valid expressions that evaluate to the desired coordinates.
- `dyn($src0, X + 2, Y + 3)` is equivalent to `$src0[2, 3]`.
