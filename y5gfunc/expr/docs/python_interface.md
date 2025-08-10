## 1. Introduction

### 1.1. Purpose

The goal of this generator is to allow you to write complex pixel-processing logic using familiar, readable Python syntax. Instead of writing raw, hard-to-maintain postfix notation (Reverse Polish Notation), you can leverage Python's operators, functions, and programmatic structures to build your expression, which is then transpiled into the target DSL string.

### 1.2. Core Concepts

The generator works on the principle of **Lazy Evaluation** using an **Expression Tree**.

1.  **`DSLExpr` Objects**: Every variable, constant, or result of an operation in this library is a special `DSLExpr` object, not a plain Python value.
2.  **Building an Expression Tree**: When you write `c = a + b`, you are not actually performing addition. You are creating a "Binary Operation" node in a tree that records that `a` should be added to `b`.
3.  **Generating the DSL**: The final DSL script is only generated when you call the `.dsl` property on the final `DSLExpr` object. This call traverses the entire expression tree and translates it into the correct sequence of DSL statements.

## 2. Getting Started: A Basic Example

This example takes a source clip, increases its brightness by 10, and generates the corresponding script.

```python
# 1. Import the necessary components
from y5gfunc.expr.python2infix import SourceClip

# 2. Define a source clip. 'a' is an alias for $src3 in the DSL.
clip_a = SourceClip('a')

# 3. Create a new expression. This builds a tree, it does not calculate anything.
brighter_clip = clip_a + 10

# 4. Generate the final DSL string by accessing the .dsl property
dsl_script = brighter_clip.dsl

# 5. Print the result
print(dsl_script)
```

**Generated DSL:**

```
v_0 = ($src3 + 10)
RESULT = v_0
```

As you can see, the intermediate variable is named `v_0`, not `brighter_clip`. This is the default behavior, designed for maximum performance. See the next section for details on how to generate more readable code.

## 2.1. Performance and Readability (`varname_toggle`)

By default, the generator does **not** use your Python variable names in the output DSL. This is a deliberate design choice to maximize performance, as inspecting the Python call stack to retrieve variable names can be slow, especially for complex expressions.

The default output uses simple, auto-incrementing names like `v_0`, `v_1`, etc.

If you need more readable DSL for debugging or inspection, you can temporarily enable variable name detection using the `varname_toggle` context manager.

**Example:**

```python
from y5gfunc.expr import python2infix
from y5gfunc.expr.python2infix import SourceClip

clip_a = SourceClip('a')

# Enable varname detection just for this block
with python2infix.varname_toggle(True):
    brighter_clip = clip_a + 10
    much_brighter_clip = brighter_clip * 1.5

dsl_script = much_brighter_clip.dsl
print(dsl_script)
```

**Generated DSL with `varname_toggle(True)`:**

```
brighter_clip_0 = ($src3 + 10)
much_brighter_clip_0 = (brighter_clip_0 * 1.5)
RESULT = much_brighter_clip_0
```

> **Note:** It is recommended to use `varname_toggle(True)` only when needed, as it can impact the generation speed of very large and complex expression trees.

## 3. API Reference

### 3.1. Source Clips (`SourceClip`)

Source clips are the primary inputs to your expression, representing the video data.

- **Creation**: `SourceClip(identifier)`
- **Identifier**: Can be a lowercase letter alias (`'x'`, `'y'`, `'z'`, `'a'`-`'w'`) or an integer.
  - `SourceClip('x')` -> `$src0`
  - `SourceClip('a')` -> `$src3`
  - `SourceClip(4)` -> `$src4`
  - `SourceClip('src5')` -> `$src5`

```python
# These are equivalent
clip_x = SourceClip('x')
clip_0 = SourceClip(0)
clip_src0 = SourceClip('src0')
```

### 3.2. Built-in Constants (`Constant`)

The generator provides access to all built-in DSL constants through the `Constant` namespace.

- **Usage**: `Constant.constant_name`

**Available Constants:**

| Python Code       | DSL Equivalent |
| ----------------- | -------------- |
| `Constant.pi`     | `$pi`          |
| `Constant.N`      | `$N`           |
| `Constant.X`      | `$X`           |
| `Constant.Y`      | `$Y`           |
| `Constant.width`  | `$width`       |
| `Constant.height` | `$height`      |

**Example:**

```python
# Represents the expression: `$X / $width`
normalized_x = Constant.X / Constant.width
```

### 3.3. Operators

Most standard Python operators are overloaded to build expression nodes. They are listed below in order of precedence (lowest to highest).

| Category       | Python Operator      | DSL Operator | Example                       |
| -------------- | -------------------- | ------------ | ----------------------------- |
| **Logical**    | `or`                 | `||`         | `(a > 10) || (b < 20)`        |
|                | `and`                | `&&`         | `(a > 10) && (b < 20)`        |
| **Bitwise**    | `~` (Invert)         | `~`          | `~a`                          |
|                | _See Note 1_         | `|`, `^`, `&`| `BuiltInFunc.bitwise_or(a, b)`|
| **Relational** | `<`, `<=`, `>`, `>=` | same         | `a > b`                       |
| **Equality**   | `==`, `!=`           | same         | `a == 128`                    |
| **Arithmetic** | `+`, `-`             | same         | `a + b`                       |
|                | `*`, `/`             | same         | `a * 1.5`                     |
|                | `%`                  | same         | `a % 2`                       |
|                | `**`                 | same         | `a ** 2`                      |
| **Unary**      | `-` (Negation)       | `-`          | `-a`                          |
|                | _See Note 2_         | `!`          | `BuiltInFunc.logical_not(a)`  |

> **Note 1:** Because `&`, `|`, and `^` are used for logical AND/OR and other purposes, the DSL's bitwise operators are available as explicit functions in `BuiltInFunc`:
> `BuiltInFunc.bitwise_and(a, b)`, `BuiltInFunc.bitwise_or(a, b)`, `BuiltInFunc.bitwise_xor(a, b)`

> **Note 2:** Python's `not` keyword cannot be overloaded. Use `BuiltInFunc.logical_not(a)` for logical negation.

### 3.4. Functions (`BuiltInFunc`)

All built-in DSL functions are available as static methods in the `BuiltInFunc` class.

**Common Functions:**

- `BuiltInFunc.sin(expr)`
- `BuiltInFunc.cos(expr)`
- `BuiltInFunc.sqrt(expr)`
- `BuiltInFunc.abs(expr)`
- `BuiltInFunc.round(expr)`
- `BuiltInFunc.floor(expr)`
- `BuiltInFunc.min(expr1, expr2)`
- `BuiltInFunc.max(expr1, expr2)`
- `BuiltInFunc.clamp(value, min_val, max_val)`
- `BuiltInFunc.log(expr)`
- ......

**Special Function: `sort`**

The `nth_N` family of functions is accessed via `BuiltInFunc.sort`. It returns a list of `DSLExpr` objects, each representing the Nth smallest value in the input list.

- **Usage**: `BuiltInFunc.sort([expr1, expr2, ...])[N]`
- `[0]` maps to `nth_1`, `[1]` to `nth_2`, and so on.

**Example:**

```python
a, b, c = SourceClip('a'), SourceClip('b'), SourceClip('c')

# Get the median of three values (the 2nd smallest)
median = BuiltInFunc.sort([a, b, c])[1]
# Generated DSL contains: nth_2($src3, $src4, $src5)
```

### 3.5. Conditional Logic (`if_then_else`)

**You cannot use Python's native `if` statements or ternary operator.** This will raise a `TypeError` with an informative message.

To implement conditional logic, use `BuiltInFunc.if_then_else`, which maps directly to the DSL's `? :` ternary operator.

- **Syntax**: `BuiltInFunc.if_then_else(condition, true_expression, false_expression)`

**Example:**

```python
# DO THIS:
clip_a = SourceClip('a')
# If clip_a is dark, double its value; otherwise, keep it.
result = BuiltInFunc.if_then_else(clip_a < 50, clip_a * 2, clip_a)

# DO NOT DO THIS (will raise an error):
# if clip_a < 50:
#     result = clip_a * 2
# else:
#     result = clip_a
```

### 3.6. Clip-Specific Operations

#### Static Pixel Access (`[...]`)

To access pixels from a source clip at a fixed, constant offset from the current coordinate (`X`, `Y`), use item access (`[...]`) on a `SourceClip` object. This maps directly to the DSL's static relative pixel access feature.

- **Syntax**: `clip[offset_x, offset_y]`
- `offset_x` and `offset_y` **must be Python integers**. Expressions are not allowed.

**Example:**

```python
clip_a = SourceClip('a')
# Access the pixel from clip 'a' 1 pixel to the right and 1 pixel down
neighbor_pixel = clip_a[1, 1]
# Generates DSL: ... = $src3[1,1]
```

#### Dynamic Pixel Access (`.access()`)

To dynamically access pixels from a clip using absolute coordinates (which can be expressions), use the `.access()` method on a `SourceClip` object. This maps to the DSL's `dyn` function.

- **Syntax**: `clip.access(x_coordinate, y_coordinate)`

**Example:**

```python
clip_a = SourceClip('a')
# Access pixel from clip 'a' 10 pixels to the right and 5 pixels up from the current pixel
shifted_pixel = clip_a.access(Constant.X + 10, Constant.Y - 5)
```

#### Frame Property Access (`.props`)

To access a frame property, use the `.props` property on a `SourceClip` object, followed by item access (`[...]`).

- **Syntax**: `clip.props['PropertyName']`

**Example:**

```python
clip_a = SourceClip('a')

# Access the _Matrix property from the frame properties of clip 'a'
matrix = clip_a.props['_Matrix']

# The result can be used in other expressions
is_bt709 = matrix == 1
```
