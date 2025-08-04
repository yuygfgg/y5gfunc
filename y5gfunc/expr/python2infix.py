"""
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
from dsl_generator import SourceClip, BuiltInFunc

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
brighter_clip = ($src3 + 10)
RESULT = brighter_clip
```

## 3. API Reference

### 3.1. Source Clips (`SourceClip`)

Source clips are the primary inputs to your expression, representing the video data.

-   **Creation**: `SourceClip(identifier)`
-   **Identifier**: Can be a lowercase letter alias (`'x'`, `'y'`, `'z'`, `'a'`-`'w'`) or an integer.
    -   `SourceClip('x')` -> `$src0`
    -   `SourceClip('a')` -> `$src3`
    -   `SourceClip(4)` -> `$src4`
    -   `SourceClip('src5')` -> `$src5`

```python
# These are equivalent
clip_x = SourceClip('x')
clip_0 = SourceClip(0)
clip_src0 = SourceClip('src0')
```

### 3.2. Built-in Constants (`Constant`)

The generator provides access to all built-in DSL constants through the `Constant` namespace.

-   **Usage**: `Constant.constant_name`

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

| Category      | Python Operator     | DSL Operator | Example                               |
| ------------- | ------------------- | ------------ | ------------------------------------- |
| **Logical**   | `or`                | `||`         | `(a > 10) | (b < 20)`                 |
|               | `and`               | `&&`         | `(a > 10) & (b < 20)`                 |
| **Bitwise**   | `~` (Invert)        | `~`          | `~a`                                  |
|               | *See Note 1*        | `|`, `^`, `&` | `BuiltInFunc.bitwise_or(a, b)`       |
| **Relational**| `<`, `<=`, `>`, `>=`| same         | `a > b`                               |
| **Equality**  | `==`, `!=`          | same         | `a == 128`                            |
| **Arithmetic**| `+`, `-`            | same         | `a + b`                               |
|               | `*`, `/`            | same         | `a * 1.5`                             |
|               | `%`                 | same         | `a % 2`                               |
|               | `**`                | same         | `a ** 2`                              |
| **Unary**     | `-` (Negation)      | `-`          | `-a`                                  |
|               | *See Note 2*        | `!`          | `BuiltInFunc.logical_not(a)`          |

> **Note 1:** Because `&`, `|`, and `^` are used for logical AND/OR and other purposes, the DSL's bitwise operators are available as explicit functions in `BuiltInFunc`:
> `BuiltInFunc.bitwise_and(a, b)`, `BuiltInFunc.bitwise_or(a, b)`, `BuiltInFunc.bitwise_xor(a, b)`

> **Note 2:** Python's `not` keyword cannot be overloaded. Use `BuiltInFunc.logical_not(a)` for logical negation.

### 3.4. Functions (`BuiltInFunc`)

All built-in DSL functions are available as static methods in the `BuiltInFunc` class.

**Common Functions:**

-   `BuiltInFunc.sin(expr)`
-   `BuiltInFunc.cos(expr)`
-   `BuiltInFunc.sqrt(expr)`
-   `BuiltInFunc.abs(expr)`
-   `BuiltInFunc.round(expr)`
-   `BuiltInFunc.floor(expr)`
-   `BuiltInFunc.min(expr1, expr2)`
-   `BuiltInFunc.max(expr1, expr2)`
-   `BuiltInFunc.clamp(value, min_val, max_val)`
-   `BuiltInFunc.log(expr)`
-   ......

**Special Function: `sort`**

The `nth_N` family of functions is accessed via `BuiltInFunc.sort`. It returns a list of `DSLExpr` objects, each representing the Nth smallest value in the input list.

-   **Usage**: `BuiltInFunc.sort([expr1, expr2, ...])[N]`
-   `[0]` maps to `nth_1`, `[1]` to `nth_2`, and so on.

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

-   **Syntax**: `BuiltInFunc.if_then_else(condition, true_expression, false_expression)`

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

#### Dynamic Pixel Access (`.access()`)

To dynamically access pixels from a clip (the `access` function), use the `.access()` method on a `SourceClip` object.

-   **Syntax**: `clip.access(x_coordinate_expr, y_coordinate_expr)`

**Example:**
```python
clip_a = SourceClip('a')
# Access pixel from clip 'a' at offset (X+10, Y)
shifted_pixel = clip_a.access(Constant.X + 10, Constant.Y)
```

#### Frame Property Access (`.props`)

To access a frame property, use the `.props` property on a `SourceClip` object, followed by item access (`[...]`).

-   **Syntax**: `clip.props['PropertyName']`

**Example:**
```python
clip_a = SourceClip('a')

# Access the _Matrix property from the frame properties of clip 'a'
matrix = clip_a.props['_Matrix']

# The result can be used in other expressions
is_bt709 = matrix == 1
```
"""

import varname
from typing import Union, Sequence, Optional


def _get_preferred_name() -> Optional[str]:
    try:
        name_or_tuple = varname.core.varname(frame=3)
        if isinstance(name_or_tuple, str):
            return name_or_tuple
        return None
    except Exception:
        return None


class _DSLBuilder:
    def __init__(self):
        self.statements: list[str] = []
        self.node_to_name: dict["_Node", str] = {}
        self.name_counter: int = 0
        self.used_names: set[str] = set()

    def _get_unique_name(self, node: "_Node") -> str:
        base_name = getattr(node, "preferred_name", None)

        if base_name and base_name not in self.used_names:
            self.used_names.add(base_name)
            return f"{base_name}_0"

        if base_name:
            i = 1
            while f"{base_name}_{i}" in self.used_names:
                i += 1
            name = f"{base_name}_{i}"
            self.used_names.add(name)
            return name
        else:
            name = f"v_{self.name_counter}"
            while name in self.used_names:
                self.name_counter += 1
                name = f"v_{self.name_counter}"
            self.name_counter += 1
            self.used_names.add(name)
            return name

    def build(self, final_node: "DSLExpr") -> str:
        final_name = self._process_node(final_node._node)
        script_lines = self.statements
        script_lines.append(f"RESULT = {final_name}")
        return "\n".join(script_lines)

    def _process_node(self, node: "_Node") -> str:
        if node in self.node_to_name:
            return self.node_to_name[node]

        dsl_fragment = node.to_dsl_fragment(self)

        if isinstance(node, (_InputNode, _ConstantNode, _LiteralNode)):
            self.node_to_name[node] = dsl_fragment
            return dsl_fragment

        name = self._get_unique_name(node)
        statement = f"{name} = {dsl_fragment}"
        self.statements.append(statement)
        self.node_to_name[node] = name
        return name


class _Node:
    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def to_dsl_fragment(self, builder: _DSLBuilder) -> str:
        raise NotImplementedError


class DSLExpr:
    def __init__(self, node: "_Node"):
        self._node = node

    def __bool__(self):
        raise TypeError(
            "Cannot evaluate a DSLExpr to a boolean. DSL expressions do not have a "
            "truth value in Python at generation time.\n"
            "Use 'BuiltInFunc.if_then_else(condition, true_expr, false_expr)' "
            "for conditional logic."
        )

    def __add__(self, other):
        return DSLExpr(_BinaryOpNode("+", self, other))

    def __sub__(self, other):
        return DSLExpr(_BinaryOpNode("-", self, other))

    def __mul__(self, other):
        return DSLExpr(_BinaryOpNode("*", self, other))

    def __truediv__(self, other):
        return DSLExpr(_BinaryOpNode("/", self, other))

    def __pow__(self, other):
        return DSLExpr(_BinaryOpNode("**", self, other))

    def __mod__(self, other):
        return DSLExpr(_BinaryOpNode("%", self, other))

    def __lt__(self, other):
        return DSLExpr(_BinaryOpNode("<", self, other))

    def __le__(self, other):
        return DSLExpr(_BinaryOpNode("<=", self, other))

    def __gt__(self, other):
        return DSLExpr(_BinaryOpNode(">", self, other))

    def __ge__(self, other):
        return DSLExpr(_BinaryOpNode(">=", self, other))

    def __eq__(self, other):
        return DSLExpr(_BinaryOpNode("==", self, other))

    def __ne__(self, other):
        return DSLExpr(_BinaryOpNode("!=", self, other))

    def __and__(self, other):
        return DSLExpr(_BinaryOpNode("&&", self, other))

    def __or__(self, other):
        return DSLExpr(_BinaryOpNode("||", self, other))

    def __neg__(self):
        return DSLExpr(_UnaryOpNode("-", self))

    def __invert__(self):
        return DSLExpr(_UnaryOpNode("~", self))

    @property
    def dsl(self) -> str:
        builder = _DSLBuilder()
        return builder.build(self)


class _InputNode(_Node):
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(("input", self.name))

    def __eq__(self, other):
        return isinstance(other, _InputNode) and self.name == other.name

    def to_dsl_fragment(self, builder) -> str:
        return f"${self.name}"


class _ConstantNode(_Node):
    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(("const", self.name))

    def __eq__(self, other):
        return isinstance(other, _ConstantNode) and self.name == other.name

    def to_dsl_fragment(self, builder) -> str:
        return f"${self.name}"


class _LiteralNode(_Node):
    def __init__(self, value: Union[int, float]):
        self.value = value

    def __hash__(self):
        return hash(("literal", self.value))

    def __eq__(self, other):
        return isinstance(other, _LiteralNode) and self.value == other.value

    def to_dsl_fragment(self, builder) -> str:
        return str(self.value)


def _to_node(expr: Union[DSLExpr, int, float]) -> _Node:
    return expr._node if isinstance(expr, DSLExpr) else _LiteralNode(expr)


class _BinaryOpNode(_Node):
    def __init__(self, op: str, left: DSLExpr, right: Union[DSLExpr, int, float]):
        self.op, self.left, self.right = op, _to_node(left), _to_node(right)
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        return hash(("binary_op", self.op, self.left, self.right))

    def __eq__(self, other):
        return (
            isinstance(other, _BinaryOpNode)
            and self.op == other.op
            and self.left == other.left
            and self.right == other.right
        )

    def to_dsl_fragment(self, builder) -> str:
        return f"({builder._process_node(self.left)} {self.op} {builder._process_node(self.right)})"


class _UnaryOpNode(_Node):
    def __init__(self, op: str, operand: DSLExpr):
        self.op, self.operand = op, _to_node(operand)
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        return hash(("unary_op", self.op, self.operand))

    def __eq__(self, other):
        return (
            isinstance(other, _UnaryOpNode)
            and self.op == other.op
            and self.operand == other.operand
        )

    def to_dsl_fragment(self, builder) -> str:
        return f"{self.op}({builder._process_node(self.operand)})"


class _FunctionCallNode(_Node):
    def __init__(self, func_name: str, args: Sequence[Union[DSLExpr, int, float]]):
        self.func_name, self.args = func_name, tuple(_to_node(arg) for arg in args)
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        return hash(("func_call", self.func_name, self.args))

    def __eq__(self, other):
        return (
            isinstance(other, _FunctionCallNode)
            and self.func_name == other.func_name
            and self.args == other.args
        )

    def to_dsl_fragment(self, builder) -> str:
        arg_names = [builder._process_node(arg) for arg in self.args]
        return f"{self.func_name}({', '.join(arg_names)})"


class _TernaryOpNode(_Node):
    def __init__(self, cond: DSLExpr, true_expr: DSLExpr, false_expr: DSLExpr):
        self.cond, self.true_expr, self.false_expr = (
            _to_node(cond),
            _to_node(true_expr),
            _to_node(false_expr),
        )
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        return hash(("ternary", self.cond, self.true_expr, self.false_expr))

    def __eq__(self, other):
        return (
            isinstance(other, _TernaryOpNode)
            and self.cond == other.cond
            and self.true_expr == other.true_expr
            and self.false_expr == other.false_expr
        )

    def to_dsl_fragment(self, builder: _DSLBuilder) -> str:
        return f"({builder._process_node(self.cond)} ? {builder._process_node(self.true_expr)} : {builder._process_node(self.false_expr)})"


class _PropertyAccessNode(_Node):
    def __init__(self, clip_node: _InputNode, prop_name: str):
        self.clip_node = clip_node
        self.prop_name = prop_name
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        return hash(("prop_access", self.clip_node, self.prop_name))

    def __eq__(self, other):
        return (
            isinstance(other, _PropertyAccessNode)
            and self.clip_node == other.clip_node
            and self.prop_name == other.prop_name
        )

    def to_dsl_fragment(self, builder: _DSLBuilder) -> str:
        clip_dsl_name = builder._process_node(self.clip_node)
        return f"{clip_dsl_name}.{self.prop_name}"


class _PropertyAccessor:
    def __init__(self, clip: "SourceClip"):
        self.clip = clip

    def __getitem__(self, prop_name: str) -> "DSLExpr":
        if not isinstance(prop_name, str):
            raise TypeError("Property name must be a string.")
        if not isinstance(self.clip._node, _InputNode):
            raise TypeError("Property access is only valid on a direct source clip.")
        return DSLExpr(_PropertyAccessNode(self.clip._node, prop_name))


class SourceClip(DSLExpr):
    _ALIASES = "xyzabcdefghijklmnopqrstuvw"

    def __init__(self, identifier: Union[str, int]):
        if isinstance(identifier, str):
            if identifier in self._ALIASES:
                name = f"src{self._ALIASES.index(identifier)}"
            elif identifier.startswith("src"):
                name = identifier
            else:
                raise ValueError(f"Invalid string identifier: {identifier}")
        elif isinstance(identifier, int):
            name = f"src{identifier}"
        else:
            raise TypeError("SourceClip identifier must be a string or integer.")
        super().__init__(_InputNode(name))

    def access(self, x_expr: DSLExpr, y_expr: DSLExpr) -> "DSLExpr":
        return DSLExpr(_FunctionCallNode("dyn", [self, x_expr, y_expr]))

    @property
    def props(self) -> _PropertyAccessor:
        return _PropertyAccessor(self)


class Constant:
    pi = DSLExpr(_ConstantNode("pi"))
    N = DSLExpr(_ConstantNode("N"))
    X = DSLExpr(_ConstantNode("X"))
    Y = DSLExpr(_ConstantNode("Y"))
    width = DSLExpr(_ConstantNode("width"))
    height = DSLExpr(_ConstantNode("height"))


class BuiltInFunc:
    @staticmethod
    def sin(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("sin", [x]))

    @staticmethod
    def cos(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("cos", [x]))

    @staticmethod
    def log(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("log", [x]))

    @staticmethod
    def exp(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("exp", [x]))

    @staticmethod
    def sqrt(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("sqrt", [x]))

    @staticmethod
    def abs(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("abs", [x]))

    @staticmethod
    def min(x: DSLExpr, y: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("min", [x, y]))

    @staticmethod
    def max(x: DSLExpr, y: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("max", [x, y]))

    @staticmethod
    def clamp(x: DSLExpr, min_val: DSLExpr, max_val: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("clamp", [x, min_val, max_val]))

    @staticmethod
    def sort(items: list[DSLExpr]) -> list[DSLExpr]:
        if not isinstance(items, list) or not items:
            raise TypeError("Input for sort must be a non-empty list of expressions.")
        return [
            DSLExpr(_FunctionCallNode(f"nth_{i + 1}", items)) for i in range(len(items))
        ]

    @staticmethod
    def trunc(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("trunc", [x]))

    @staticmethod
    def round(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("round", [x]))

    @staticmethod
    def floor(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("floor", [x]))

    @staticmethod
    def if_then_else(cond: DSLExpr, true_expr: DSLExpr, false_expr: DSLExpr) -> DSLExpr:
        return DSLExpr(_TernaryOpNode(cond, true_expr, false_expr))

    @staticmethod
    def logical_not(x: DSLExpr) -> DSLExpr:
        return DSLExpr(_UnaryOpNode("!", x))

    @staticmethod
    def bitwise_or(x: DSLExpr, y: DSLExpr) -> DSLExpr:
        return DSLExpr(_BinaryOpNode("|", x, y))

    @staticmethod
    def bitwise_xor(x: DSLExpr, y: DSLExpr) -> DSLExpr:
        return DSLExpr(_BinaryOpNode("^", x, y))

    @staticmethod
    def bitwise_and(x: DSLExpr, y: DSLExpr) -> DSLExpr:
        return DSLExpr(_BinaryOpNode("&", x, y))
