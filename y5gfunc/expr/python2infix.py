"""
Python interface to generate infix DSL code, that can be further converted to vapoursynth Expr RPN code.

Refer to [this document](docs/python_interface.md) for documentations about the Python interface.
"""

import contextlib
import varname
from typing import Union, Sequence, Optional, TypeAlias


_USE_VARNAME = False


@contextlib.contextmanager
def varname_toggle(enable: bool):
    """
    A context manager to temporarily enable or disable the use of the `varname` package.
    Disabling this can significantly speed up the generation of complex DSL expressions.
    It is disabled by default for performance reasons.

    Usage:
        with varname_toggle(True):
            # code where varname is enabled
            ...
        # varname is disabled again here
    """
    global _USE_VARNAME
    old_value = _USE_VARNAME
    _USE_VARNAME = enable
    try:
        yield
    finally:
        _USE_VARNAME = old_value


ExprLike: TypeAlias = Union["DSLExpr", int, float]


def _get_preferred_name() -> Optional[str]:
    if not _USE_VARNAME:
        return None
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
    def __init__(self):
        self._hash_cache: Optional[int] = None

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

    def __add__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("+", self, other))

    def __sub__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("-", self, other))

    def __mul__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("*", self, other))

    def __truediv__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("/", self, other))

    def __pow__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("**", self, other))

    def __mod__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("%", self, other))

    def __lt__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("<", self, other))

    def __le__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("<=", self, other))

    def __gt__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode(">", self, other))

    def __ge__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode(">=", self, other))

    def __eq__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("==", self, other))

    def __ne__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("!=", self, other))

    def __and__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("&&", self, other))

    def __or__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("||", self, other))

    def __radd__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("+", other, self))

    def __rsub__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("-", other, self))

    def __rmul__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("*", other, self))

    def __rtruediv__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("/", other, self))

    def __rpow__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("**", other, self))

    def __rmod__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("%", other, self))

    def __rand__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("&&", other, self))

    def __ror__(self, other: ExprLike):
        return DSLExpr(_BinaryOpNode("||", other, self))

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
        super().__init__()
        self.name = name

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("input", self.name))
        return self._hash_cache

    def __eq__(self, other):
        return isinstance(other, _InputNode) and self.name == other.name

    def to_dsl_fragment(self, builder) -> str:
        return f"${self.name}"


class _ConstantNode(_Node):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("const", self.name))
        return self._hash_cache

    def __eq__(self, other):
        return isinstance(other, _ConstantNode) and self.name == other.name

    def to_dsl_fragment(self, builder) -> str:
        return f"${self.name}"


class _LiteralNode(_Node):
    def __init__(self, value: Union[int, float]):
        super().__init__()
        self.value = value

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("literal", self.value))
        return self._hash_cache

    def __eq__(self, other):
        return isinstance(other, _LiteralNode) and self.value == other.value

    def to_dsl_fragment(self, builder) -> str:
        return str(self.value)


def _to_node(expr: ExprLike) -> _Node:
    return expr._node if isinstance(expr, DSLExpr) else _LiteralNode(expr)


class _BinaryOpNode(_Node):
    def __init__(
        self,
        op: str,
        left: ExprLike,
        right: ExprLike,
    ):
        super().__init__()
        self.op, self.left, self.right = op, _to_node(left), _to_node(right)
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("binary_op", self.op, self.left, self.right))
        return self._hash_cache

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
    def __init__(self, op: str, operand: ExprLike):
        super().__init__()
        self.op, self.operand = op, _to_node(operand)
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("unary_op", self.op, self.operand))
        return self._hash_cache

    def __eq__(self, other):
        return (
            isinstance(other, _UnaryOpNode)
            and self.op == other.op
            and self.operand == other.operand
        )

    def to_dsl_fragment(self, builder) -> str:
        return f"{self.op}({builder._process_node(self.operand)})"


class _FunctionCallNode(_Node):
    def __init__(self, func_name: str, args: Sequence[ExprLike]):
        super().__init__()
        self.func_name, self.args = func_name, tuple(_to_node(arg) for arg in args)
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("func_call", self.func_name, self.args))
        return self._hash_cache

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
    def __init__(
        self,
        cond: ExprLike,
        true_expr: ExprLike,
        false_expr: ExprLike,
    ):
        super().__init__()
        self.cond, self.true_expr, self.false_expr = (
            _to_node(cond),
            _to_node(true_expr),
            _to_node(false_expr),
        )
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("ternary", self.cond, self.true_expr, self.false_expr))
        return self._hash_cache

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
        super().__init__()
        self.clip_node = clip_node
        self.prop_name = prop_name
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(("prop_access", self.clip_node, self.prop_name))
        return self._hash_cache

    def __eq__(self, other):
        return (
            isinstance(other, _PropertyAccessNode)
            and self.clip_node == other.clip_node
            and self.prop_name == other.prop_name
        )

    def to_dsl_fragment(self, builder: _DSLBuilder) -> str:
        clip_dsl_name = builder._process_node(self.clip_node)
        return f"{clip_dsl_name}.{self.prop_name}"


class _PixelAccessNode(_Node):
    def __init__(self, clip_node: _InputNode, x_node: _Node, y_node: _Node):
        super().__init__()
        self.clip_node = clip_node
        self.x_node = x_node
        self.y_node = y_node
        self.preferred_name = _get_preferred_name()

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        self._hash_cache = hash(
            ("pixel_access", self.clip_node, self.x_node, self.y_node)
        )
        return self._hash_cache

    def __eq__(self, other):
        return (
            isinstance(other, _PixelAccessNode)
            and self.clip_node == other.clip_node
            and self.x_node == other.x_node
            and self.y_node == other.y_node
        )

    def to_dsl_fragment(self, builder: _DSLBuilder) -> str:
        clip_dsl_name = builder._process_node(self.clip_node)
        x_dsl_name = builder._process_node(self.x_node)
        y_dsl_name = builder._process_node(self.y_node)
        return f"{clip_dsl_name}[{x_dsl_name},{y_dsl_name}]"


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

    def __getitem__(self, key):
        if not isinstance(self._node, _InputNode):
            raise TypeError("Pixel access is only valid on a direct source clip.")
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "Pixel access requires a tuple of two indices, e.g., a[x, y]."
            )

        x, y = key
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("Pixel access indices must be integers, not expressions.")

        x_node = _to_node(x)
        y_node = _to_node(y)

        return DSLExpr(_PixelAccessNode(self._node, x_node, y_node))

    def access(
        self,
        x_expr: ExprLike,
        y_expr: ExprLike,
    ) -> "DSLExpr":
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
    def sin(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("sin", [x]))

    @staticmethod
    def cos(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("cos", [x]))

    @staticmethod
    def log(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("log", [x]))

    @staticmethod
    def exp(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("exp", [x]))

    @staticmethod
    def sqrt(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("sqrt", [x]))

    @staticmethod
    def abs(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("abs", [x]))

    @staticmethod
    def min(x: ExprLike, y: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("min", [x, y]))

    @staticmethod
    def max(x: ExprLike, y: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("max", [x, y]))

    @staticmethod
    def clamp(
        x: ExprLike,
        min_val: ExprLike,
        max_val: ExprLike,
    ) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("clamp", [x, min_val, max_val]))

    @staticmethod
    def sort(items: list[ExprLike]) -> list[DSLExpr]:
        if not isinstance(items, list) or not items:
            raise TypeError("Input for sort must be a non-empty list of expressions.")
        return [
            DSLExpr(_FunctionCallNode(f"nth_{i + 1}", items)) for i in range(len(items))
        ]

    @staticmethod
    def trunc(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("trunc", [x]))

    @staticmethod
    def round(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("round", [x]))

    @staticmethod
    def floor(x: ExprLike) -> DSLExpr:
        return DSLExpr(_FunctionCallNode("floor", [x]))

    @staticmethod
    def if_then_else(
        cond: ExprLike,
        true_expr: ExprLike,
        false_expr: ExprLike,
    ) -> DSLExpr:
        return DSLExpr(_TernaryOpNode(cond, true_expr, false_expr))

    @staticmethod
    def logical_not(x: ExprLike) -> DSLExpr:
        return DSLExpr(_UnaryOpNode("!", x))

    @staticmethod
    def bitwise_or(x: ExprLike, y: ExprLike) -> DSLExpr:
        return DSLExpr(_BinaryOpNode("|", x, y))

    @staticmethod
    def bitwise_xor(x: ExprLike, y: ExprLike) -> DSLExpr:
        return DSLExpr(_BinaryOpNode("^", x, y))

    @staticmethod
    def bitwise_and(x: ExprLike, y: ExprLike) -> DSLExpr:
        return DSLExpr(_BinaryOpNode("&", x, y))
