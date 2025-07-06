from ...util import _UNARY_OPS, _BINARY_OPS, _TERNARY_OPS, _CLIP_OPS
from ...middle_end import tokenize_expr, is_token_numeric, token_pattern
import regex as re


def convert_var_expr(expr: str) -> str:
    """Rewrite variable store/load (`x!`, `x@`) to stack ops (`dup`, `swap`, `drop`)."""
    tokens = tokenize_expr(expr)
    if not tokens:
        return ""

    # ---------- static analysis ----------
    var_defs: dict[str, list[dict]] = {}
    var_map: dict[int, dict] = {}

    for idx, tk in enumerate(tokens):
        is_store = (
            tk.endswith("!")
            and len(tk) > 1
            and not tk.startswith("[")
            and not token_pattern.match(tk)
        )
        if is_store:
            name = tk[:-1]
            if name not in var_defs:
                var_defs[name] = []
            if var_defs[name]:
                var_defs[name][-1]["scope_end"] = idx
            v_def = {
                "name": f"{name}_{len(var_defs[name])}",
                "store_idx": idx,
                "load_indices": [],
                "scope_end": len(tokens),
            }
            var_defs[name].append(v_def)
            var_map[idx] = v_def

    for idx, tk in enumerate(tokens):
        is_load = (
            tk.endswith("@")
            and len(tk) > 1
            and not tk.startswith("[")
            and not token_pattern.match(tk)
        )
        if is_load:
            name = tk[:-1]
            if name in var_defs:
                for v_def in reversed(var_defs[name]):
                    if v_def["store_idx"] < idx < v_def["scope_end"]:
                        v_def["load_indices"].append(idx)
                        var_map[idx] = v_def
                        break

    def get_stack_effect(tk: str) -> int:
        """Return net stack delta for a token."""
        if (
            is_token_numeric(tk)
            or token_pattern.match(tk)
            or tk in ("X", "Y", "N", "width", "height")
        ):
            return 1
        if tk in _UNARY_OPS or tk.startswith(("swap", "sort")):
            return 0
        if tk in _BINARY_OPS:
            return -1
        if tk in _TERNARY_OPS or tk in _CLIP_OPS:
            return -2
        if tk.endswith("[]") and len(tk) > 2 and not token_pattern.match(tk):
            return -1
        if tk.startswith("dup"):
            return 1
        if tk.startswith("drop"):
            n = 1
            if len(tk) > 4:
                try:
                    n = int(tk[4:])
                except ValueError:
                    pass
            return -n
        return 0

    # ---------- transformation ----------
    new_tokens: list[str] = []
    stack_size = 0
    loads_to_serve: dict[str, list[dict]] = {}
    abandoned: list[dict] = []

    for idx, tk in enumerate(tokens):
        is_store = (
            tk.endswith("!")
            and len(tk) > 1
            and not tk.startswith("[")
            and not token_pattern.match(tk)
        )
        is_load = (
            tk.endswith("@")
            and len(tk) > 1
            and not tk.startswith("[")
            and not token_pattern.match(tk)
        )

        if is_store:
            v_def = var_map.get(idx)
            if not v_def:
                continue
            k = len(v_def["load_indices"])
            stack_size -= 1  # pop stored value
            if k == 0:
                new_tokens.append("drop")
            else:
                if k > 1:
                    new_tokens.extend(["dup"] * (k - 1))
                stack_size += k
                base = stack_size - k
                copies = [
                    {"def_name": v_def["name"], "stack_pos": base + j} for j in range(k)
                ]
                abandoned.extend(copies)
                loads_to_serve[v_def["name"]] = sorted(
                    copies, key=lambda c: c["stack_pos"], reverse=True
                )
        elif is_load:
            v_def = var_map.get(idx)
            if not v_def:
                new_tokens.append(tk)
                stack_size += get_stack_effect(tk)
                continue
            copies = loads_to_serve.get(v_def["name"])
            if not copies:
                # The value was dropped, so this load is a no-op.
                continue
            copy_use = copies.pop(0)
            depth = (stack_size - 1) - copy_use["stack_pos"]
            if depth < 0:
                raise ValueError(
                    f"Invalid expression: var load depth is negative ({depth}) for {v_def['name']}"
                )
            new_tokens.append("dup" if depth == 0 else f"dup{depth}")
            stack_size += 1
        else:
            old_stack_size = stack_size
            new_tokens.append(tk)
            stack_size += get_stack_effect(tk)
            if stack_size < old_stack_size:
                # stack has shrunk, remove dropped items
                for v_name, copies in list(loads_to_serve.items()):
                    updated_copies = [c for c in copies if c["stack_pos"] < stack_size]
                    if not updated_copies:
                        del loads_to_serve[v_name]
                    else:
                        loads_to_serve[v_name] = updated_copies
                abandoned = [c for c in abandoned if c["stack_pos"] < stack_size]

    # ---------- cleanup ----------
    if abandoned:
        abandoned.sort(key=lambda v: v["stack_pos"], reverse=True)
        final_size = stack_size
        for i, var in enumerate(abandoned):
            depth = (final_size - 1 - var["stack_pos"]) - i
            if depth > 0:
                new_tokens.append(f"swap{depth}")
            new_tokens.append("drop")
            if depth > 0:
                new_tokens.append(f"swap{depth-1}")

    real_vals = stack_size - len(abandoned)
    if real_vals > 1:
        new_tokens.extend(["drop"] * (real_vals - 1))

    return " ".join(new_tokens)


def expand_rpn_sort(expr: str) -> str:
    """
    Expand sortN instructions in an RPN string to their equivalent min/max/swap/dup sequences.
    """

    def adjacent_compare_swap(i: int) -> list[str]:
        """
        Generate RPN code to compare and swap the elements at stack depth i and i+1.
        """
        CS_TOP_SEQUENCE = ["dup1", "dup0", "max", "swap2", "min"]

        if i < 0:
            raise ValueError("Stack depth must be non-negative.")

        commands = []

        for k in range(i, 0, -1):
            commands.append(f"swap{k}")

        commands.extend(CS_TOP_SEQUENCE)

        for k in range(1, i + 1):
            commands.append(f"swap{k}")

        return commands

    def generate_sort_rpn(n: int) -> list[str]:
        """
        Generate a complete RPN instruction sequence for sortN.
        """
        if n < 1:
            return []
        if n == 1:
            return []

        if n == 2:
            return adjacent_compare_swap(0)

        total_commands = []
        for phase in range(n):
            if phase % 2 == 0:
                for i in range(0, n - 1, 2):
                    total_commands.extend(adjacent_compare_swap(i))
            else:
                for i in range(1, n - 1, 2):
                    total_commands.extend(adjacent_compare_swap(i))

        return total_commands

    sort_pattern = re.compile(r"\bsort(\d+)\b")

    tokens = expr.split()

    final_rpn = []
    for token in tokens:
        match = sort_pattern.match(token)
        if match:
            n = int(match.group(1))
            sort_commands = generate_sort_rpn(n)
            final_rpn.extend(sort_commands)
        else:
            final_rpn.append(token)

    return " ".join(final_rpn)


def replace_drop_in_expr(expr: str) -> str:
    """
    Replaces all 'drop' and 'dropN' operations in an expression string with their std.Expr emulated equivalents.
    """

    def emulate_drop(n: int) -> str:
        """
        Emulate the 'dropN' operation from akarin.Expr using std.Expr operators.
        """
        if n < 0:
            raise ValueError("Drop count cannot be negative.")
        if n == 0:
            return ""

        drop_one = "0 * +"

        return " ".join([drop_one] * n)

    tokens = expr.split()
    new_tokens = []
    for token in tokens:
        if token.startswith("drop"):
            n_str = token[4:]
            n = 1
            if n_str:
                try:
                    n = int(n_str)
                except ValueError:
                    new_tokens.append(token)
                    continue

            replacement = emulate_drop(n)
            if replacement:
                new_tokens.append(replacement)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)

def replace_clip_names(expr: str) -> str:
    """
    Replace clip names with their std.Expr equivalents.
    """

    pattern = re.compile(r'\bsrc(\d+)\b')
    clip_vars = "xyzabcdefghijklmnopqrstuvw"

    def get_replacement(match) -> str:
        n = int(match.group(1))
        if 0 <= n < len(clip_vars):
            return clip_vars[n]
        raise ValueError(f"Clip index {n} is out of range. std.Expr supports up to {len(clip_vars)} clips (src0 ~ src{len(clip_vars)-1}).")

    return pattern.sub(get_replacement, expr)