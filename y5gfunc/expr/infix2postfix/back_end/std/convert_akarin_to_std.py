from ....utils import (
    get_stack_effect,
    tokenize_expr,
    _REL_STATIC_PATTERN_INFIX,
    _CLAMP_OPS,
)
import regex as re
from itertools import groupby


def convert_var(expr: str) -> str:
    """Rewrite variable store/load (`x!`, `x@`) to stack ops (`dup`, `swap`, `drop`)."""
    tokens = tokenize_expr(expr)

    if not tokens:
        return ""
    
    # Expand dropN to N * drop
    new_tokens = []
    for token in tokens:
        if token.startswith("drop") and not token.endswith("!") and not token.startswith("@"):
            n_str = token[4:]
            if not n_str:
                new_tokens.append("drop")
            else:
                new_tokens.extend(["drop"] * int(n_str))
        else:
            new_tokens.append(token)

    tokens = new_tokens

    # ---------- static analysis ----------
    var_defs: dict[str, list[dict]] = {}
    var_map: dict[int, dict] = {}

    for idx, tk in enumerate(tokens):
        is_store = (
            tk.endswith("!")
            and len(tk) > 1
            and not tk.startswith("[")
            and not _REL_STATIC_PATTERN_INFIX.match(tk)
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
            and not _REL_STATIC_PATTERN_INFIX.match(tk)
        )
        if is_load:
            name = tk[:-1]
            if name in var_defs:
                for v_def in reversed(var_defs[name]):
                    if v_def["store_idx"] < idx < v_def["scope_end"]:
                        v_def["load_indices"].append(idx)
                        var_map[idx] = v_def
                        break

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
            and not _REL_STATIC_PATTERN_INFIX.match(tk)
        )
        is_load = (
            tk.endswith("@")
            and len(tk) > 1
            and not tk.startswith("[")
            and not _REL_STATIC_PATTERN_INFIX.match(tk)
        )

        if is_store:
            v_def = var_map.get(idx)
            if not v_def:
                continue
            k = len(v_def["load_indices"])
            if k == 0:
                stack_size -= 1  # pop stored value
                new_tokens.append("drop")
                continue

            if k > 1:
                new_tokens.extend(["dup"] * (k - 1))

            stack_size += k - 1

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

            # --- Parse stack ops ---
            is_drop_op, drop_count = False, 0
            if tk.startswith("drop"):
                n_str = tk[4:]
                if not n_str:
                    drop_count = 1
                else:
                    try:
                        drop_count = int(n_str)
                    except ValueError:
                        pass
                if drop_count > 0:
                    is_drop_op = True

            is_swap_op, swap_depth = False, 0
            if tk.startswith("swap"):
                n_str = tk[4:]
                if not n_str:
                    swap_depth = 1
                else:
                    try:
                        swap_depth = int(n_str)
                    except ValueError:
                        pass
                if swap_depth > 0:
                    is_swap_op = True

            # Protected Drop
            is_protected_drop = False
            if is_drop_op:
                top_pos = old_stack_size - 1
                if any(
                    c["stack_pos"] == top_pos
                    for copies in loads_to_serve.values()
                    for c in copies
                ):
                    is_protected_drop = True

            if is_drop_op and is_protected_drop and (old_stack_size - 1) >= drop_count:
                new_tokens.extend([f"swap{drop_count}", f"drop{drop_count}"])
                stack_size = old_stack_size - drop_count

                top_pos = old_stack_size - 1
                all_tracked = [
                    c for copies in loads_to_serve.values() for c in copies
                ] + abandoned
                for var in all_tracked:
                    if var["stack_pos"] == top_pos:
                        var["stack_pos"] -= drop_count

            # Swap
            elif is_swap_op:
                new_tokens.append(tk)
                # stack_size is unchanged by swap
                if old_stack_size > swap_depth:
                    pos1 = old_stack_size - 1
                    pos2 = old_stack_size - 1 - swap_depth

                    all_tracked = [
                        c for copies in loads_to_serve.values() for c in copies
                    ] + abandoned
                    var1, var2 = None, None
                    for v in all_tracked:
                        if v["stack_pos"] == pos1:
                            var1 = v
                        elif v["stack_pos"] == pos2:
                            var2 = v
                    if var1:
                        var1["stack_pos"] = pos2
                    if var2:
                        var2["stack_pos"] = pos1

            # Normal Op (including unprotected drops)
            else:
                new_tokens.append(tk)
                stack_size += get_stack_effect(tk)

            # --- Cleanup for stack shrinking ---
            if stack_size < old_stack_size:
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
            if depth > 1:
                new_tokens.append(f"swap{depth-1}")

    real_vals = stack_size - len(abandoned)
    if real_vals > 1:
        new_tokens.extend(["drop"] * (real_vals - 1))
    
    # Merge consecutive drops into a single dropN
    merged_tokens = []
    for key, group in groupby(new_tokens):
        if key == "drop":
            count = len(list(group))
            if count > 1:
                merged_tokens.append(f"drop{count}")
            elif count == 1:
                merged_tokens.append("drop")
        else:
            merged_tokens.extend(list(group))
    new_tokens = merged_tokens

    return " ".join(new_tokens)


def convert_sort(expr: str) -> str:
    """
    Expand sortN instructions in an RPN string to their equivalent min/max/swap/dup sequences.
    """

    def adjacent_compare_swap(i: int) -> list[str]:
        """
        Generate RPN code to compare and swap the elements at stack depth i and i+1.
        """
        CS_TOP_SEQUENCE = ["dup1", "dup1", "max", "swap2", "min"]

        if i < 0:
            raise ValueError("Stack depth must be non-negative.")

        if i == 0:
            return CS_TOP_SEQUENCE

        commands = []
        # Bring s_i and s_{i+1} to top
        commands.append(f"swap{i+1}")
        commands.append(f"swap{i}")

        commands.extend(CS_TOP_SEQUENCE)

        # Put them back
        commands.append(f"swap{i}")
        commands.append(f"swap{i+1}")

        return commands

    def generate_sort_rpn(n: int) -> list[str]:
        """
        Generate a complete RPN instruction sequence for sortN.
        """

        if n <= 1:
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


def convert_drop(expr: str) -> str:
    """
    Replaces all 'drop' and 'dropN' operations in an expression string with their std.Expr emulated equivalents.
    """

    tokens = tokenize_expr(expr)
    new_tokens = []
    stack_size = 0
    pending_plus_count = 0

    for token in tokens:
        if token.startswith("drop"):
            n_str = token[4:]
            n = 1
            if n_str:
                try:
                    n = int(n_str)
                except ValueError:
                    # Not a valid dropN, treat as normal token.
                    new_tokens.append(token)
                    stack_size += get_stack_effect(token)
                    continue

            for _ in range(n):
                if stack_size >= 2:
                    new_tokens.extend(["0", "*", "+"])
                    stack_size -= 1
                elif stack_size == 1:
                    new_tokens.extend(["0", "*"])
                    # stack: [v1] -> [v1, 0] -> [0]. size remains 1.
                    pending_plus_count += 1
        else:
            new_tokens.append(token)
            stack_size += get_stack_effect(token)

        while pending_plus_count > 0 and stack_size >= 2:
            new_tokens.append("+")
            stack_size -= 1
            pending_plus_count -= 1

    return " ".join(new_tokens)


def convert_clip_names(expr: str) -> str:
    """
    Replace clip names with their std.Expr equivalents.
    """

    pattern = re.compile(r"\bsrc(\d+)\b")
    clip_vars = "xyzabcdefghijklmnopqrstuvw"

    def get_replacement(match) -> str:
        n = int(match.group(1))
        if 0 <= n < len(clip_vars):
            return clip_vars[n]
        raise ValueError(
            f"Clip index {n} is out of range. std.Expr supports up to {len(clip_vars)} clips (src0 ~ src{len(clip_vars)-1})."
        )

    return pattern.sub(get_replacement, expr)


def convert_pow(expr: str) -> str:
    """
    Replace ** with pow.
    """
    return " ".join("pow" if token == "**" else token for token in tokenize_expr(expr))


def convert_clip_clamp(expr: str) -> str:
    """
    Replace clip and clamp operators with their std.Expr equivalents.
    """

    tokens = tokenize_expr(expr)
    new_tokens = []

    for token in tokens:
        if token in _CLAMP_OPS:
            new_tokens.extend(["swap2", "swap", "max", "min"])
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def to_std_expr(expr: str) -> str:
    """
    Convert an akarin.Expr expression to a std.Expr expression.
    """
    # FIXME: convert math functions (trunc / round / floor / fmod) (possible?)
    ret = convert_drop(
        convert_clip_names(
            convert_var(convert_sort(convert_pow(convert_clip_clamp(expr))))
        )
    )

    return ret
