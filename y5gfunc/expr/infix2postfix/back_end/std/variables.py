from ...util import _UNARY_OPS, _BINARY_OPS, _TERNARY_OPS, _CLIP_OPS
from ...middle_end import tokenize_expr, is_token_numeric, token_pattern


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
                    updated_copies = [
                        c for c in copies if c["stack_pos"] < stack_size
                    ]
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