from ....utils import (
    get_stack_effect,
    tokenize_expr,
    _CLAMP_OPS,
    _REL_STATIC_PATTERN_POSTFIX,
    get_used_variable_names,
)
import regex as re
import random
import time


def convert_var(expr: str) -> str:
    """Rewrite variable store/load (`x!`, `x@`) to stack ops (`dup`, `swap`, `drop`)."""
    tokens = tokenize_expr(expr)
    if not tokens:
        return ""

    # Find all variable names used
    var_names = get_used_variable_names(tokens)

    if not var_names:
        return " ".join(tokens)

    var_list = sorted(var_names)
    var_count = len(var_list)

    var_to_pos = {var: i for i, var in enumerate(var_list)}

    new_tokens = []

    for _ in range(var_count):
        new_tokens.append("0")

    stack_size = var_count

    for token in tokens:
        if token.endswith("!") and len(token) > 1:
            if not token.startswith("[") and not _REL_STATIC_PATTERN_POSTFIX.match(
                token
            ):
                var_name = token[:-1]
                if var_name in var_to_pos:
                    pos = var_to_pos[var_name]
                    swap_index = stack_size - 1 - pos
                    if swap_index == 1:
                        new_tokens.append("swap")
                    else:
                        new_tokens.append(f"swap{swap_index}")
                    new_tokens.append("drop")
                    stack_size -= 1
                else:
                    new_tokens.append(token)
                    stack_size += get_stack_effect(token)
            else:
                # Not a variable store, keep as-is
                new_tokens.append(token)
                stack_size += get_stack_effect(token)
        elif token.endswith("@") and len(token) > 1:
            if not token.startswith("[") and not _REL_STATIC_PATTERN_POSTFIX.match(
                token
            ):
                var_name = token[:-1]
                if var_name in var_to_pos:
                    pos = var_to_pos[var_name]
                    dup_index = stack_size - 1 - pos
                    if dup_index == 0:
                        new_tokens.append("dup")
                    else:
                        new_tokens.append(f"dup{dup_index}")
                    stack_size += 1  # dup increases stack size
                else:
                    new_tokens.append(token)
                    stack_size += get_stack_effect(token)
            else:
                new_tokens.append(token)
                stack_size += get_stack_effect(token)
        else:
            new_tokens.append(token)
            stack_size += get_stack_effect(token)

    for _ in range(var_count):
        new_tokens.append("swap")
        new_tokens.append("drop")
        stack_size -= 1

    return " ".join(new_tokens)


def convert_sort(expr: str) -> str:
    """
    Expand sortN instructions in an RPN string to their equivalent min/max/swap/dup/varload(@)/varstore(!) sequences.
    """

    def generate_unique_temp_prefix(used_vars: set[str]) -> str:
        """Generate a unique temporary variable prefix that doesn't conflict with existing variables."""
        # Use timestamp and random number to create a unique suffix
        timestamp = str(int(time.time() * 1000000))[
            -6:
        ]  # Last 6 digits of microsecond timestamp
        random_part = str(random.randint(1000, 9999))

        base_prefix = f"__sort_tmp_{timestamp}_{random_part}"

        # Ensure the prefix doesn't conflict with any existing variable
        counter = 0
        while any(var.startswith(base_prefix) for var in used_vars):
            counter += 1
            base_prefix = f"__sort_tmp_{timestamp}_{random_part}_{counter}"

        return base_prefix

    def generate_sort_network(n: int, temp_prefix: str) -> list[str]:
        """
        Generate Batcher odd-even mergesort network for top n elements.
        """
        if n <= 1:
            return []

        ops = []

        # Find the next power of 2 for padding
        padded_n = 1
        while padded_n < n:
            padded_n *= 2

        for i in range(n):
            ops.append(f"{temp_prefix}_{n-1-i}!")

        for i in range(n, padded_n):
            ops.extend(["1e30", f"{temp_prefix}_{i}!"])

        def compare_and_swap(i: int, j: int):
            """Generate operations to compare and swap elements at positions i and j"""
            ops.extend(
                [
                    f"{temp_prefix}_{i}@",
                    f"{temp_prefix}_{j}@",
                    f"{temp_prefix}_{j}@",
                    f"{temp_prefix}_{i}@",
                    "min",
                    f"{temp_prefix}_{i}!",
                    "max",
                    f"{temp_prefix}_{j}!",
                ]
            )

        def odd_even_merge_sort(lo: int, length: int):
            """Batcher odd-even mergesort for elements from lo to lo+length-1"""
            if length <= 1:
                return

            mid = length // 2
            # Sort first and second half
            odd_even_merge_sort(lo, mid)
            odd_even_merge_sort(lo + mid, mid)
            # Merge the sorted halves
            odd_even_merge(lo, length, 1)

        def odd_even_merge(lo: int, length: int, r: int):
            """Odd-even merge: merge two sorted sequences into one"""
            step = r * 2
            if step < length:
                odd_even_merge(lo, length, step)  # even positions
                odd_even_merge(lo + r, length, step)  # odd positions

                # Compare and swap elements at distance r
                i = lo + r
                while i + r < lo + length:
                    compare_and_swap(i, i + r)
                    i += step
            else:
                # compare adjacent elements
                if lo + r < padded_n:
                    compare_and_swap(lo, lo + r)

        odd_even_merge_sort(0, padded_n)

        # Push back to stack in sorted order, with min at top (only original n elements)
        for i in range(n - 1, -1, -1):
            ops.append(f"{temp_prefix}_{i}@")

        return ops

    tokens = tokenize_expr(expr)
    if not tokens:
        return ""

    used_vars = get_used_variable_names(tokens)

    temp_prefix = generate_unique_temp_prefix(used_vars)

    new_tokens = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token.startswith("sort"):
            # parse sortN
            n_str = token[4:]
            if not n_str:
                n = 1
            else:
                try:
                    n = int(n_str)
                except ValueError:
                    # not a valid sortN, treat as normal token
                    new_tokens.append(token)
                    i += 1
                    continue

            if n <= 0:
                # invalid sort count
                new_tokens.append(token)
                i += 1
                continue

            # generate sort network operations
            sort_ops = generate_sort_network(n, temp_prefix)
            new_tokens.extend(sort_ops)
        else:
            new_tokens.append(token)

        i += 1

    return " ".join(new_tokens)


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
