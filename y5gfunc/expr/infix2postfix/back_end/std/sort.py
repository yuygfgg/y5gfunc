import regex as re

CS_TOP_SEQUENCE = ["dup1", "dup0", "max", "swap2", "min"]


def adjacent_compare_swap(i: int) -> list[str]:
    """
    Generate RPN code to compare and swap the elements at stack depth i and i+1.
    """
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


def expand_rpn_sort(expr: str) -> str:
    """
    Expand sortN instructions in an RPN string to their equivalent min/max/swap/dup sequences.
    """
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
