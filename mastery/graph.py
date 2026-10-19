"""Карта понятий как ориентированный граф: requires[x] - понятия, без которых x не понять."""

Requires = dict[int, set[int]]


def drop_cycles(nodes: list[int], requires: Requires) -> Requires:
    """LLM иногда пишет "A нужно для B" и "B нужно для A". Обратные рёбра выкидываем обходом в глубину."""
    clean: Requires = {node: set() for node in nodes}
    state: dict[int, int] = {}  # 1 - в стеке обхода, 2 - пройден

    def visit(node: int):
        state[node] = 1
        for parent in sorted(requires.get(node, ())):
            if parent not in clean or parent == node:
                continue
            if state.get(parent) == 1:
                continue
            clean[node].add(parent)
            if parent not in state:
                visit(parent)
        state[node] = 2

    for node in nodes:
        if node not in state:
            visit(node)
    return clean


def depths(nodes: list[int], requires: Requires) -> dict[int, int]:
    """Глубина - длина самой длинной цепочки предпосылок. У базовых понятий 0."""
    memo: dict[int, int] = {}

    def depth(node: int) -> int:
        if node not in memo:
            memo[node] = 1 + max((depth(parent) for parent in requires.get(node, ())), default=-1)
        return memo[node]

    return {node: depth(node) for node in nodes}


def reverse(requires: Requires) -> Requires:
    unlocks: Requires = {node: set() for node in requires}
    for node, parents in requires.items():
        for parent in parents:
            unlocks.setdefault(parent, set()).add(node)
    return unlocks


def closure(node: int, edges: Requires) -> set[int]:
    seen: set[int] = set()
    stack = list(edges.get(node, ()))
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(edges.get(current, ()))
    return seen


def ancestors(node: int, requires: Requires) -> set[int]:
    return closure(node, requires)


def descendants(node: int, requires: Requires) -> set[int]:
    return closure(node, reverse(requires))


def weakest_chain(node: int, requires: Requires, p: dict[int, float], solid: float) -> list[int]:
    """Путь от понятия вниз по слабым предпосылкам, пока не упрёмся в твёрдо освоенное.
    Последний элемент - настоящая причина, почему понятие не даётся."""
    chain = [node]
    current = node
    while requires.get(current):
        weakest = min(requires[current], key=lambda parent: p.get(parent, 0))
        if p.get(weakest, 0) >= solid or weakest in chain:
            break
        chain.append(weakest)
        current = weakest
    return chain


def reduce_transitive(requires: Requires) -> Requires:
    """Если C <- B <- A, прямое ребро C <- A карту только запутывает. Граф должен быть уже без циклов."""
    return {node: {p for p in parents if not any(p in closure(q, requires) for q in parents if q != p)}
            for node, parents in requires.items()}
