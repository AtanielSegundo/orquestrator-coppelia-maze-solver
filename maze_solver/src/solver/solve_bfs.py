from collections import deque
from typing import Dict, List, Optional, Tuple

from src.solver.constants import DIRECTIONS


def solve_bfs(view, start: Tuple[int, int],
             goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Caminho minimo em celulas navegaveis, start e goal inclusos.

    Ground truth por BFS: ainda nao ha um agente treinado plugado aqui, entao
    o caminho otimo e a propria solucao. Trocar por uma policy e so trocar o
    corpo desta funcao, o resto do orquestrador so consome a lista de celulas.

    `view` needs only `.in_bounds(cell)` and `.is_wall(cell)` -- both
    MazeView (raylib) and any lightweight grid wrapper satisfy that.
    """
    if not view.in_bounds(start) or not view.in_bounds(goal):
        return []
    if view.is_wall(start) or view.is_wall(goal):
        return []
    if start == goal:
        return [start]

    previous: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    frontier = deque([start])

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for dr, dc, _ in DIRECTIONS:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in previous or not view.in_bounds(nxt) or view.is_wall(nxt):
                continue
            previous[nxt] = current
            frontier.append(nxt)

    if goal not in previous:
        return []

    path, cursor = [], goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path
