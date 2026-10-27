import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from src.definitions import PATHS
from src.solver.solve_bfs import solve_bfs


def _import_rl_lib():
    """dynamic_maze_start_goal lives in the maze_impl checkout (the saecollab
    submodule predates it) and imports its siblings as top-level packages
    (env, models, ablation, StackedCollab), so that checkout has to be on
    sys.path. Imported lazily: the BFS path never pays for torch."""
    lib = PATHS.RL_LIB
    if not os.path.isfile(os.path.join(lib, "dynamic_maze_start_goal.py")):
        raise FileNotFoundError(f"dynamic_maze_start_goal.py nao encontrado em "
                                f"{lib} (ajuste PATHS.RL_LIB)")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import dynamic_maze_start_goal as dms
    # Its default is cwd-relative; keep ours out of maze_impl's own cache (same
    # tag + size there may come from a different .maze file).
    dms.TOPOLOGY_CACHE_DIR = os.path.join(PATHS.APP, "logs", "rl_topology_cache")
    return dms


class RLSolver:
    """Drop-in for solve_bfs: same `(view, start, goal) -> [cells]` contract,
    start and goal included, [] when there is no route.

    The route is the greedy rollout of a goal-conditioned agent trained by
    maze_impl/dynamic_maze_start_goal.py, loaded the same way
    dynamic_maze_viewer.py does (load_trained_run). Same failure rules as the
    viewer too: revisiting a cell (bumping a wall included) is a loop, and
    rows * cols steps is the limit. Either way this call falls back to
    solve_bfs and sets `fell_back`, so the caller knows the route on hand is
    BFS's and can retry `rollout()` later from somewhere else (the runner does
    at every cell boundary, and goes back to RL as soon as it stops looping).

    The agent sees the LIVE walls `view` reports, not the .maze file's: they
    are re-read on every call and the topology (neighbours + the
    possible-actions feature) rebuilt when they changed, exactly like the
    viewer's obstacle editor. So a discovered obstacle is routed around the
    same way BFS does -- or not, if the policy does not generalise to it.

    `view` needs `.rows`, `.cols`, `.in_bounds(cell)` and `.is_wall(cell)`.
    """

    def __init__(self, model_path: str, maze_path: str, device: str = "cpu"):
        self._dms = _import_rl_lib()
        # A run directory means its model_last.pth, the viewer's default.
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model_last.pth")
        self.run_dir    = os.path.dirname(os.path.abspath(model_path))
        self.model_name = os.path.basename(model_path)
        self.device     = device
        self.fell_back  = False     # last __call__ returned solve_bfs's route
        self._last_fail: Optional[Tuple] = None
        self._load(maze_path)

    def _load(self, maze_path: str) -> None:
        (_, self.topo, self.env, self.agent,
         self.summary, self.model_path) = self._dms.load_trained_run(
            self.run_dir, self.model_name, maze_path, self.device)
        self._walls = ~self.topo.walkable.reshape(self.topo.rows, self.topo.cols)

        trained = self.summary.get("maze")
        if trained and trained != self.topo.tag:
            print(f"[RL][WARN] modelo treinado em '{trained}', "
                  f"rodando em '{self.topo.tag}'")
        print(f"[RL] {self.summary.get('model', 'dense')} "
              f"({self.summary.get('encoder', '?')}) -- {self.model_path}")

    def _sync_walls(self, view) -> None:
        t = self.topo
        walls = np.array([[view.is_wall((r, c)) for c in range(t.cols)]
                          for r in range(t.rows)], dtype=bool)
        if np.array_equal(walls, self._walls):
            return
        self._walls = walls
        # Same rebuild as the viewer's Obstacles._rebuild. topo.dist stays the
        # file maze's: the rollout never reads it.
        t.grid     = np.where(walls, self._dms.WALL, 0).astype(np.uint8)
        t.walkable = ~walls.reshape(-1)
        t.nav      = np.flatnonzero(t.walkable).astype(np.int32)
        t.nbr      = t._build_neighbours()
        if self.env._use_pa:
            self.env._pa = (t.nbr >= 0).astype(np.float32)

    def _fail(self, start, goal, reason: str, steps: int, verbose: bool) -> None:
        # The runner's holding states retry on a timer with the same inputs;
        # a greedy policy fails the same way every time, so say it once.
        key = (start, goal, reason, self._walls.tobytes())
        if verbose and key != self._last_fail:
            self._last_fail = key
            print(f"[RL] {reason} de {start} a {goal} apos {steps} passos "
                  f"-- usando o BFS")
        return None

    def __call__(self, view, start: Tuple[int, int],
                 goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = self.rollout(view, start, goal)
        self.fell_back = path is None
        return solve_bfs(view, start, goal) if path is None else path

    def rollout(self, view, start: Tuple[int, int], goal: Tuple[int, int],
                verbose: bool = True) -> Optional[List[Tuple[int, int]]]:
        """The agent's own route, no fallback: the cell list, [] when start or
        goal is not a navigable cell, None when the policy loops or runs out
        of steps."""
        if not view.in_bounds(start) or not view.in_bounds(goal):
            return []
        if view.is_wall(start) or view.is_wall(goal):
            return []
        if start == goal:
            return [start]

        if (view.rows, view.cols) != (self.topo.rows, self.topo.cols):
            # Maze swapped under us (main.py's `maze` command): rebuild the
            # env for it. An encoder sized to the old maze refuses here.
            try:
                self._load(view.env.file_path)
            except Exception as e:                  # noqa: BLE001
                print(f"[RL][ERROR] modelo nao serve para "
                      f"{view.rows}x{view.cols}: {e} -- usando o BFS")
                return None
        self._sync_walls(view)

        t = self.topo
        goal_idx  = t.idx(*goal)
        state     = self.env.reset(t.idx(*start), goal_idx)
        path      = [tuple(start)]
        visited   = {t.idx(*start)}
        max_steps = t.rows * t.cols
        for steps in range(1, max_steps + 1):
            state, _, done, info = self.env.step(self.agent.act(state, eval=True))
            nxt = int(info["next_idx"])
            if done:
                path.append(tuple(goal))
                return path
            if nxt in visited:
                return self._fail(start, goal, "LOOP", steps, verbose)
            visited.add(nxt)
            path.append(t.rc(nxt))
        return self._fail(start, goal, "LIMITE", max_steps, verbose)
