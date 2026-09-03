import os
from typing import Optional

from src.definitions import PATHS


def resolve_maze(name: str) -> Optional[str]:
    """Aceita caminho absoluto, caminho relativo ao cwd ou so o nome do maze."""
    for path in (name,
                os.path.join(PATHS.MAZES, name),
                os.path.join(PATHS.MAZES, f"{name}.maze")):
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None
