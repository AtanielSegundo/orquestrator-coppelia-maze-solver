import os


def get_maze_tag(path:str) -> str:
    return os.path.basename(path).split(".")[0]
