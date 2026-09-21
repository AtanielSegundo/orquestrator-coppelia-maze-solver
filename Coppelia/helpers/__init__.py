"""
CoppeliaSim scene-building helpers: floor/maze construction and grid<->world
placement utilities. Each function lives in its own module; this re-exports
them so `import Coppelia.helpers as cop; cop.render_maze(...)` keeps working.
"""
from Coppelia.helpers.remove_object import remove_object
from Coppelia.helpers.change_floor import change_floor
from Coppelia.helpers.get_maze_tag import get_maze_tag
from Coppelia.helpers.maze_origin import maze_origin
from Coppelia.helpers.cell_to_world import cell_to_world
from Coppelia.helpers.world_to_cell import world_to_cell
from Coppelia.helpers.local_bounding_box import local_bounding_box
from Coppelia.helpers.kart_center_offset import KART_CENTER_OFFSET_LOCAL
from Coppelia.helpers.set_planar_pose import set_planar_pose
from Coppelia.helpers.render_maze import render_maze
from Coppelia.helpers.move_maze_marker import move_maze_marker
from Coppelia.helpers.add_maze_obstacle import add_maze_obstacle

__all__ = [
    "remove_object", "change_floor", "get_maze_tag",
    "maze_origin", "cell_to_world", "world_to_cell", "local_bounding_box",
    "KART_CENTER_OFFSET_LOCAL", "set_planar_pose", "render_maze",
    "move_maze_marker", "add_maze_obstacle",
]
