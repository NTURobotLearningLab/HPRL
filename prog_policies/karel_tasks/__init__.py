from __future__ import annotations
from prog_policies.base import BaseTask

from .stair_climber import StairClimber, StairClimberSparse
from .maze import Maze, MazeSparse
from .four_corners import FourCorners, FourCornersSparse
from .top_off import TopOff, TopOffSparse
from .harvester import Harvester, HarvesterSparse
from .clean_house import CleanHouse, CleanHouseSparse
from .door_key import DoorKey
from .one_stroke import OneStroke
from .seeder import Seeder
from .snake import Snake
from .wall_avoider import WallAvoider
from .path_follow import PathFollow

TASK_NAME_LIST = [
    "StairClimber",
    "StairClimberSparse",
    "Maze",
    "MazeSparse",
    "FourCorners",
    "FourCornersSparse",
    "TopOff",
    "TopOffSparse",
    "Harvester",
    "HarvesterSparse",
    "CleanHouse",
    "CleanHouseSparse",
    "DoorKey",
    "OneStroke",
    "Seeder",
    "Snake",
    "WallAvoider",
    "PathFollow",
]

def get_task_cls(task_cls_name: str) -> type[BaseTask]:
    task_cls = globals().get(task_cls_name)
    assert issubclass(task_cls, BaseTask)
    return task_cls