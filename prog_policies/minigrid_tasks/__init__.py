from __future__ import annotations
from prog_policies.base import BaseTask

from .lavagap import LavaGap
from .redbluedoor import RedBlueDoor
from .putnear import PutNear

TASK_NAME_LIST = [
    "LavaGap",
    "RedBlueDoor",
    "PutNear",
]

def get_task_cls(task_cls_name: str):
    task_cls = globals().get(task_cls_name)
    return task_cls
