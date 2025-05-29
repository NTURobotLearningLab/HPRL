from __future__ import annotations

from .base_search import BaseSearch
from .hill_climbing import HillClimbing
from .cem import CEM
from .cebs import CEBS
from .hill_climbing_latent import HillClimbingLatent
from .scheduled_hill_climbing import Scheduled_HillClimbing

def get_search_method_cls(search_cls_name: str) -> type[BaseSearch]:
    search_cls = globals().get(search_cls_name)
    assert issubclass(search_cls, BaseSearch)
    return search_cls
