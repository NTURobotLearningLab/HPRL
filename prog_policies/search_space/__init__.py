from __future__ import annotations

from .base_space import BaseSearchSpace
from .latent_space import LatentSpace
from .programmatic_space import ProgrammaticSpace, MinigridProgrammaticSpace

def get_search_space_cls(search_space_cls_name: str) -> type[BaseSearchSpace]:
    search_space_cls = globals().get(search_space_cls_name)
    return search_space_cls
