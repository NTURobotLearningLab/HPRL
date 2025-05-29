from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Union, Callable

import numpy as np

class BaseEnvironment(ABC):
    
    def __init__(self, actions: dict[str, Callable], bool_features: dict[str, Callable],
                 int_features: dict[str, Callable], state_shape: tuple[int, ...],
                 initial_state: Union[Any, None] = None, max_calls: int = 10000):
        self.actions = actions
        self.actions_list = list(actions.keys())
        self.bool_features = bool_features
        self.bool_features_list = list(bool_features.keys())
        self.int_features = int_features
        self.int_features_list = list(int_features.keys())
        self.max_calls = max_calls
        self.state_shape = state_shape
        self.num_calls: int = 0
        self.crashed: bool = False
        if initial_state is not None:
            self.set_state(initial_state)
        else:
            self.set_state(self.default_state())

    def is_crashed(self) -> bool:
        return self.crashed
    
    def crash(self):
        self.crashed = True

    def get_bool_feature(self, feature: str):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.crashed = True
        return self.bool_features[feature]()

    def get_int_feature(self, feature: str):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.crashed = True
        return self.int_features[feature]()

    def run_action(self, action: str):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.crashed = True
        self.actions[action]()
        
    def run_action_index(self, action_index: int):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.crashed = True
        self.actions_list[action_index]()

    @abstractmethod
    def default_state(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self):
        pass
    
    @abstractmethod
    def __eq__(self, other: BaseEnvironment) -> bool:
        pass

    @classmethod
    @abstractmethod
    def from_string(cls, state_str: str):
        pass


    @abstractmethod
    def to_image(self, grid_size: int = 100, root_dir: str = "./") -> np.ndarray:
        pass

    @abstractmethod
    def to_string_worldcoder_style(self, state: np.ndarray = None) -> str:
        # The prompt of Grid-based Environment is from the paper "WorldCoder" arXiv:2402.12275
        pass
    
    # For record
    @abstractmethod    
    def record_partial_state(self) -> str:
        pass