from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
import numpy as np
import torch

from ..base.dsl import BaseDSL, dsl_nodes

class BaseSearchSpace(ABC):
    
    def __init__(self, dsl: BaseDSL, sigma: float = 0.25) -> None:
        super().__init__()
        self.dsl = dsl
        self.sigma = sigma
        self.np_rng = np.random.RandomState(1)
        self.torch_device = torch.device('cpu')
        self.torch_rng = torch.Generator(device=self.torch_device)
        self.torch_rng.manual_seed(self.np_rng.randint(1000000))

    def set_seed(self, seed: int):
        """Sets a manual seed for the search space

        Args:
            seed (int): Seed for RNGs
        """
        self.np_rng = np.random.RandomState(seed)
        self.torch_rng.manual_seed(self.np_rng.randint(1000000))
    
    @abstractmethod
    def initialize_individual(self) -> tuple[Any, dsl_nodes.Program]:
        """Initializes an individual in the search space

        Returns:
            tuple[Any, dsl_nodes.Program]: Individual and its corresponding program
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, individual: Any, k: int = 1) -> list[tuple[Any, dsl_nodes.Program]]:
        """Returns k neighbors of a given individual

        Args:
            individual (Any): Input individual
            k (int, optional): Number of neighbors. Defaults to 1.

        Returns:
            list[tuple[Any, dsl_nodes.Program]]: List of individuals as tuple of individual and
            associated program
        """
        pass
