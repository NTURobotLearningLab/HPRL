from __future__ import annotations

from ..search_space import BaseSearchSpace
from ..base import dsl_nodes, BaseTask, BaseDSL

from .base_search import BaseSearch
import math
import numpy as np


class Scheduled_HillClimbing(BaseSearch):
    
    def __init__(self, k: int, e: int, start_k: int, end_k: int, max_program_nums: int, interpolation_type: str, scheduler_type: str, ratio_type: str) -> None:
        self.start_k = start_k
        self.end_k = end_k
        self.interpolation_type = interpolation_type
        self.scheduler_type = scheduler_type
        self.ratio_type = ratio_type
        self.max_program_nums = max_program_nums
        super().__init__(k, e)
        
    def get_k_schedule(self, program_num: int) -> None:
        if program_num <= 0:
            return self.start_k
        elif program_num >= self.max_program_nums:
            return self.end_k
        
        if self.ratio_type == "log":
            program_num = np.log10(program_num)
            max_program_nums = np.log10(self.max_program_nums)
        elif self.ratio_type == "linear":
            program_num = program_num
            max_program_nums = self.max_program_nums
        else:
            raise Exception("wrong ratio type")
        
        ratio = program_num / max_program_nums
        if self.scheduler_type == "sin":
            ratio = (math.sin((ratio * 2 - 1) * math.pi / 2) + 1) / 2
        elif self.scheduler_type == "linear":
            pass
        else:
            raise Exception("wrong scheduler type")
        
        if self.interpolation_type == "log":
            start_k_log = np.log2(self.start_k)
            end_k_log = np.log2(self.end_k)
            
            k_log = start_k_log + (end_k_log - start_k_log) * ratio
            k = math.ceil(2 ** k_log)
        elif self.interpolation_type == "linear":
            start_k = self.start_k
            end_k = self.end_k
            
            k = start_k + (end_k - start_k) * ratio
            k = math.ceil(k)
        else:
            raise Exception("wrong interpolation type")
        
        return k
    
    def record_search(self, search_space: BaseSearchSpace, task_envs: list[BaseTask],
               seed: int | None = None, n_iterations: int = 10000, init = None, dsl: BaseDSL = None, record_type: str = "") -> tuple[list[dsl_nodes.Program], list[float]]:
        """Performs hill climbing in the search space (any search space can be used), stopping when
        a local maximum is reached or when the maximum number of iterations is reached

        Args:
            search_space (BaseSearchSpace): Search space instance
            task_envs (list[BaseTask]): List of task environments for evaluation
            seed (int, optional): If provided, sets the search space RNG seed. Defaults to None.
            n_iterations (int, optional): Maximum number of iterations. Defaults to 10000.

        Returns:
            list[float]: List of rewards obtained at each iteration
        """
        self.k = self.get_k_schedule(task_envs[0].program_num)
        rewards = []
        if seed:
            search_space.set_seed(seed)
        if init is None:
            best_ind, best_prog = search_space.initialize_individual()
        else:
            best_ind, best_prog = init
        best_reward = self.record_evaluate_program(best_prog, task_envs, dsl, record_type=record_type)
        rewards.append(best_reward)
        progs = [best_prog]
        for _ in range(n_iterations):
            if best_reward >= 1.0:
                break
            candidates = search_space.get_neighbors(best_ind, k=self.k)
            in_local_maximum = True
            for ind, prog in candidates:
                reward = self.record_evaluate_program(prog, task_envs, dsl, record_type=record_type)
                if reward > best_reward:
                    best_ind = ind
                    best_prog = prog
                    best_reward = reward
                    in_local_maximum = False
                    break
            if in_local_maximum:
                break
            rewards.append(best_reward)
            progs.append(best_prog)
        return progs, rewards
    
    def search(self, search_space: BaseSearchSpace, task_envs: list[BaseTask],
               seed: int | None = None, n_iterations: int = 10000, init = None) -> tuple[list[dsl_nodes.Program], list[float]]:
        """Performs hill climbing in the search space (any search space can be used), stopping when
        a local maximum is reached or when the maximum number of iterations is reached

        Args:
            search_space (BaseSearchSpace): Search space instance
            task_envs (list[BaseTask]): List of task environments for evaluation
            seed (int, optional): If provided, sets the search space RNG seed. Defaults to None.
            n_iterations (int, optional): Maximum number of iterations. Defaults to 10000.

        Returns:
            list[float]: List of rewards obtained at each iteration
        """
        self.k = self.get_k_schedule(task_envs[0].program_num)
        rewards = []
        if seed:
            search_space.set_seed(seed)
        if init is None:
            best_ind, best_prog = search_space.initialize_individual()
        else:
            best_ind, best_prog = init
        best_reward = self.evaluate_program(best_prog, task_envs)
        rewards.append(best_reward)
        progs = [best_prog]
        for _ in range(n_iterations):
            if best_reward >= 1.0:
                break
            candidates = search_space.get_neighbors(best_ind, k=self.k)
            in_local_maximum = True
            for ind, prog in candidates:
                reward = self.evaluate_program(prog, task_envs)
                if reward > best_reward:
                    best_ind = ind
                    best_prog = prog
                    best_reward = reward
                    in_local_maximum = False
                    break
            if in_local_maximum:
                break
            rewards.append(best_reward)
            progs.append(best_prog)
        return progs, rewards