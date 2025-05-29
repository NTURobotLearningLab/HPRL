from __future__ import annotations

from ..search_space import BaseSearchSpace
from ..base import dsl_nodes, BaseTask, BaseDSL

from .base_search import BaseSearch


class HillClimbingLatent(BaseSearch):
    
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
        rewards = []
        if seed:
            search_space.set_seed(seed)
        if init is None:
            best_ind, best_prog = search_space.initialize_individual()
        else:
            _, init_prog = init
            n_tries = 0
            while n_tries < 50:
                try :
                    best_ind = search_space._encode(init_prog)
                    best_prog = search_space._decode(best_ind)
                    break
                except:
                    n_tries += 1
                    continue
            if n_tries >= 50:
                best_ind, best_prog = search_space.initialize_individual()
                
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
        rewards = []
        if seed:
            search_space.set_seed(seed)
        if init is None:
            best_ind, best_prog = search_space.initialize_individual()
        else:
            _, init_prog = init
            n_tries = 0
            while n_tries < 50:
                try :
                    best_ind = search_space._encode(init_prog)
                    best_prog = search_space._decode(best_ind)
                    break
                except:
                    n_tries += 1
                    continue
            if n_tries >= 50:
                best_ind, best_prog = search_space.initialize_individual()
        
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
