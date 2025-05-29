from __future__ import annotations

import torch

from prog_policies.base.dsl import BaseDSL

from ..search_space import LatentSpace
from ..base import dsl_nodes, BaseTask

from .base_search import BaseSearch


class CEBS(BaseSearch):
    
    def record_search(self, search_space: LatentSpace, task_envs: list[BaseTask],
               seed: int | None = None, n_iterations: int = 10000, init = None, dsl: BaseDSL = None, record_type: str = "") -> tuple[list[dsl_nodes.Program], list[float]]:
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
        best_elite_mean = -float('inf')

        candidates = search_space.get_neighbors(best_ind, k=self.k)        
        for _ in range(n_iterations):
            
            candidate_rewards = []
            for _, prog in candidates:
                reward = self.record_evaluate_program(prog, task_envs, dsl, record_type=record_type)
                candidate_rewards.append(reward)
                if reward > best_reward:
                    best_reward = reward
                    best_prog = prog
            
            torch_candidates = torch.stack([ind for ind, _ in candidates])
            torch_rewards = torch.tensor(candidate_rewards, device=torch_candidates.device)
            
            elite_indices = torch.topk(torch_rewards, self.e, largest=True).indices
            elite_candidates = torch_candidates[elite_indices]
            elite_rewards = torch_rewards[elite_indices]
            
            mean_elite_reward = torch.mean(elite_rewards, dim=0)
            if mean_elite_reward > best_elite_mean:
                best_elite_mean = mean_elite_reward
            else:
                break
            
            candidates = []
            for candidate in elite_candidates:
                candidates += search_space.get_neighbors(candidate, k=self.k//self.e)
                
            rewards.append(best_reward)
            progs.append(best_prog)
            
        return progs, rewards
    
    def search(self, search_space: LatentSpace, task_envs: list[BaseTask],
               seed: int | None = None, n_iterations: int = 10000) -> tuple[list[dsl_nodes.Program], list[float]]:
        rewards = []
        if seed:
            search_space.set_seed(seed)
        best_ind, best_prog = search_space.initialize_individual()
        best_reward = self.evaluate_program(best_prog, task_envs)
        rewards.append(best_reward)
        progs = [best_prog]
        best_elite_mean = -float('inf')

        candidates = search_space.get_neighbors(best_ind, k=self.k)        
        for _ in range(n_iterations):
            
            candidate_rewards = []
            for _, prog in candidates:
                reward = self.evaluate_program(prog, task_envs)
                candidate_rewards.append(reward)
                if reward > best_reward:
                    best_reward = reward
                    best_prog = prog
            
            torch_candidates = torch.stack([ind for ind, _ in candidates])
            torch_rewards = torch.tensor(candidate_rewards, device=torch_candidates.device)
            
            elite_indices = torch.topk(torch_rewards, self.e, largest=True).indices
            elite_candidates = torch_candidates[elite_indices]
            elite_rewards = torch_rewards[elite_indices]
            
            mean_elite_reward = torch.mean(elite_rewards, dim=0)
            if mean_elite_reward > best_elite_mean:
                best_elite_mean = mean_elite_reward
            else:
                break
            
            candidates = []
            for candidate in elite_candidates:
                candidates += search_space.get_neighbors(candidate, k=self.k//self.e)
                
            rewards.append(best_reward)
            progs.append(best_prog)
            
        return progs, rewards
