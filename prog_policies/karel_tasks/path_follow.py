from __future__ import annotations
import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment

class PathFollow(BaseTask):
    
    def pos_to_id(self, y: int, x: int, env_height: int) -> int:
        return y*env_height + x
    
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        # Place walls
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        # Place karel at the left bottom corner and facing up
        state[0, env_height - 2, 1] = True
        
        # Put markers, Up means 0, right means 1
        # Record legal positions
        self.path = [0] * (env_height - 3) + [1] * (env_width - 3)
        self.rng.shuffle(self.path)
        
        # Start pos
        self.legal_pos: set[int] = set()
        start_pos = [env_height - 2, 1]
        state[6, start_pos[0], start_pos[1]] = True
        self.legal_pos.add(self.pos_to_id(start_pos[0], start_pos[1], env_height))
        self.legal_pos.add(self.pos_to_id(start_pos[0] - 1, start_pos[1], env_height))
        self.legal_pos.add(self.pos_to_id(start_pos[0] + 1, start_pos[1], env_height))
        self.legal_pos.add(self.pos_to_id(start_pos[0], start_pos[1] - 1, env_height))
        self.legal_pos.add(self.pos_to_id(start_pos[0], start_pos[1] + 1, env_height))
        
        # Path 
        for p in self.path:
            if p == 0:
                start_pos[0] -= 1
            else:
                start_pos[1] += 1
            state[6, start_pos[0], start_pos[1]] = True
            self.legal_pos.add(self.pos_to_id(start_pos[0], start_pos[1], env_height))
            self.legal_pos.add(self.pos_to_id(start_pos[0] - 1, start_pos[1], env_height))
            self.legal_pos.add(self.pos_to_id(start_pos[0] + 1, start_pos[1], env_height))
            self.legal_pos.add(self.pos_to_id(start_pos[0], start_pos[1] - 1, env_height))
            self.legal_pos.add(self.pos_to_id(start_pos[0], start_pos[1] + 1, env_height))
            
        self.initial_number_of_markers = env_height + env_width - 5
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_number_of_markers = self.initial_number_of_markers
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        num_markers = env.markers_grid.sum()
        
        _, env_height, env_width = env.state_shape
        
        reward = (self.previous_number_of_markers - num_markers) / self.initial_number_of_markers
        
        if num_markers > self.previous_number_of_markers:
            reward = self.crash_penalty
            terminated = True
        
        elif num_markers == 0:
            terminated = True
            
        agent_y, agent_x, _ = env.get_hero_pos()
        
        if self.pos_to_id(agent_y, agent_x, env_height) not in self.legal_pos:
            reward = self.crash_penalty
            terminated = True
        
        self.previous_number_of_markers = num_markers
        
        return terminated, reward            
        