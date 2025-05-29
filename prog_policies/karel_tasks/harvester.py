import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class Harvester(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        init_x = self.rng.randint(2, env_width - 2)
        init_pos = [env_height - 2, init_x]
        
        state[1, init_pos[0], init_pos[1]] = True
        
        state[6, 1:env_height - 1, 1:env_width - 1] = True
        
        self.initial_number_of_markers = (env_height - 2) * (env_width - 2)
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_number_of_markers = self.initial_number_of_markers

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        
        num_markers = env.markers_grid.sum()
        
        reward = (self.previous_number_of_markers - num_markers) / self.initial_number_of_markers
        
        if num_markers > self.previous_number_of_markers:
            reward = self.crash_penalty
            terminated = True
        
        elif num_markers == 0:
            terminated = True
        
        self.previous_number_of_markers = num_markers
        
        return terminated, reward


class HarvesterSparse(Harvester):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        num_markers = env.markers_grid.sum()
        
        if num_markers > self.previous_number_of_markers:
            reward = self.crash_penalty
            terminated = True
        
        elif num_markers == 0:
            reward = 1.
            terminated = True
        
        return terminated, reward