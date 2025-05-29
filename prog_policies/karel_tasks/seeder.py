import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class Seeder(BaseTask):
        
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
        init_y = self.rng.randint(1, env_height - 1)
        
        state[1, init_y, init_x] = True
        
        state[5, :, :] = True
        
        self.max_number_of_markers = (env_width - 2) * (env_height - 2)
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_number_of_markers = 0

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        
        num_markers = env.markers_grid.sum()
        
        reward = (num_markers - self.previous_number_of_markers) / self.max_number_of_markers
        
        if (env.markers_grid > 1).any():
            reward = self.crash_penalty
            terminated = True
        
        elif num_markers < self.previous_number_of_markers:
            reward = self.crash_penalty
            terminated = True
        
        elif num_markers == self.max_number_of_markers:
            terminated = True
        
        self.previous_number_of_markers = num_markers
        
        return terminated, reward
