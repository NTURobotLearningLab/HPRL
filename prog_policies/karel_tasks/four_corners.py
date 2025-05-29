import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class FourCorners(BaseTask):
        
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
        
        self.goal_markers = [
            [1, 1],
            [env_height - 2, 1],
            [1, env_width - 2],
            [env_height - 2, env_width - 2]
        ]
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.num_previous_correct_markers = 0

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        
        num_placed_markers = env.markers_grid.sum()
        num_correct_markers = 0

        for marker in self.goal_markers:
            if env.markers_grid[marker[0], marker[1]]:
                num_correct_markers += 1
        
        reward = (num_correct_markers - self.num_previous_correct_markers) / len(self.goal_markers)

        if num_placed_markers > num_correct_markers:
            terminated = True
            reward = self.crash_penalty
        
        elif num_correct_markers == len(self.goal_markers):
            terminated = True
            
        self.num_previous_correct_markers = num_correct_markers
        
        return terminated, reward


class FourCornersSparse(FourCorners):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        num_placed_markers = env.markers_grid.sum()
        num_correct_markers = 0
        
        for marker in self.goal_markers:
            if env.markers_grid[marker[0], marker[1]]:
                num_correct_markers += 1

        if num_placed_markers > num_correct_markers:
            terminated = True
            reward = self.crash_penalty
        
        elif num_correct_markers == len(self.goal_markers):
            terminated = True
            reward = 1.
        
        return terminated, reward