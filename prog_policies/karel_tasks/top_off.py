import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class TopOff(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        state[1, env_height - 2, 1] = True
        
        self.possible_marker_locations = [
            [env_height - 2, i] for i in range(2, env_width - 1)
        ]
        
        self.rng.shuffle(self.possible_marker_locations)
        
        self.num_markers = self.rng.randint(1, len(self.possible_marker_locations))
        self.markers = self.possible_marker_locations[:self.num_markers]
        
        for marker in self.markers:
            state[6, marker[0], marker[1]] = True
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.num_previous_correct_markers = 0

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        
        num_markers = env.markers_grid.sum()
        num_correct_markers = 0

        for marker in self.markers:
            if env.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif env.markers_grid[marker[0], marker[1]] == 0:
                return True, self.crash_penalty
        
        reward = (num_correct_markers - self.num_previous_correct_markers) / len(self.markers)
        
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = self.crash_penalty
        
        elif num_correct_markers == len(self.markers):
            terminated = True
            
        self.num_previous_correct_markers = num_correct_markers
        
        return terminated, reward


class TopOffSparse(TopOff):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        num_correct_markers = 0
        reward = 0.

        for marker in self.markers:
            if env.markers_grid[marker[0], marker[1]] == 2:
                num_correct_markers += 1
            elif env.markers_grid[marker[0], marker[1]] == 0:
                return True, self.crash_penalty
        
        num_markers = env.markers_grid.sum()
        if num_markers > num_correct_markers + len(self.markers):
            terminated = True
            reward = self.crash_penalty
        
        if num_correct_markers == len(self.markers):
            terminated = True
            reward = 1.
        
        return terminated, reward