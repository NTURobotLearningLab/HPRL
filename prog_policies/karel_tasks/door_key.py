from math import ceil
import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class DoorKey(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        wall_column = ceil(env_width / 2)
        state[4, :, wall_column] = True
        
        self.key_cell = (self.rng.randint(1, env_height - 1), self.rng.randint(1, wall_column))
        self.end_marker_cell = (self.rng.randint(1, env_height - 1), self.rng.randint(wall_column + 1, env_width - 1))
        
        state[5, :, :] = True
        state[6, self.key_cell[0], self.key_cell[1]] = True
        state[5, self.key_cell[0], self.key_cell[1]] = False
        state[6, self.end_marker_cell[0], self.end_marker_cell[1]] = True
        state[5, self.end_marker_cell[0], self.end_marker_cell[1]] = False
        
        valid_loc = False
        while not valid_loc:
            y_agent = self.rng.randint(1, env_height - 1)
            x_agent = self.rng.randint(1, wall_column)
            if not state[6, y_agent, x_agent]:
                valid_loc = True
                state[1, y_agent, x_agent] = True

        self.door_cells = [(2, wall_column), (3, wall_column)]
        self.door_locked = True
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.door_locked = True

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.
        num_markers = env.markers_grid.sum()
        
        if self.door_locked:
            if num_markers > 2:
                terminated = True
                reward = self.crash_penalty
            # Check if key has been picked up
            elif env.markers_grid[self.key_cell[0], self.key_cell[1]] == 0:
                self.door_locked = False
                for door_cell in self.door_cells:
                    env.state[4, door_cell[0], door_cell[1]] = False
                reward = 0.5
        else:
            if num_markers > 1:
                # Check if end marker has been topped off
                if env.markers_grid[self.end_marker_cell[0], self.end_marker_cell[1]] == 2:
                    terminated = True
                    reward = 0.5
                else:
                    terminated = True
                    reward = self.crash_penalty
            elif num_markers == 0:
                terminated = True
                reward = self.crash_penalty
        
        return terminated, reward
