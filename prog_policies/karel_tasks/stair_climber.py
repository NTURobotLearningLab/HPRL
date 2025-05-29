import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment

class StairClimber(BaseTask):
    
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        for i in range(1, env_width - 2):
            state[4, env_height - i - 1, i + 1] = True
            state[4, env_height - i - 1, i + 2] = True
        
        on_stair_positions = [
            [env_height - i - 1, i] for i in range(1, env_width - 1)
        ]
        
        one_block_above_stair_positions = [
            [env_height - i - 2, i] for i in range(1, env_width - 2)
        ]
        
        # One cell above the stairs
        self.valid_positions = on_stair_positions + one_block_above_stair_positions
        
        # Initial position has to be on stair but cannot be on last step
        initial_position_index = self.rng.randint(0, len(on_stair_positions) - 1)
        
        # Marker has to be after initial position
        marker_position_index = self.rng.randint(initial_position_index + 1, len(on_stair_positions))
        
        self.initial_position = on_stair_positions[initial_position_index]
        state[1, self.initial_position[0], self.initial_position[1]] = True
        
        self.marker_position = on_stair_positions[marker_position_index]
        state[5, :, :] = True
        state[6, self.marker_position[0], self.marker_position[1]] = True
        state[5, self.marker_position[0], self.marker_position[1]] = False
        
        self.initial_distance = abs(self.initial_position[0] - self.marker_position[0]) \
            + abs(self.initial_position[1] - self.marker_position[1])
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.previous_distance = self.initial_distance
        
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        current_distance = abs(karel_pos[0] - self.marker_position[0]) \
            + abs(karel_pos[1] - self.marker_position[1])
        
        # Reward is how much closer Karel is to the marker, normalized by the initial distance
        reward = (self.previous_distance - current_distance) / self.initial_distance
        
        if [karel_pos[0], karel_pos[1]] not in self.valid_positions:
            reward = self.crash_penalty
            terminated = True
            
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
        
        self.previous_distance = current_distance
        
        return terminated, reward


class StairClimberSparse(StairClimber):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            reward = 1.
            terminated = True
        elif [karel_pos[0], karel_pos[1]] not in self.valid_positions:
            reward = self.crash_penalty
            terminated = True
        
        return terminated, reward