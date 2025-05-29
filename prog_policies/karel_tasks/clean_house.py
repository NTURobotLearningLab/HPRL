import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class CleanHouse(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        world_map = [
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            ['-',   0,   0, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',   0, '-',   0,   0,   0,   0,   0,   0, '-'],
            ['-',   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-', '-', '-', '-',   0, '-', '-'],
            ['-', '-',   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-',   0,   0,   0,   0,   0,   0,   0,   0, '-',   0,   0,   0,   0,   0,   0,   0,   0,   0, '-', '-'],
            ['-', '-', '-',   0, '-',   0, '-', '-', '-',   0, '-',   0,   0, '-', '-', '-',   0, '-',   0, '-', '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-',   0,   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-', '-',   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-',   0,   0,   0, '-',   0,   0,   0, '-', '-',   0,   0, '-', '-',   0,   0,   0, '-',   0,   0, '-', '-'],
            ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        ]
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        assert env_height == 14 and env_width == 22
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        agent_pos = (1, 13)
        hardcoded_invalid_marker_locations = set([(1, 13), (2, 12), (3, 10), (4, 11), (5, 11), (6, 10)])
        state[2, agent_pos[0], agent_pos[1]] = True
        
        state[5, :, :] = True
        possible_marker_locations = []
        
        for y1 in range(env_height):
            for x1 in range(env_width):
                if world_map[y1][x1] == '-':
                    state[4, y1, x1] = True
        
        expected_marker_positions = set()
        for y1 in range(env_height):
            for x1 in range(env_width):
                if state[4, y1, x1]:
                    if y1 - 1 > 0 and not state[4, y1 -1, x1]: expected_marker_positions.add((y1 - 1,x1))
                    if y1 + 1 < env_height - 1 and not state[4, y1 +1, x1]: expected_marker_positions.add((y1 + 1,x1))
                    if x1 - 1 > 0 and not state[4, y1, x1 - 1]: expected_marker_positions.add((y1,x1 - 1))
                    if x1 + 1 < env_width - 1 and not state[4, y1, x1 + 1]: expected_marker_positions.add((y1,x1 + 1))
        
        possible_marker_locations = list(expected_marker_positions - hardcoded_invalid_marker_locations)
        self.rng.shuffle(possible_marker_locations)
        
        for marker_location in possible_marker_locations[:10]:
            state[5, marker_location[0], marker_location[1]] = False
            state[6, marker_location[0], marker_location[1]] = True
        
        # put 1 marker near start point for end condition
        state[5, agent_pos[0]+1, agent_pos[1]-1] = False
        state[6, agent_pos[0]+1, agent_pos[1]-1] = True
        
        self.initial_number_of_markers = state[6, :, :].sum()
        
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


class CleanHouseSparse(CleanHouse):
    
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