import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class Maze(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        def get_neighbors(cur_pos):
            neighbor_list = []
            #neighbor top
            if cur_pos[0] - 2 > 0: neighbor_list.append([cur_pos[0] - 2, cur_pos[1]])
            # neighbor bottom
            if cur_pos[0] + 2 < env_height - 1: neighbor_list.append([cur_pos[0] + 2, cur_pos[1]])
            # neighbor left
            if cur_pos[1] - 2 > 0: neighbor_list.append([cur_pos[0], cur_pos[1] - 2])
            # neighbor right
            if cur_pos[1] + 2 < env_width - 1: neighbor_list.append([cur_pos[0], cur_pos[1] + 2])
            return neighbor_list
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        state[4, :, :] = True
        
        init_pos = [env_height - 2, 1]
        state[1, init_pos[0], init_pos[1]] = True
        state[4, init_pos[0], init_pos[1]] = False
        visited = np.zeros((env_height, env_width), dtype=bool)
        visited[init_pos[0], init_pos[1]] = True
        
        stack = [init_pos]
        while len(stack) > 0:
            cur_pos = stack.pop()
            neighbors = get_neighbors(cur_pos)
            self.rng.shuffle(neighbors)
            for neighbor in neighbors:
                if not visited[neighbor[0], neighbor[1]]:
                    visited[neighbor[0], neighbor[1]] = True
                    state[4, (cur_pos[0] + neighbor[0]) // 2, (cur_pos[1] + neighbor[1]) // 2] = False
                    state[4, neighbor[0], neighbor[1]] = False
                    stack.append(neighbor)
        
        valid_loc = False
        state[5, :, :] = True
        while not valid_loc:
            ym = self.rng.randint(1, env_height - 1)
            xm = self.rng.randint(1, env_width - 1)
            if not state[4, ym, xm] and not state[1, ym, xm]:
                valid_loc = True
                state[6, ym, xm] = True
                state[5, ym, xm] = False
                self.marker_position = [ym, xm]
        
        self.initial_distance = abs(init_pos[0] - self.marker_position[0]) \
            + abs(init_pos[1] - self.marker_position[1])
        
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

        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
        
        self.previous_distance = current_distance
        
        return terminated, reward


class MazeSparse(Maze):
    
    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.

        karel_pos = env.get_hero_pos()
        
        if karel_pos[0] == self.marker_position[0] and karel_pos[1] == self.marker_position[1]:
            terminated = True
            reward = 1.
        
        return terminated, reward