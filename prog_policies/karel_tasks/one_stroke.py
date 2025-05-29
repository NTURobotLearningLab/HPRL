import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment

# TODO: make trail of visited cells be markers instead of walls
class OneStroke(BaseTask):
        
    def generate_initial_environment(self, env_args):
        
        reference_env = KarelEnvironment(**env_args)
        
        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]        
        
        state = np.zeros(reference_env.state_shape, dtype=bool)
        
        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True
        
        self.initial_agent_x = self.rng.randint(1, env_width - 1)
        self.initial_agent_y = self.rng.randint(1, env_height - 1)
        
        state[1, self.initial_agent_y, self.initial_agent_x] = True
        
        state[5, :, :] = True
        
        self.max_number_cells = (env_width - 2) * (env_height - 2) - 1
        
        return KarelEnvironment(initial_state=state, **env_args)
    
    def reset_environment(self):
        super().reset_environment()
        self.prev_agent_x = self.initial_agent_x
        self.prev_agent_y = self.initial_agent_y
        self.number_cells_visited = 0

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        
        agent_y, agent_x, _ = env.get_hero_pos()
        
        if (agent_x == self.prev_agent_x) and (agent_y == self.prev_agent_y):
            reward = 0.
        else:
            self.number_cells_visited += 1
            reward = 1. / self.max_number_cells
            # Place a wall where the agent was
            env.state[4, self.prev_agent_y, self.prev_agent_x] = True

        if self.number_cells_visited == self.max_number_cells:
            terminated = True
        
        self.prev_agent_x = agent_x
        self.prev_agent_y = agent_y
        
        return terminated, reward
