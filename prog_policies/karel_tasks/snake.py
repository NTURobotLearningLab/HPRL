import numpy as np

from prog_policies.base import BaseTask
from prog_policies.karel import KarelEnvironment


class Snake(BaseTask):

    def generate_initial_environment(self, env_args):

        reference_env = KarelEnvironment(**env_args)

        env_height = reference_env.state_shape[1]
        env_width = reference_env.state_shape[2]

        state = np.zeros(reference_env.state_shape, dtype=bool)

        state[4, :, 0] = True
        state[4, :, env_width - 1] = True
        state[4, 0, :] = True
        state[4, env_height - 1, :] = True

        self.initial_agent_x = self.rng.randint(2, env_width - 2)
        self.initial_agent_y = self.rng.randint(1, env_height - 1)

        state[1, self.initial_agent_y, self.initial_agent_x] = True

        state[5, :, :] = True

        self.initial_body_size = 2

        valid_loc = False
        while not valid_loc:
            ym = self.rng.randint(1, env_height - 1)
            xm = self.rng.randint(1, env_width - 1)
            if not state[1, ym, xm]:
                valid_loc = True
                state[6, ym, xm] = True
                state[5, ym, xm] = False
                self.initial_marker_position = [ym, xm]

        return KarelEnvironment(initial_state=state, **env_args)

    def reset_environment(self):
        super().reset_environment()
        self.body_size = self.initial_body_size
        self.body_list = [(self.initial_agent_y, self.initial_agent_x)]
        self.marker_position = self.initial_marker_position.copy()

    def get_reward(self, env: KarelEnvironment):
        terminated = False
        reward = 0.0

        # Update body and check if it reached marker
        agent_y, agent_x, _ = env.get_hero_pos()
        if (agent_y == self.marker_position[0]) and (
            agent_x == self.marker_position[1]
        ):
            self.body_size += 1
            # env.state[6, self.marker_position[0], self.marker_position[1]] = False
            # env.state[5, self.marker_position[0], self.marker_position[1]] = True
            env.pick_marker_env(self.marker_position[0], self.marker_position[1])
            reward = 1 / 20
            if self.body_size == 20 + self.initial_body_size:
                terminated = True
            else:
                valid_loc = False
                while not valid_loc:
                    ym = self.rng.randint(1, env.state_shape[1] - 1)
                    xm = self.rng.randint(1, env.state_shape[2] - 1)
                    if np.sum(env.state[:5, ym, xm]) <= 0 and ((ym, xm) not in self.body_list):
                        valid_loc = True
                        # env.state[6, ym, xm] = True
                        # env.state[5, ym, xm] = False
                        env.put_marker_env(ym, xm)
                        self.marker_position = [ym, xm]

        last_y, last_x = self.body_list[-1]
        if (agent_y, agent_x) in self.body_list[:-1]:
            terminated = True
            reward = self.crash_penalty
        elif agent_y != last_y or agent_x != last_x:
            # env.state[6, last_y, last_x] = True
            # env.state[5, last_y, last_x] = False
            env.put_marker_env(last_y, last_x)
            self.body_list.append((agent_y, agent_x))
            if len(self.body_list) > self.body_size:
                first_y, first_x = self.body_list.pop(0)
                # env.state[6, first_y, first_x] = False
                # env.state[5, first_y, first_x] = True
                env.pick_marker_env(first_y, first_x)

        return terminated, reward
