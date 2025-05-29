import gymnasium as gym
from gymnasium import spaces
import numpy as np
import imageio

class KarelGymWrapper(gym.Env):
    def __init__(self, env_task="stairClimber_sparse", grid_size=8):
        super().__init__()
        from karel_env.karel_option_key2door import Karel_world

        self.grid_size = grid_size
        self.num_channels = 8  # based on your state_table
        self.karel = Karel_world(env_task=env_task, reward_diff=True, task_definition='custom')
        self.max_steps = 50
        self.step_count = 0

        # Define gym-style spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, self.num_channels), dtype=np.float32)
        self.action_space = spaces.Discrete(self.karel.num_actions)

        # Initial reset
        self._reset_world()

    def _reset_world(self):
        from karel_env.generator_option_key2door import KarelStateGenerator
        s_gen = KarelStateGenerator()
        state, _, _, _, metadata = s_gen.generate_single_state_stair_climber()
        self.karel.set_new_state(state, metadata=metadata)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_world()
        self.step_count = 0
        obs = self.karel.render().astype(np.float32)
        return obs, {}  # second return value is the "info" dict as per Gymnasium API

    def step(self, action):
        one_hot_action = np.eye(self.action_space.n)[action]
        try:
            self.karel.state_transition(one_hot_action)
        except RuntimeError:
            pass  # invalid action

        obs = self.karel.render().astype(np.float32)
        reward, done = self.karel._get_state_reward(self.karel.get_location())
        self.step_count += 1

        # New API: separate `terminated` (task completed) vs `truncated` (timeout or interrupt)
        terminated = done
        truncated = self.step_count >= self.max_steps

        info = {
            "steps": self.step_count,
            "progress": self.karel.progress_ratio,
            "done_flag": self.karel.done,
        }

        return obs, reward, terminated, truncated, info


    def render(self, mode='human'):
        return self.karel.render()
    def save_gif(self, path, s_h):
        # create video
        frames = []
        for s in s_h:
            frames.append(np.uint8(self.karel.state2image(s=s).squeeze()))
        frames = np.stack(frames, axis=0)
        imageio.mimsave(path, frames, format='GIF-PIL', fps=5)
        #optimize(path)

        return
    # def render(self, mode='rgb_array'):

    #     import numpy as np
    #     import cv2
    #     grid = self.karel.render().astype(np.float32)  # shape (H, W, 8)
    #     H, W, _ = grid.shape


    #     img = np.ones((H * 20, W * 20, 3), dtype=np.uint8) * 255  # white canvas

    #     for i in range(H):
    #         for j in range(W):
    #             cell = grid[i, j]
    #             color = (200, 200, 200)

    #             if cell[4] == 1:
    #                 color = (0, 0, 0)  # wall
    #             elif cell[7] == 1:
    #                 color = (0, 0, 255)
    #             elif cell[6] == 1:
    #                 color = (0, 255, 255)
    #             elif cell[5] == 1:
    #                 color = (0, 255, 0)

    #             for d in range(4):
    #                 if cell[d] == 1:
    #                     color = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (127, 0, 255)][d]

    #             cv2.rectangle(img, (j * 20, i * 20), (j * 20 + 19, i * 20 + 19), color, -1)

    #     return img


    def seed(self, seed=None):
        np.random.seed(seed)
