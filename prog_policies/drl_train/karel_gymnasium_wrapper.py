import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os


from prog_policies.karel_tasks import StairClimberSparse,  StairClimber, get_task_cls   # or StairClimber
from prog_policies.karel import KarelEnvironment


# Index → primitive action the env can execute
_ID2ACTION = ["move", "turn_left", "turn_right", "pick_marker", "put_marker"]


class KarelGymEnv(gym.Env):
    """
    Thin adapter that exposes a `BaseTask` instance as a Gymnasium Env.
    Works with SB3 out-of-the-box (supports reset(seed), step, render, close).
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        task_name='StairClimberSparse',
        env_args: dict | None = None,
        max_episode_steps: int = 50,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        task_cls = get_task_cls(task_name)
        env_args = env_args or {}
        self.task = task_cls(env_args=env_args, seed=seed)
        self.base_env: KarelEnvironment = self.task.get_environment()

        # SB3 works with CHW order, float32 0/1 tensor
        c, h, w = self.base_env.state_shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(_ID2ACTION))
        self.seed = seed
        self.render_mode = render_mode
        self._max_steps = max_episode_steps
        self._step_cnt = 0
        self.render(task_name)

    # ----------  Gymnasium API  ----------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.task.reset_environment()
        self.base_env = self.task.environment
        self._step_cnt = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self._step_cnt += 1

        # Execute primitive

        getattr(self.base_env, _ID2ACTION[action])()

        terminated, reward = self.task.get_reward(self.base_env)
        crashed = getattr(self.base_env, "crashed", False)
        truncated = self._step_cnt >= self._max_steps
        done = terminated or crashed or truncated

        obs = self._get_obs()
        info = {"crashed": crashed}
        return obs, reward, done, truncated, info

    def render(self, task_name):
        if self._step_cnt < 100:
            state_image = self.base_env.to_image()
            from PIL import Image
            img = Image.fromarray(state_image)
            save_dir = f"prog_policies/drl_train/gifs/{task_name}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.seed}.png")
            img.save(save_path)

    def close(self):
        pass

    # ----------  Helpers  ----------
    def _get_obs(self):
        # bool → float32 (0/1); CHW
        return self.base_env.get_state().astype(np.float32)
