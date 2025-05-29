import os
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from prog_policies.drl_train import save_gif


class GifEvalCallback(EvalCallback):
    """
    Standard EvalCallback + record the first eval episode as a GIF.
    The GIF is saved to  <save_dir>/eval_<global_step>.gif
    """

    def __init__(self, *args, save_dir: str = "gifs", **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------
    def _vec_reset(self, vec_env):
        """Reset that works for both old (SB3<=1.8) and new (SB3>=2.2) APIs."""
        result = vec_env.reset()
        if isinstance(result, tuple):          # gymnasium: (obs, info)
            obs, _ = result
        else:                                  # gym: obs
            obs = result
        return obs

    def _vec_step(self, vec_env, action):
        """Step wrapper returning obs, done, truncated compatible flags."""
        result = vec_env.step(action)
        if len(result) == 4:                   # gym: obs, rew, done, info
            obs, _, done, _ = result
            truncated = np.zeros_like(done, dtype=bool)
        else:                                  # gymnasium: obs, rew, term, trunc, info
            obs, _, done, truncated, _ = result
        return obs, done, truncated

    # ------------------------------------------------------------
    def _run_gif_episode(self):
        vec_env      = self.eval_env             # DummyVecEnv
        env          = vec_env.envs[0]           # underlying KarelGymEnv

        obs    = self._vec_reset(vec_env)
        states = [env.base_env.get_state().copy()]
        

        done = truncated = False
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, done_vec, trunc_vec = self._vec_step(vec_env, action)

            # done_vec / trunc_vec are 1-element arrays → scalar
            done       = bool(done_vec[0]) if np.ndim(done_vec) else bool(done_vec)
            truncated  = bool(trunc_vec[0]) if np.ndim(trunc_vec) else bool(trunc_vec)

            states.append(env.base_env.get_state().copy())

        gif_path = os.path.join(self.save_dir, f"eval_{self.num_timesteps}.gif")
        save_gif(gif_path, states)

    # ------------------------------------------------------------
    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        # After each evaluation phase (`EvalCallback` sets self.n_calls)
        if self.n_calls % self.eval_freq == 0:
            self._run_gif_episode()

        return continue_training
