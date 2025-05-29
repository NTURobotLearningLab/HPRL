# train_karel_agent.py
import sys
sys.path.insert(0, '.')
import gymnasium as gym
from stable_baselines3 import PPO  # or DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from drlgs_experts.register_env import register_karel_env
from drlgs_experts.custom_cnn import CustomCNN

from stable_baselines3.common.callbacks import BaseCallback
import wandb
import imageio
from moviepy import VideoFileClip
import os



class CustomWandbVideoCallback(BaseCallback):
    def __init__(self, session_name, log_freq=1000, rollout_freq=5000, max_rollout_steps=50, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rollout_freq = rollout_freq
        self.max_rollout_steps = max_rollout_steps

    def unwrap_env(self, env):
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _on_step(self) -> bool:
        # WandB reward logging
        if self.n_calls % self.log_freq == 0:
            wandb.log({
                "reward": self.locals["rewards"][0],
                "timesteps": self.num_timesteps,
            }, step=self.num_timesteps)

        # Run rollout and log video
        if self.n_calls % self.rollout_freq == 0:
            rollout_env = gym.make("KarelStairClimber-v1")
            base_rollout_env = self.unwrap_env(rollout_env)
            obs, _ = rollout_env.reset()
            s_h = []

            for _ in range(self.max_rollout_steps):
                last_state = rollout_env.render()
                s_h.append(last_state)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = rollout_env.step(action)
                if terminated or truncated:
                    break

            # Save and log video
            gif_path = f"drlgs_experts/{session_name}/rollout_{self.num_timesteps}.gif"
            if not os.path.exists(os.path.dirname(gif_path)):
                os.makedirs(os.path.dirname(gif_path))
            base_rollout_env.save_gif(gif_path, s_h)


            # Cleanup
            rollout_env.close()

        return True

# 1. Register env
session_name = input("Enter your WandB entity name: ")
wandb.init(project="karel_stair_climber", name=session_name)
register_karel_env()

# 2. Create environment
env = gym.make("KarelStairClimber-v1")
env = Monitor(env)  # logs stats
env = DummyVecEnv([lambda: env])  # vectorized wrapper

# 3. Initialize model
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    policy='CnnPolicy',
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
)
# 4. Train
model.learn(
    total_timesteps=1_000_000,
    callback=CustomWandbVideoCallback(
        session_name=session_name,
        log_freq=1000,
        rollout_freq=5000,
        max_rollout_steps=50
    )
)


# 5. Save model
model.save("ppo_karel_stairclimber")
