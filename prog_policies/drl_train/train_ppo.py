"""
Train PPO on StairClimberSparse.
Run with:
    python train_ppo.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from prog_policies.drl_train import KarelGymEnv
from prog_policies.drl_train import KarelCNN



from prog_policies.karel import KarelEnvironment
from prog_policies.drl_train import save_gif
from prog_policies.drl_train import GifEvalCallback
from prog_policies.karel_tasks import get_task_cls

import numpy as np
MASTER_SEED = 12345                       # ≤ change once per experiment
rng = np.random.RandomState(MASTER_SEED)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Harvester", help="Task name for the Karel environment")
args = parser.parse_args()

task_name = args.task

# task_name = "StairClimber"
# task_name = "Maze"
# task_name = "DoorKey"
# task_name = "Harvester"


env_args = {
    "env_height": 8,
    "env_width": 8,
    "crashable": True,
    "leaps_behaviour": False,
    "max_calls": 10000,
}

def make_env(seed: int = 0):
    # Gymnasium’s register() not strictly needed, we just return the wrapper.
    def _thunk():
        env = KarelGymEnv(
            task_name=task_name,
            env_args=env_args,
            seed=seed
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # For PyTorch CNNs SB3 expects CHW; our wrapper already gives CHW,
        # so only wrap with VecTransposeImage if you ever switch to HWC order.
        return env
    return _thunk


if __name__ == "__main__":
    # Vectorised env so PPO can use multiple workers (here 8)
    n_envs = 20
    eval_seed = rng.randint(0, 2**31 - 1)
    env = DummyVecEnv([make_env(s) for s in range(n_envs)])
    # ──────────────────────────────────────────────────────────────
    # Quick single-env sanity check – why are episodes length 1?
    # ──────────────────────────────────────────────────────────────
    print("\n=== Sanity-check a fresh environment =====================")
    dbg_env = make_env(seed=eval_seed)()       # plain Gym env, no vector wrapper
    for ep in range(3):
        obs, _ = dbg_env.reset()
        done = False
        t = 0
        print(f"\nEpisode {ep} starts.")
        while not done and t < 100:             # cap at 10 steps for readability
            action = dbg_env.action_space.sample()
            obs, rew, term, trunc, info = dbg_env.step(action)
            print(f"  t={t:2d} | a={action:2d} | r={rew:5.2f} | term={term} | trunc={trunc}")
            done = term or trunc
            t += 1
        print(f"Episode {ep} finished after {t} steps "
            f"({'terminated' if term else 'truncated'})")
    # ──────────────────────────────────────────────────────────────


    # Optional evaluation env (no randomness makes learning curves cleaner)
    
    def generate_eval_env(seed):
        eval_env =  KarelGymEnv(
                    task_name=task_name,
                    env_args=env_args,
                    seed=seed
                )
        return eval_env
    # eval_callback = EvalCallback(
    #     eval_env,
    #     eval_freq=10_000,
    #     n_eval_episodes=20,
    #     callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1),
    #     verbose=0,
    # )
    eval_callback = GifEvalCallback(
        eval_env=generate_eval_env(rng.randint(0, 2**31 - 1)),
        eval_freq=10_000,
        n_eval_episodes=20,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=0.9, verbose=1
        ),
        save_dir=f"prog_policies/drl_train/gifs/{task_name}",          # directory for .gif files
        verbose=0,
    )

    policy_kwargs = dict(
        features_extractor_class=KarelCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=128,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=None,
        device="auto",
    )

    model.learn(total_timesteps=10_000_000, callback=eval_callback)
    model.save(f"/home/hubertchang/HPRL/expert_ckpt/ppo_karel_{task_name}")
