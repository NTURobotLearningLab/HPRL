# register_karel_env.py

from gymnasium.envs.registration import register

def register_karel_env():
    register(
        id='KarelStairClimber-v1',
        entry_point='drlgs_experts.karel_gym_wrapper:KarelGymWrapper',  # adjust to your actual module path
        max_episode_steps=50,
    )
