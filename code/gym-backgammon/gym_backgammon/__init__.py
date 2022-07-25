from gym.envs.registration import register

register(
    id='backgammon',
    entry_point='gym_backgammon.envs:BackgammonEnv',
    max_episode_steps=None,
)