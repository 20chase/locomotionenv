from gym.envs.registration import register


# ------------------------------------------------------------------------------
#  environments
# ------------------------------------------------------------------------------

register(
    id='Hexapod-2Dof-MoveForward-v1',
    entry_point='locomotionenv.envs:Hexapod2DofMoveForwardEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
)

register(
    id='Hexapod-3Dof-MoveForward-v1',
    entry_point='locomotionenv.envs:Hexapod3DofMoveForwardEnv',
    max_episode_steps=1000,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
)


