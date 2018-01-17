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

register(
    id='AntObstaclesBig-v1',
    entry_point='locomotionenv.envs:AntObstaclesBigEnv',
    max_episode_steps=3000,
)

register(
    id='AntBandits-v1',
    entry_point='locomotionenv.envs:AntBanditsEnv',
    max_episode_steps=1000,
)

register(
    id='AntMovement-v1',
    entry_point='locomotionenv.envs:AntMovementEnv',
    max_episode_steps=600,
)

register(
    id='AntObstacles-v1',
    entry_point='locomotionenv.envs:AntObstaclesEnv',
    max_episode_steps=1000,
)


register(
    id='AntObstaclesGen-v1',
    entry_point='locomotionenv.envs:AntObstaclesGenEnv',
    max_episode_steps=1000,
)

register(
    id='Hexapod-2Dof-MoveOverObstalces-Stairs-v1',
    entry_point='locomotionenv.envs:Hexapod2DofMoveOverObstalcesStairsEnv',
    max_episode_steps=4000,
)

