import os
from pathlib import Path
LIB_PATH = Path(__file__).resolve().parent


from gym.envs.registration import register
from gymnasium.envs.registration import register as gymnasium_register
from ddiffpg.env.d4rl.locomotion import maze_env

"""
register(
    id='antmaze-umaze-v0',
    entry_point='ddiffpg.env.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)
"""

register(
    id='antmaze-v1',
    entry_point='ddiffpg.env.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=500,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.MAZE_v1,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'random_init': False,
    }
)

register(
    id='antmaze-v2',
    entry_point='ddiffpg.env.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=500,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.MAZE_v2,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'random_init': False,
    }
)

register(
    id='antmaze-v3',
    entry_point='ddiffpg.env.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.MAZE_v3,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'random_init': False,
    }
)

register(
    id='antmaze-v4',
    entry_point='ddiffpg.env.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.MAZE_v4,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
        'non_zero_reset':False,
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'random_init': False,
    }
)


ENV_IDS = []

for task in ["Reach", "PegInsertion", "DrawerMulti", "Cabinet"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:

            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"Panda{task}{control_suffix}{reward_suffix}-v3"

            gymnasium_register(
                id=env_id,
                entry_point=f"ddiffpg.env.panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100,
            )

            # adding randomize starting point to Reach and PegInsertion task
            if task in ["Reach", "PegInsertion"]:
                random_suffix = 'Random'
                reward_suffix = "Dense" if reward_type == "dense" else ""
                control_suffix = "Joints" if control_type == "joints" else ""
                env_id = f"Panda{task}{control_suffix}{reward_suffix}{random_suffix}-v3"

                gymnasium_register(
                    id=env_id,
                    entry_point=f"panda_gym.envs:Panda{task}Env",
                    kwargs={"reward_type": reward_type, "control_type": control_type, "random_init_pos": True},
                    max_episode_steps=100,
                )

            ENV_IDS.append(env_id)