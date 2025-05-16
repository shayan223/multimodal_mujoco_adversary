from itertools import count
import hydra
import wandb
import gymnasium
import gym
import numpy as np
import torch
from omegaconf import DictConfig

import ddiffpg
from ddiffpg.algo import alg_name_to_path
from ddiffpg.utils.common import init_wandb
from ddiffpg.utils.common import load_class_from_path
from ddiffpg.utils.common import set_random_seed
from ddiffpg.utils.common import capture_keyboard_interrupt
from ddiffpg.utils.model_util import load_model
from ddiffpg.utils.model_util import save_model
from ddiffpg.wrappers.d4rl_wrapper import D4RLEnvWrapper
from ddiffpg.wrappers.pybullet_wrapper import PybulletEnvWrapper
from ddiffpg.utils.common import Tracker, preprocess_cfg
from ddiffpg.utils.plot_util import plot_cluster, plot_traj, plot_hierarchy
from ddiffpg.utils.torch_util import add_embedding

    
@hydra.main(config_path=ddiffpg.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    cfg = preprocess_cfg(cfg)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    wandb_run = init_wandb(cfg)
    
    if 'antmaze' in cfg.env.name:
        env = gym.make(cfg.env.name, reward_type=cfg.env.reward_type, random_init=cfg.env.random_init)
        episode_len = env._max_episode_steps
        env_kwargs = env.env.env.spec.kwargs
        cfg.env.env_kwargs = env_kwargs
        env = gym.vector.make(cfg.env.name, reward_type=cfg.env.reward_type, num_envs=cfg.num_envs, random_init=cfg.env.random_init)
        env = D4RLEnvWrapper(env, episode_len)
        eval_env = gym.vector.make(cfg.env.name, reward_type=cfg.env.reward_type, num_envs=cfg.eval_num_envs, random_init=cfg.env.random_init)
        eval_env = D4RLEnvWrapper(eval_env, episode_len)
    else:
        env = gymnasium.vector.make(cfg.env.name, control_type='joints', num_envs=cfg.num_envs)
        env = PybulletEnvWrapper(env)
        eval_env = gymnasium.vector.make(cfg.env.name, control_type='joints', num_envs=cfg.eval_num_envs)
        eval_env = PybulletEnvWrapper(eval_env)

    algo_name = cfg.algo.name
    if 'Agent' not in algo_name:
        algo_name = 'Agent' + algo_name
    agent_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    agent = agent_class(env=env, cfg=cfg)

    if cfg.artifact is not None:
        load_model(agent.actor, "actor", cfg)
        load_model(agent.critic, "critic", cfg)
        if cfg.algo.obs_norm:
            load_model(agent.obs_rms, "obs_rms", cfg)

    global_steps = 0
    agent.reset_agent()
    ret_max = float('-inf')

    steps = agent.explore_env(env, cfg.algo.warm_up, random=True)
    agent.diffusion_buffer.update_cluster()
    agent.explore_embedding = agent.diffusion_buffer.embeddings[0]

    for iter_t in count():
        # evaluation
        if iter_t % cfg.eval_freq == 0:
            # update cluster
            agent.diffusion_buffer.update_cluster() 

            num_envs = cfg.eval_num_envs
            max_step = eval_env.max_episode_length
            actor = agent.actor
            normalizer = agent.obs_rms
            return_tracker = Tracker(num_envs)
            step_tracker = Tracker(num_envs)
            current_returns = torch.zeros(num_envs, dtype=torch.float32, device=cfg.device)
            current_lengths = torch.zeros(num_envs, dtype=torch.float32, device=cfg.device)
            traj_states = []
            obs = eval_env.reset()
            for _ in range(max_step):  # run an episode
                if cfg.algo.obs_norm:
                    action = actor(normalizer.normalize(obs))
                else:
                    # add random embedding [zeros]
                    obs = add_embedding(obs, torch.zeros(cfg.algo.embedding_dim, device=obs.device))
                    action = actor(obs).detach()
                next_obs, reward, done, info = eval_env.step(action)
                current_returns += reward
                current_lengths += 1
                traj_states.append(obs[:, :2].cpu().numpy())
                env_done_indices = torch.where(done)[0]
                return_tracker.update(current_returns[env_done_indices])
                step_tracker.update(current_lengths[env_done_indices])
                current_returns[env_done_indices] = 0
                current_lengths[env_done_indices] = 0
                obs = next_obs

            ret_mean = return_tracker.mean()
            step_mean = step_tracker.mean()
            if ret_mean >= ret_max:
                ret_max = ret_mean

            if 'antmaze' in cfg.env.name and len(traj_states) != 0:
                img = plot_traj(env_kwargs, np.concatenate(traj_states, axis=0))
                img = wandb.Image(img)
                wandb.log({'eval/map': img})

            if iter_t % (cfg.eval_freq * 5) == 0:
                if agent.diffusion_buffer.plot is not None:
                    traj_list, clusters, Z = agent.diffusion_buffer.plot
                    if 'antmaze' in cfg.env.name:
                        cluster_img = plot_cluster(env_kwargs, traj_list, clusters)
                        cluster_img = wandb.Image(cluster_img)
                        wandb.log({'cluster': cluster_img})
                    if Z is not None:
                        print("Hierarchy threshold:", 0.7*max(Z[:,2]))
                        hierarchy_img = plot_hierarchy(Z)
                        hierarchy_img = wandb.Image(hierarchy_img)
                        wandb.log({'hierarchy': hierarchy_img})

                if 'antmaze' in cfg.env.name:
                    if agent.diffusion_buffer.unsuccess_plot is not None:
                        traj_list, clusters = agent.diffusion_buffer.unsuccess_plot
                        cluster_img = plot_cluster(env_kwargs, traj_list, clusters)
                        cluster_img = wandb.Image(cluster_img)
                        wandb.log({'unsuccess_cluster': cluster_img})

                    explore_img = agent.pos_history.plot_heatmap()
                    explore_img = wandb.Image(explore_img)
                    wandb.log({'exploration_map': explore_img})
                
                Q_dicts = []
                for Q in agent.diffusion_buffer.Qs:
                    Q_dicts.append(Q["Q"].state_dict())
                save_model(path=f"{wandb_run.dir}/model.pth",
                       actor=actor.state_dict(),
                       critic=Q_dicts,
                       rms=normalizer.get_states() if cfg.algo.obs_norm else None,
                       wandb_run=wandb_run,
                       ret_max=f'{ret_mean}',
                       embedding=agent.diffusion_buffer.embeddings,
                       coverage=agent.pos_history.mat if 'antmaze' in cfg.env.name else None,
                       )

            wandb.log({'eval/return': ret_mean,
                        'eval/episode_length': step_mean,
                       })
        
        steps = agent.explore_env(env, cfg.algo.horizon_len, random=False, total_steps=global_steps)
        global_steps += steps
        log_info = agent.update_net()

        if iter_t % cfg.log_freq == 0:
            log_info['global_steps'] = global_steps
            wandb.log(log_info, step=global_steps)

        if global_steps > cfg.max_step:
            break


if __name__ == '__main__':
    main()
