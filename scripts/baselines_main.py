from itertools import count
import hydra
import wandb
import gym
import gymnasium
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig

import ddiffpg
from ddiffpg.algo import alg_name_to_path
from ddiffpg.utils.common import init_wandb
from ddiffpg.replay.simple_replay import ReplayBuffer
from ddiffpg.utils.common import load_class_from_path
from ddiffpg.utils.common import set_random_seed
from ddiffpg.utils.common import capture_keyboard_interrupt
from ddiffpg.utils.model_util import load_model
from ddiffpg.utils.model_util import save_model
from ddiffpg.wrappers.d4rl_wrapper import D4RLEnvWrapper
from ddiffpg.wrappers.pybullet_wrapper import PybulletEnvWrapper
from ddiffpg.utils.common import Tracker, preprocess_cfg
from ddiffpg.utils.plot_util import plot_traj

from art.defences.preprocessor import PixelDefend 

    
@hydra.main(config_path=ddiffpg.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig, generate_dataset=True):
    cfg = preprocess_cfg(cfg, if_ddiffpg=False)
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
    is_off_policy = cfg.algo.name != 'PPO'
    if is_off_policy:
        memory = ReplayBuffer(capacity=int(cfg.algo.memory_size),
                              obs_dim=agent.obs_dim,
                              action_dim=agent.action_dim,
                              device=cfg.device)
        trajectory, steps = agent.explore_env(env, cfg.algo.warm_up, random=True)
        if trajectory is not None:
            memory.add_to_buffer(trajectory)

    if(generate_dataset == True):
        benign_dataset = []
        adv_dataset = []

    for iter_t in count():
        if iter_t % cfg.eval_freq == 0:

            #For Dataset Collection
            dataset_buffer = [] #holds onto observations until we know they are worth keeping
            dataset_adv_buffer = [] #holds onto adversarial observations in parrallel with dataset_buffer

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
            for i_step in range(max_step):  # run an episode

                ######## Adversarial injection ##########

                if(generate_dataset == True):
                    #hold on to the benign observation for the dataset
                    dataset_buffer.append(obs.clone().detach().cpu().numpy())
                
                #we don't apply the adversarial perturbation when collecting data, as not to disrupt the agent
                else:
                    obs = adversary(agent,obs)

                if(generate_dataset == True):
                    #Hold onto the perturbed sample as well
                    dataset_adv_buffer.append(obs.clone().detach().cpu().numpy())
                else:
                    ######## Defence purification ##########
                    obs = defender(obs,defence='Gaussian')

                #########################################

                if cfg.algo.obs_norm:
                    action = actor(normalizer.normalize(obs))
                else:
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

            #If the episode reward is positive, the recorded observations are meaningfull enough for the dataset
            if(generate_dataset == True):
                if(ret_mean > 0):
                    print("Sucessfull episode, saving data!")
                    benign_dataset.append(dataset_buffer)
                    adv_dataset.append(dataset_adv_buffer)

            step_mean = step_tracker.mean()
            if ret_mean >= ret_max:
                ret_max = ret_mean
            
            if 'antmaze' in cfg.env.name:
                img = plot_traj(env_kwargs, np.concatenate(traj_states, axis=0))
                img = wandb.Image(img)
                wandb.log({'eval/map': img})

            if iter_t % (cfg.eval_freq * 5) == 0:
                if 'antmaze' in cfg.env.name:
                    explore_img = agent.pos_history.plot_heatmap()
                    explore_img = wandb.Image(explore_img)
                    wandb.log({'exploration_map': explore_img})
                
                save_model(path=f"{wandb_run.dir}/model.pth",
                       actor=actor.state_dict(),
                       critic=agent.critic.state_dict(),
                       rms=normalizer.get_states() if cfg.algo.obs_norm else None,
                       wandb_run=wandb_run,
                       ret_max=f'{ret_mean}',
                       embedding=None,
                       coverage=agent.pos_history.mat if 'antmaze' in cfg.env.name else None,
                       )

            wandb.log({'eval/return': ret_mean,
                       'eval/episode_length': step_mean,
                       })
        
        trajectory, steps = agent.explore_env(env, cfg.algo.horizon_len, random=False)
        global_steps += steps
        
        if is_off_policy:
            if trajectory is not None:
                memory.add_to_buffer(trajectory)
            log_info = agent.update_net(memory)
        else:
            log_info = agent.update_net(trajectory)

        if iter_t % cfg.log_freq == 0:
            log_info['global_steps'] = global_steps
            wandb.log(log_info, step=global_steps)

        if global_steps > cfg.max_step:
            break

    #Save the collected dataset of observations
    if(generate_dataset == True):
        print('##############')
        print('Saving Dataset! ')
        benign_dataset_df = pd.DataFrame(benign_dataset)
        benign_dataset_df.to_csv('./benign_obs_data.csv')
        adv_dataset_df = pd.DataFrame(adv_dataset)
        adv_dataset_df.to_csv('./adversarial_obs_data.csv')
        print('Dataset Saved!')
        print('###############')

def adversary(actor, obs):

    #print('ACTOR: ',actor)
    #print('OBSERVATION: ', obs.shape)
    #print('OUTPUT: ', actor(obs).shape)

    obs = fgsm_attack(model=actor, input_vals=obs, eps=0.007, target_modality='angular')

    return obs



def fgsm_attack(model, input_vals, eps=0.007, target_modality=None) :
    
    #input_vals = input_vals.to(device)
    #labels = labels.to(device)
    input_vals.requires_grad = True
            
    outputs = model.actor(input_vals)
    
    model.actor.zero_grad()
    #cost = loss(outputs, labels).to(device)
    actor_loss = model.update_actor(input_vals,skip_weight_update=True)
    #cost.backward()
    #print(input_vals)
    #modify just one modality (13 through 18 are velocity vectors)
    #print(input_vals.shape)
    #print(input_vals[13:19].shape)
    if(target_modality == 'velocity'):
        targeted_features = input_vals.clone()
        targeted_features[13:19] += eps*input_vals.grad[13:19].sign()
        perturbed_out = targeted_features
    if(target_modality == 'angular'):
        targeted_features = input_vals.clone()
        targeted_features[0:13] += eps*input_vals.grad[0:13].sign()
        targeted_features[19:] += eps*input_vals.grad[19:].sign()
        perturbed_out = targeted_features
    else:
        perturbed_out = input_vals + eps*input_vals.grad.sign()
    #perturbed_out = torch.clamp(perturbed_out, -1, 1)
    #print(perturbed_out)
    
    return perturbed_out.detach()


def defender(adv_input, defence=None):

    if(defence == 'PixelDefend'):
        defence_method = PixelDefend(clip_values=(-1000,1000))
        purified_input = defence_method(adv_input)
    elif(defence == 'Gaussian'):
        scaling_factor = 0.005
        purified_input = adv_input + (torch.randn_like(adv_input) * scaling_factor)
    else:
        purified_input = adv_input

    
    return purified_input


if __name__ == '__main__':
    main()
