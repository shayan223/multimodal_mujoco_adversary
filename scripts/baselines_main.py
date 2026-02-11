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
from ddiffpg.utils.common import Tracker, RewardCurveTracker, preprocess_cfg
from ddiffpg.utils.plot_util import plot_traj

#from art.defences.preprocessor import PixelDefend 
from gymnasium.vector import VectorEnvWrapper
from gymnasium import ObservationWrapper

from defense_vae import VAE_3d, VAE_simple

from ADVERSARIAL_CONFIGS import adversarial_cfg

from diffusion import Diffusion_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path=ddiffpg.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    cfg = preprocess_cfg(cfg, if_ddiffpg=False)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    wandb_run = init_wandb(cfg)
    
    adv_cfg = adversarial_cfg()
    generate_dataset=adv_cfg.GENERATE_DATASET
    defence_method=adv_cfg.DEF_METHOD
    train_on_defense=adv_cfg.TRAIN_ON_DEF
    target_modality=adv_cfg.TARGET_MODALITY
    data_prefix=adv_cfg.DATA_PREFIX
    max_steps_override=adv_cfg.MAX_STEPS_OVERRIDE
    enable_attack=adv_cfg.ENABLE_ATTACK
    save_path=adv_cfg.SAVE_PATH

    if(max_steps_override):
        cfg.max_step = max_steps_override

    if 'antmaze' in cfg.env.name:
        env = gym.make(cfg.env.name, reward_type=cfg.env.reward_type, random_init=cfg.env.random_init)

        episode_len = env._max_episode_steps
        env_kwargs = env.env.env.spec.kwargs
        cfg.env.env_kwargs = env_kwargs

        env = gym.vector.make(cfg.env.name, reward_type=cfg.env.reward_type, num_envs=cfg.num_envs, random_init=cfg.env.random_init)
        print('CURRENT ENV TYPE: ', type(env))
        if(defence_method == 'VAE_3d'):
            def_model = VAE_3d()
            def_model.load_state_dict(torch.load(save_path+'defence_vae.pth', weights_only=True))
            def_model.eval()
            def_model = def_model.to(device)
        if(defence_method == 'VAE'):
            def_model = VAE_simple()
            def_model.load_state_dict(torch.load(save_path+'defence_vae.pth', weights_only=True))
            def_model.eval()
            def_model = def_model.to(device)
        if(defence_method == 'DDPM'):
            def_model = Diffusion_model(experiment_name='diffusion_defense_1')
            def_model.load_params()
            #def_model.eval()
            #def_model = def_model.to(device)
        else:
            def_model=None
        #Here we wrap the env to include our defense method in training
        if(defence_method is not None) and (train_on_defense == True):
            env = DefenceObsWrapper(env, episode_len, defence_method,defence_model=def_model)

        else:
            env = D4RLEnvWrapper(env, episode_len)

        eval_env = gym.vector.make(cfg.env.name, reward_type=cfg.env.reward_type, num_envs=cfg.eval_num_envs, random_init=cfg.env.random_init)
        #Here we wrap the env to include our defense method in training
        #if(defence_method is not None):
            #eval_env = DefenceObsWrapper(eval_env, episode_len,defence_method)
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
    rt_cfg = getattr(cfg, "reward_tracking", None) or {}
    reward_curve_tracker = RewardCurveTracker(
        min_reward_threshold=rt_cfg.get("min_reward_threshold"),
        drop_threshold_pct=rt_cfg.get("drop_threshold_pct", 0.10),
    )
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
                    buffer_obs = obs.clone()#.detach().cpu().numpy()
                    buffer_list = [row.detach().cpu().numpy() for row in buffer_obs]
                    dataset_buffer.extend(buffer_list)
                
                #we don't apply the adversarial perturbation when collecting data, as not to disrupt the agent
                if(generate_dataset == True):
                    #Seperate standard observation from the perturbed one for data collection
                    adv_obs = adversary(agent,obs,target_modality=target_modality)
                    
                elif(enable_attack == True):
                    #We can skip this line if attack is disabled otherwise
                    obs = adversary(agent,obs,target_modality=target_modality)

                #Repeat data collection steps on adversarial data
                if(generate_dataset == True):
                    #Hold onto the perturbed sample as well
                    #dataset_adv_buffer.append(obs.clone().detach().cpu().numpy())
                    buffer_obs = adv_obs.clone()#.detach().cpu().numpy()
                    buffer_list = [row.detach().cpu().numpy() for row in buffer_obs]
                    dataset_adv_buffer.extend(buffer_list)
                
                if(defence_method is not None):
                    ######## Defence purification ##########
                    defence_func = defender(defence=defence_method,defence_model=def_model)
                    obs = defence_func(obs)

                #########################################

                if cfg.algo.obs_norm:
                    action = actor(normalizer.normalize(obs))
                else:
                    action = actor(obs).detach()
                next_obs, reward, done, info = eval_env.step(action)
                current_returns += reward
                current_lengths += 1
                traj_states.append(obs[:, :2].detach().cpu().numpy())
                env_done_indices = torch.where(done)[0]
                return_tracker.update(current_returns[env_done_indices])
                step_tracker.update(current_lengths[env_done_indices])
                current_returns[env_done_indices] = 0
                current_lengths[env_done_indices] = 0
                obs = next_obs

            ret_mean = return_tracker.mean()

            if(generate_dataset == True):
                #If we only want to collect successfull attacks
                '''if(collect_only_success == True):
                    if(global_steps > 1.5e6 and ret_mean < 2):
                        print("Sucessfull attack, saving data!")
                        #extend dataset to include new values, rather than nesting lists
                        benign_dataset.extend(dataset_buffer)
                        adv_dataset.extend(dataset_adv_buffer)'''
                #If the episode reward is positive, the recorded observations are meaningfull enough for the dataset
                if(global_steps > 1.5e6):  #if(ret_mean > 0):
                    print("Sucessfull episode, saving data!")
                    #extend dataset to include new values, rather than nesting lists
                    benign_dataset.extend(dataset_buffer)
                    adv_dataset.extend(dataset_adv_buffer)

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

            reward_metrics = reward_curve_tracker.update(global_steps, ret_mean, ret_max)
            wandb.log({
                'eval/return': ret_mean,
                'eval/episode_length': step_mean,
                **reward_metrics,
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

    # Log run-level summary metrics for easy comparison across runs
    summary_metrics = reward_curve_tracker.summary_metrics()
    # Add adversarial configuration info for tabular comparison
    summary_metrics.update({
        "summary/FGSM_MAGNITUDE": adv_cfg.FGSM_MAGNITUDE,
        "summary/DEF_METHOD": defence_method,
        "summary/TARGET_MODALITY": "both" if target_modality is None else target_modality,
    })
    wandb.log(summary_metrics)

    # Log summary as a table so it appears in the run's Charts section
    summary_columns = list(summary_metrics.keys())
    summary_row = [summary_metrics[k] for k in summary_columns]
    summary_table = wandb.Table(columns=summary_columns, data=[summary_row])
    wandb.log({"run_summary": summary_table})

    #Save the collected dataset of observations
    if(generate_dataset == True):
        print('##############')
        print('Saving Dataset! ')
        benign_dataset_df = pd.DataFrame(benign_dataset)
        benign_dataset_df.to_csv(save_path+data_prefix+'_benign_obs_data.csv')
        adv_dataset_df = pd.DataFrame(adv_dataset)
        adv_dataset_df.to_csv(save_path+data_prefix+'_adversarial_obs_data.csv')
        print('Dataset Saved!')
        print('###############')

def adversary(actor, obs, target_modality=None):

    #print('ACTOR: ',actor)
    #print('OBSERVATION: ', obs.shape)
    #print('OUTPUT: ', actor(obs).shape)

    obs = fgsm_attack(model=actor, input_vals=obs, eps=0.007, target_modality=target_modality)

    return obs



def fgsm_attack(model, input_vals, eps=0.015, target_modality=None,outputs=None) :
    
    input_vals.requires_grad = True
            
    if(outputs is None):
        outputs = model.actor(input_vals)
    
    model.actor.zero_grad()
    actor_loss = model.update_actor(input_vals,skip_weight_update=True)

    #modify just one modality (13 through 18 are velocity vectors)
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

    
    return perturbed_out#.detach()


def defender(defence=None, defence_model=None):

    if(defence == None):
        def no_defend(obs):
            return obs
        return no_defend

    if(defence == 'Gaussian'):
        def gaussian_defend(obs):
            scaling_factor = 0.005
            if isinstance(obs, np.ndarray):
                obs = obs + (np.random.normal(size=obs.shape) * scaling_factor)
            else:
                obs = obs + (torch.randn_like(obs) * scaling_factor)
            return obs
        return gaussian_defend
    
    if(defence == 'VAE_3d'):
        def vae_defend(obs):
            #Reshape to square vector
            batch_size = obs.size(0)
            N = obs.size(1)
            obs = torch.nn.functional.pad(obs, (0, 36 - N))
            obs = obs.view(batch_size, 1, 6, 6)#.squeeze(0)

            obs, _, _= defence_model(obs)
            #return vector to its original dims
            #N = 29
            #obs = obs.view(-1) #flatten to (36,)
            #obs = obs[:N] #remove the padding
            obs = obs.view(batch_size, -1) #flatten to (batch_size, 36)
            obs = obs[:, :N] #remove the padding

            return obs
        
    if(defence == 'VAE'):
        def vae_defend(obs):
            obs = torch.Tensor(obs).to(device)
            obs, _, _= defence_model(obs)
            #obs = obs.cpu().detach().numpy()

            return obs
        return vae_defend
    
    if(defence == 'DDPM'):
        def ddpm_defend(obs):
            obs = torch.Tensor(obs).to(device)
            obs = defence_model.inference(obs)
            return obs
        return ddpm_defend





class DefenceObsWrapper:
    def __init__(self, env, episode_len, defence_method, defence_model=None):
        self.env = env
        self.observation_space = np.zeros(self.env.observation_space.shape[1])
        self.action_space = np.zeros(self.env.action_space.shape[1])
        self.max_episode_length = episode_len
        self.device = torch.device("cuda:0")
        self.defence_method = defence_method
        self.defence_func = defender(defence_method,defence_model=defence_model)


    def reset(self):
        ob = self.env.reset()
        ob = self.defence_func(ob)
        return self.cast(ob)

    def step(self, actions):
        actions = actions.cpu().numpy()
        next_obs, rewards, dones, infos = self.env.step(actions)
        next_obs = self.defence_func(next_obs)
        timeout = torch.zeros(dones.shape).bool().to(self.device)
        success = torch.zeros(dones.shape).to(self.device)
        for i in range(len(infos)):
            if len(infos[i]) == 0:
                pass
            else:
                if "TimeLimit.truncated" in infos[i].keys():
                    timeout[i] = infos[i]["TimeLimit.truncated"]
                if "success" in infos[i].keys():
                    success[i] = infos[i]["success"]
        info_ret = {"time_outs": timeout, "success": success}

        return (
            self.cast(next_obs),
            self.cast(rewards),
            self.cast(dones).long(),
            info_ret,
        )

    def cast(self, x):
        x = torch.Tensor(x).to(self.device)
        return x


if __name__ == '__main__':

    adv_cfg = adversarial_cfg()

    main()
