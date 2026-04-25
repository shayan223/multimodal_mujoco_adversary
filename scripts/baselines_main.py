from itertools import count
import json
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
from ddiffpg.utils.common import update_wandb_summary
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

from ADVERSARIAL_CONFIGS import adversarial_cfg, build_adv_wandb_metadata

from diffusion import Diffusion_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path=ddiffpg.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    cfg = preprocess_cfg(cfg, if_ddiffpg=False)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    adv_cfg = adversarial_cfg()
    adv_wandb_metadata = build_adv_wandb_metadata(adv_cfg, cfg.seed)
    adv_wandb_metadata.update({
        "algo_name": cfg.algo.name,
        "env_name": cfg.env.name,
    })
    wandb_run = init_wandb(
        cfg,
        run_name_base=adv_wandb_metadata["run_name_base"],
        config_updates=adv_wandb_metadata,
    )
    wandb.config.update(adv_wandb_metadata, allow_val_change=True)
    update_wandb_summary(wandb_run, {
        "adv_preset": adv_wandb_metadata["adv_preset"],
        "attack_type": adv_wandb_metadata["attack_type"],
        "defense_type": adv_wandb_metadata["defense_type"],
        "target_modality": adv_wandb_metadata["target_modality"],
        "fgsm_magnitude": adv_wandb_metadata["fgsm_magnitude"],
        "enable_attack": adv_wandb_metadata["enable_attack"],
        "train_on_defense": adv_wandb_metadata["train_on_defense"],
        "algo_name": adv_wandb_metadata["algo_name"],
        "env_name": adv_wandb_metadata["env_name"],
    })
    generate_dataset = adv_cfg.GENERATE_DATASET
    defence_method = adv_cfg.DEF_METHOD
    train_on_defense = adv_cfg.TRAIN_ON_DEF
    defense_mode = adv_cfg.DEFENSE_MODE
    enable_defense_train = adv_cfg.ENABLE_DEFENSE_TRAIN
    enable_defense_eval = adv_cfg.ENABLE_DEFENSE_EVAL
    ddpm_experiment_name = adv_cfg.DDPM_EXPERIMENT_NAME
    ddpm_renoise_strength = adv_cfg.DDPM_RENOISE_STRENGTH
    ddpm_inference_steps = adv_cfg.DDPM_INFERENCE_STEPS
    target_modality = adv_cfg.TARGET_MODALITY
    data_prefix = adv_cfg.DATA_PREFIX
    max_steps_override = adv_cfg.MAX_STEPS_OVERRIDE
    enable_attack = adv_cfg.ENABLE_ATTACK
    save_path = adv_cfg.SAVE_PATH
    attack_choice = adv_cfg.ATTACK_CHOICE
    fgsm_eps = adv_cfg.FGSM_MAGNITUDE

    if(max_steps_override):
        cfg.max_step = max_steps_override

    if 'antmaze' in cfg.env.name:
        env = gym.make(cfg.env.name, reward_type=cfg.env.reward_type, random_init=cfg.env.random_init)

        episode_len = env._max_episode_steps
        env_kwargs = env.env.env.spec.kwargs
        cfg.env.env_kwargs = env_kwargs

        env = gym.vector.make(cfg.env.name, reward_type=cfg.env.reward_type, num_envs=cfg.num_envs, random_init=cfg.env.random_init)
        print('CURRENT ENV TYPE: ', type(env))
        def_model = None
        if defence_method == 'VAE_3d':
            def_model = VAE_3d()
            def_model.load_state_dict(torch.load(save_path+'defence_vae.pth', weights_only=True))
            def_model.eval()
            def_model = def_model.to(device)
        elif defence_method == 'VAE':
            def_model = VAE_simple()
            def_model.load_state_dict(torch.load(save_path+'defence_vae.pth', weights_only=True))
            def_model.eval()
            def_model = def_model.to(device)
        elif defence_method == 'DDPM':
            def_model = Diffusion_model(
                experiment_name=ddpm_experiment_name,
                inference_mode=defense_mode,
                inference_steps=ddpm_inference_steps,
                renoise_strength=ddpm_renoise_strength,
            )
            def_model.load_params()
            def_model.set_inference_config(
                mode=defense_mode,
                steps=ddpm_inference_steps,
                renoise_strength=ddpm_renoise_strength,
            )
        #Here we wrap the env to include our defense method in training
        if(defence_method is not None) and enable_defense_train and (train_on_defense == True):
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
            defence_func = None
            if(defence_method is not None) and enable_defense_eval:
                defence_func = defender(defence=defence_method, defence_model=def_model)
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
                    adv_obs = adversary(agent, obs, target_modality=target_modality,
                                        attack_choice=attack_choice, fgsm_eps=fgsm_eps)
                    
                elif(enable_attack == True):
                    #We can skip this line if attack is disabled otherwise
                    obs = adversary(agent, obs, target_modality=target_modality,
                                    attack_choice=attack_choice, fgsm_eps=fgsm_eps)

                #Repeat data collection steps on adversarial data
                if(generate_dataset == True):
                    #Hold onto the perturbed sample as well
                    #dataset_adv_buffer.append(obs.clone().detach().cpu().numpy())
                    buffer_obs = adv_obs.clone()#.detach().cpu().numpy()
                    buffer_list = [row.detach().cpu().numpy() for row in buffer_obs]
                    dataset_adv_buffer.extend(buffer_list)
                
                if defence_func is not None:
                    ######## Defence purification ##########
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
        "adv_preset": adv_wandb_metadata["adv_preset"],
        "attack_type": adv_wandb_metadata["attack_type"],
        "defense_type": adv_wandb_metadata["defense_type"],
        "target_modality": adv_wandb_metadata["target_modality"],
        "fgsm_magnitude": adv_wandb_metadata["fgsm_magnitude"],
        "enable_attack": adv_wandb_metadata["enable_attack"],
        "train_on_defense": adv_wandb_metadata["train_on_defense"],
        "algo_name": adv_wandb_metadata["algo_name"],
        "env_name": adv_wandb_metadata["env_name"],
        "run_name_base": adv_wandb_metadata["run_name_base"],
    })
    update_wandb_summary(wandb_run, summary_metrics)

    #Save the collected dataset of observations
    if(generate_dataset == True):
        print('##############')
        print('Saving Dataset! ')
        benign_dataset_df = pd.DataFrame(benign_dataset)
        benign_dataset_df.to_csv(save_path+data_prefix+'_benign_obs_data.csv')
        adv_dataset_df = pd.DataFrame(adv_dataset)
        adv_dataset_df.to_csv(save_path+data_prefix+'_adversarial_obs_data.csv')
        dataset_metadata = {
            "data_prefix": data_prefix,
            "attack_choice": attack_choice,
            "target_modality": target_modality,
            "fgsm_magnitude": fgsm_eps,
            "defense_method": defence_method,
            "train_on_defense": train_on_defense,
            "defense_mode": defense_mode,
            "enable_defense_train": enable_defense_train,
            "enable_defense_eval": enable_defense_eval,
            "num_benign_rows": len(benign_dataset),
            "num_adversarial_rows": len(adv_dataset),
            "save_path": save_path,
        }
        with open(save_path+data_prefix+'_metadata.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(dataset_metadata, metadata_file, indent=2)
        print('Dataset Saved!')
        print('###############')

def adversary(actor, obs, target_modality=None, attack_choice='FGSM', fgsm_eps=0.015):

    #print('ACTOR: ',actor)
    #print('OBSERVATION: ', obs.shape)
    #print('OUTPUT: ', actor(obs).shape)

    choice = attack_choice.upper()

    if choice == 'FGSM':
        obs = fgsm_attack(model=actor, input_vals=obs, eps=fgsm_eps, target_modality=target_modality)
    elif choice == 'ZEROOUT':
        obs = zero_out_single_feature(actor, obs, target_modality=target_modality)
    elif choice == 'RANDOMZEROOUT':
        obs = zero_out_random_features(actor, obs, target_modality=target_modality)
    elif choice == 'MODALITYZEROOUT':
        if target_modality not in ['velocity', 'angular']:
            raise ValueError("MODALITYZEROOUT requires TARGET_MODALITY to be 'velocity' or 'angular'")
        obs = zero_out_modality(actor, obs, modality=target_modality)
    else:
        raise ValueError(f"Unknown ATTACK_CHOICE '{attack_choice}'. "
                         "Valid options are 'FGSM', 'ZeroOut', 'RandomZeroOut', 'ModalityZeroOut'.")

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
        targeted_features[..., 13:19] += eps*input_vals.grad[..., 13:19].sign()
        perturbed_out = targeted_features
    elif(target_modality == 'angular'):
        targeted_features = input_vals.clone()
        targeted_features[..., 0:13] += eps*input_vals.grad[..., 0:13].sign()
        targeted_features[..., 19:] += eps*input_vals.grad[..., 19:].sign()
        perturbed_out = targeted_features
    else:
        perturbed_out = input_vals + eps*input_vals.grad.sign()

    
    return perturbed_out#.detach()


def zero_out_single_feature(model, input_vals, target_modality=None):
    perturbed = input_vals.clone()
    obs_dim = perturbed.shape[-1]

    if target_modality == 'velocity':
        feature_index = torch.randint(13, 19, (1,)).item()
    elif target_modality == 'angular':
        indices = []
        for i in range(13):
            indices.append(i)
        for i in range(19, obs_dim):
            indices.append(i)
        idx_tensor = torch.tensor(indices)
        rand_pos = torch.randint(0, idx_tensor.size(0), (1,)).item()
        feature_index = idx_tensor[rand_pos].item()
    else:
        feature_index = torch.randint(0, obs_dim, (1,)).item()

    perturbed[..., feature_index] = 0.0
    return perturbed


def zero_out_random_features(model, input_vals, target_modality=None):
    perturbed = input_vals.clone()
    obs_dim = perturbed.shape[-1]

    if target_modality == 'velocity':
        indices = []
        for i in range(13, 19):
            indices.append(i)
    elif target_modality == 'angular':
        indices = []
        for i in range(13):
            indices.append(i)
        for i in range(19, obs_dim):
            indices.append(i)
    else:
        indices = list(range(obs_dim))

    idx_tensor = torch.tensor(indices, device=perturbed.device)
    num_candidates = idx_tensor.size(0)
    num_features = torch.randint(1, num_candidates + 1, (1,)).item()

    perm = torch.randperm(num_candidates, device=perturbed.device)
    chosen = idx_tensor[perm[:num_features]]
    perturbed[..., chosen] = 0.0

    return perturbed


def zero_out_modality(model, input_vals, modality):
    perturbed = input_vals.clone()
    if modality == "velocity":
        # indices 13 through 18 are velocity vectors
        perturbed[..., 13:19] = 0.0
    elif modality == "angular":
        # indices 0 through 12 and 19+ are angular components
        perturbed[..., 0:13] = 0.0
        perturbed[..., 19:] = 0.0
    return perturbed


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
        return vae_defend

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

    raise ValueError(f"Unknown defense method: {defence}")


def validate_attack_targeting(
    original_obs: torch.Tensor,
    perturbed_obs: torch.Tensor,
    target_modality=None,
    attack_choice='FGSM',
):
    changed = (perturbed_obs - original_obs).abs() > 1e-8
    if changed.dim() > 1:
        changed_any = changed.any(dim=0)
    else:
        changed_any = changed

    changed_indices = torch.where(changed_any)[0].detach().cpu().tolist()

    if attack_choice.upper() != 'FGSM':
        return {
            "attack_choice": attack_choice,
            "target_modality": target_modality,
            "changed_indices": changed_indices,
        }

    if target_modality == 'velocity':
        expected = set(range(13, 19))
    elif target_modality == 'angular':
        expected = set(list(range(13)) + list(range(19, original_obs.shape[-1])))
    else:
        expected = set(range(original_obs.shape[-1]))

    actual = set(changed_indices)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual) if target_modality is not None else []

    return {
        "attack_choice": attack_choice,
        "target_modality": target_modality,
        "changed_indices": changed_indices,
        "unexpected_indices": unexpected,
        "missing_expected_indices": missing,
        "targeting_valid": len(unexpected) == 0,
    }





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
    main()
