import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from ddiffpg.models.mlp import RNDModel
from ddiffpg.utils.torch_util import RunningMeanStd


class IntrinsicM:
    def __init__(self, obs_dim, type='noveld', env_name=None, normalize=True, pos_enc=True, L=10, warm_up=1000, device='cuda'):
        self.obs_dim = obs_dim
        self.type = type
        self.env_name = env_name
        self.normalize = normalize
        self.device = device
        self.pos_enc = pos_enc
        self.update_step = 0
        self.warm_up = warm_up
        self.L = L
        if self.pos_enc:
            if 'antmaze' in self.env_name: # encode ant 2d pos
                input_dims = 2
                dims = 2
            else:  # encode end effector 3d pos
                input_dims = 2
                dims = 3
            self.embedder, dim = get_embedder(self.L, input_dims=input_dims)
            self.rnd_model = RNDModel(self.obs_dim[0] + dims * 2 * L).to(self.device)
        else:
            self.rnd_model = RNDModel(self.obs_dim).to(self.device)
        self.rnd_optimizer = torch.optim.AdamW(self.rnd_model.parameters(), 1e-4)
        self.rnd_rms = RunningMeanStd(shape=(1), device=self.device)

    def compute_reward(self, obs, next_obs=None):
        if self.pos_enc:
            obs = self.encode_obs(obs)
            if next_obs is not None:
                next_obs = self.encode_obs(next_obs)

        if self.type == 'rnd':
            novelty_obs = self.get_novelty(obs)
            if self.normalize and self.update_step > self.warm_up:
                self.rnd_rms.update(novelty_obs)
                novelty_obs = self.rnd_rms.normalize(novelty_obs)
            reward_intrinsic = novelty_obs.unsqueeze(1)
            return reward_intrinsic
        
        elif self.type == 'noveld':
            assert next_obs is not None
            novelty_obs = self.get_novelty(obs)
            novelty_nextobs = self.get_novelty(next_obs)
           
            if self.normalize and self.update_step > self.warm_up:
                self.rnd_rms.update(novelty_obs)
                self.rnd_rms.update(novelty_nextobs)
                novelty_obs = self.rnd_rms.normalize(novelty_obs)
                novelty_nextobs = self.rnd_rms.normalize(novelty_nextobs)

            intrinsic = novelty_nextobs - 0.5 * novelty_obs
            reward_intrinsic = 0.01 * torch.max(intrinsic, torch.zeros(intrinsic.shape, device=intrinsic.device)).unsqueeze(1)
            return reward_intrinsic
        
        else:
            raise NotImplementedError

    def get_novelty(self, obs):
        predict_obs, target_obs = self.rnd_model(obs)
        novelty = torch.norm(predict_obs - target_obs, dim=1, p=2).detach()
        return novelty
    
    def update(self, obs):
        if self.pos_enc:
            obs = self.encode_obs(obs)

        predict_feature, target_feature = self.rnd_model(obs)
        dynamic_loss = F.mse_loss(predict_feature, target_feature.detach())
        dynamic_grad_norm = self.optimizer_update(self.rnd_optimizer, dynamic_loss)
        self.update_step += 1
        return dynamic_loss.item(), dynamic_grad_norm.item()

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                    max_norm=1.0)
        optimizer.step()
        return grad_norm
    
    def encode_obs(self, obs):
        if 'antmaze' in self.env_name: # encode ant 2d pos
            pos = self.embedder(obs[:, :2])
            return torch.cat([pos, obs[:, 2:]], dim=1)
        else:  # encode end effector 3d pos
            pos = self.embedder(obs[:, :3])
            return torch.cat([pos, obs[:, 3:]], dim=1)


def positional_encoding(x, L=10):
    """
    Applies positional encoding to the input coordinates.
    
    Args:
    - x: A tensor of shape [N, 3] where N is the number of points, and each point has 3D coordinates (x, y, z).
    - L: The number of frequency bands used for encoding.

    Returns:
    - A tensor of shape [N, 3 * 2 * L] containing the positional encodings for each coordinate.
    """
    # Frequencies
    frequencies = 2.0 ** torch.linspace(0, L-1, L).to(x.device)
    
    # Reshape x for broadcasting: [N, 3, 1]
    x = x.unsqueeze(-1)
    
    # Encode each dimension
    encoded = torch.cat([torch.sin(x * frequencies), torch.cos(x * frequencies)], dim=-1)
    
    # Flatten the encodings
    encoded = encoded.view(x.shape[0], -1)
    
    return encoded


class Embedder:
    # https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    def __init__(self, **kwargs):
        import torch
        self.kwargs = kwargs

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        import torch
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, **kwargs):
    if i == -1:
        from torch import nn
        return nn.Identity(), 2
    import torch
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    embed_kwargs.update(kwargs)
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
