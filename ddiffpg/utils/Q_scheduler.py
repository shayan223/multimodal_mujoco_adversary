import torch
import numpy as np
from copy import deepcopy
from ddiffpg.utils.common import load_class_from_path
from ddiffpg.models import model_name_to_path
from ddiffpg.utils.torch_util import generate_embedding


class Q_scheduler:
    def __init__(self, cfg, obs_dim, action_dim):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if self.cfg.algo.cri_class == "DistributionalDoubleQ":
            Q = self.cri_class(self.obs_dim, self.action_dim, 
                                    v_min=self.cfg.algo.v_min, 
                                    v_max=self.cfg.algo.v_max,
                                    num_atoms=self.cfg.algo.num_atoms,
                                    device=self.cfg.device).to(self.cfg.device)
        optimizer = torch.optim.AdamW(Q.parameters(), self.cfg.algo.critic_lr)
        self.explore_Q = {"Q": Q, "target_Q": deepcopy(Q), "optimizer": optimizer}

        self.device = self.cfg.device
        self.Qs = []
        self.last_cluster = []
        self.explore_embedding = generate_embedding(self.cfg.algo.embedding_dim)
        self.embeddings = {0: self.explore_embedding}

    def update_cluster(self, cluster):
        # For target_actions: -1 original actions, 0 explor actions, otherwise previous group
        indices = []
        new_embeddings = {0: self.explore_embedding}
        # if the current cluster is still empty, do nothing
        if len(cluster) == 0:
            pass
        # if this is first time cluster has something, create new Qs for each cluster
        elif len(self.last_cluster) == 0:
            for i in range(len(cluster)):
                Q = deepcopy(self.explore_Q["Q"])
                optimizer = torch.optim.AdamW(Q.parameters(), self.cfg.algo.critic_lr)
                self.Qs.append({"Q": Q, "target_Q": deepcopy(self.explore_Q["target_Q"]), "optimizer": optimizer})
                indices.append(0)
                new_embeddings[i+1] = generate_embedding(self.cfg.algo.embedding_dim)

        # if there are already clusters
        else:
            # find the most similar cluster from previous cluster
            new_Qs = []
            indices = []
            overlaps = []
            for i in range(len(cluster)):
                num_overlap = 0
                idx = None
                for j in range(len(self.last_cluster)):
                    cur_overlap = len(set(cluster[i]) & set(self.last_cluster[j]))
                    if cur_overlap > num_overlap:
                        num_overlap = cur_overlap
                        idx = j
                 # if there is no overlap at all assign the explore_Q
                if idx is None:
                    Q = deepcopy(self.explore_Q["Q"])
                    optimizer = torch.optim.AdamW(Q.parameters(), self.cfg.algo.critic_lr)
                    new_Qs.append({"Q": Q, "target_Q": deepcopy(self.explore_Q["target_Q"]), "optimizer": optimizer})
                    indices.append(0)
                    new_embeddings[i+1] = generate_embedding(self.cfg.algo.embedding_dim)                
                # if the Q is already assigned to previous cluster, create a new copy
                elif idx+1 in indices:
                    Q = deepcopy(self.Qs[idx]["Q"])
                    optimizer = torch.optim.AdamW(Q.parameters(), self.cfg.algo.critic_lr)
                    new_Qs.append({"Q": Q, "target_Q": deepcopy(self.Qs[idx]["target_Q"]), "optimizer": optimizer})
                    # different from indices and Q functions, each cluster requires unique embeddings
                    max_overlap = 0
                    max_idx = None
                    for k in range(len(indices)):
                        if indices[k] == idx+1:
                            if max_overlap < overlaps[k]:
                                max_overlap = overlaps[k]
                                max_idx = k
                    assert torch.equal(new_embeddings[max_idx+1], self.embeddings[idx+1])
                    if num_overlap > max_overlap:
                        new_embeddings[i+1] = self.embeddings[idx+1]
                        new_embeddings[max_idx+1] = generate_embedding(self.cfg.algo.embedding_dim)
                    else:
                        new_embeddings[i+1] = generate_embedding(self.cfg.algo.embedding_dim)

                    indices.append(idx+1)
                # if the Q is not assigned yet
                else:
                    new_Qs.append(self.Qs[idx])
                    indices.append(idx+1)
                    new_embeddings[i+1] = self.embeddings[idx+1]
                overlaps.append(num_overlap)
            self.Qs = new_Qs
        
        self.last_cluster = cluster
        self.embeddings = new_embeddings
        #print("Qs len, cluster len:", len(self.Qs), len(cluster), len(indices))
        assert len(self.Qs) == len(cluster)
        assert len(indices) == len(cluster)
        assert len(self.embeddings) == len(cluster) + 1
        return self.explore_Q, self.Qs, indices, self.embeddings
    
    def update_Qs(self, explore_Q, Qs):
        self.explore_Q = explore_Q
        self.Qs = Qs
