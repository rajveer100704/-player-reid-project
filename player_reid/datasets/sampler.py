import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
import random

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, each with K instances.
    Total batch size = N * K
    Modified for 'Hard Teammate Mining' by allowing an optional similarity matrix 
    or team labels to prioritize sampling 'confusing' pairs.
    """
    def __init__(self, data_source, batch_size, num_instances, team_labels=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dic = defaultdict(list)
        for index, sample in enumerate(data_source):
            pid = sample[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        
        # Team-aware logic: group PIDs by team to ensure we sample teammates in the same batch
        self.team_to_pids = defaultdict(list)
        if team_labels:
            for pid, team_id in team_labels.items():
                self.team_to_pids[team_id].append(pid)
        self.team_ids = list(self.team_to_pids.keys())

    def __iter__(self):
        indices = []
        batch_indices = []
        
        # Shuffle PIDs
        random.shuffle(self.pids)
        
        # For each batch...
        for i in range(len(self.pids) // self.num_pids_per_batch):
            # Pick P identities
            if self.team_ids:
                # 10/10 Logic: Pick half identities from one team, half from another
                # This forces the model to separate teammates and opponents in same batch
                selected_pids = []
                team_a, team_b = random.sample(self.team_ids, 2)
                
                # Sample 50% from team A
                num_a = self.num_pids_per_batch // 2
                selected_pids.extend(random.sample(self.team_to_pids[team_a], min(num_a, len(self.team_to_pids[team_a]))))
                
                # Sample 50% from team B
                num_b = self.num_pids_per_batch - len(selected_pids)
                selected_pids.extend(random.sample(self.team_to_pids[team_b], min(num_b, len(self.team_to_pids[team_b]))))
            else:
                # Baseline: Random sampling
                selected_pids = self.pids[i * self.num_pids_per_batch : (i + 1) * self.num_pids_per_batch]
            
            for pid in selected_pids:
                t = self.index_dic[pid]
                if len(t) >= self.num_instances:
                    t = np.random.choice(t, size=self.num_instances, replace=False)
                else:
                    t = np.random.choice(t, size=self.num_instances, replace=True)
                batch_indices.extend(t)
            
            indices.extend(batch_indices)
            batch_indices = []
            
        return iter(indices)

    def __len__(self):
        return self.num_identities * self.num_instances
