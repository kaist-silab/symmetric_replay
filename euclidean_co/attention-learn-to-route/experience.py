import numpy as np
import torch


class Experience(object):
    def __init__(self, max_size=1000, device='cpu', reward_prioritized=False):
        self.memory = []
        self.max_size = max_size
        self.device = device
        self.reward_prioritized = reward_prioritized

    def add_experience(self, experience):
        """Experience should be a list of (x, pi, score, log_likelihood) tuples"""
        self.memory.append(experience)
        self.memory = self.memory[:self.max_size]

    def sample(self, n):
        """Randomly sample a batch of experiences from memory"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            if self.reward_prioritized:
                scores = np.array([e[2] for e in self.memory])
                sample_idx = np.random.choice(len(self.memory), size=n, replace=False, p=scores/np.sum(scores))
            # scores = np.array([e[1] for e in self.memory])
            # sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
            # sample = [self.memory[i] for i in sample]
            # pi = [x[0] for x in sample]
            # scores = [x[1] for x in sample]
            else:
                sample_idx = np.random.choice(len(self.memory), size=n, replace=False)
            x, pi, costs, ll = [], [], [], []
            for i in sample_idx:
                x.append(self.memory[i][0])
                pi.append(self.memory[i][1])
                costs.append(self.memory[i][2])
                ll.append(self.memory[i][3])

        x = torch.stack(x)  # only for TSP
        pi = torch.stack(pi)
        costs = torch.tensor(costs, dtype=torch.float)
        ll = torch.tensor(ll, dtype=torch.float)

        return x.to(self.device), pi.to(self.device), costs.to(self.device), ll.to(self.device)