import random
import numpy as np
from collections import namedtuple

from utils import SumTree

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push_one(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):

    epsilon = 0.001 # small amount to avoid 0 priority
    alpha = 0.6 # convert the importance of TD error to priority
    beta = 0.4 # annealing the bias to 1
    beta_increment_per_sampling = 0.0005
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.memory = SumTree(capacity)
        self.full = False
    
    def push_one(self, *args):
        """saves a transition"""
        data = Transition(*args)
        max_p = np.max(self.memory.tree[-self.memory.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.memory.add(max_p, data) # set the max_p for new p
        if not self.full:
            self.full = True \
            if (self.memory.position + 1) % self.memory.capacity == 0 \
            else False
            
            if self.full:
                print("Memory Full!")
        
    def sample(self, batch_size):
        batch_idx = np.empty((batch_size,), dtype=np.int32)
        batch_data = np.empty((batch_size,),dtype=object)
        norm_ISWeights = np.empty((batch_size,),dtype=np.float32)
        
        pri_seg = self.memory.total()/batch_size # priority segment
        self.beta = np.min([1., self.beta + 
                            self.beta_increment_per_sampling]) # update beta
        if not self.full:
            pos = -self.memory.capacity + self.memory.position
            min_p = np.min(self.memory.tree[-self.memory.capacity:pos])
        else:
            min_p = np.min(self.memory.tree[-self.memory.capacity:])
        min_prob = min_p / self.memory.total()
        for i in range(batch_size):
            low, hig = pri_seg * i, pri_seg * (i + 1)
            s = np.random.uniform(low, hig)
            idx, p, data = self.memory.get_leaf(s)
            batch_idx[i], batch_data[i] = idx, data
            prob = p / self.memory.total()
            
            # ISWeight = (N*prob)**(-self.beta)
            # max_ISWeight = (N*min_prob)**(-self.beta)
            # norm_ISWeights[i] = ISWeight / max_ISWeight

            # the above computation can be simplified as follows
            norm_ISWeights[i] = np.power(prob/min_prob, -self.beta)
        # print("beta:", self.beta)
        return batch_idx, batch_data, norm_ISWeights

    def batch_update(self, b_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for bi, p in zip(b_idx, ps):
            self.memory.update(bi, p)

    def __len__(self):
        return self.memory.length