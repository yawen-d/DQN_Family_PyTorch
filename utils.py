import gym
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


# input extraction

# define the resize transfomation
resize = T.Compose([T.ToPILImage(),
                    T.Resize((84,84), interpolation=Image.CUBIC),
                    T.Grayscale(num_output_channels=1),
                    T.ToTensor()])

# get a resized screen
def get_screen(env):
    '''
    return a resized screen (tensor) in (BCHW)
    '''
    # Returned screen is size (1000,1000,3), need to transpose to 
    # torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    # print(type(screen), screen.shape)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    
    # Resize, and add a batch dimension (HW)
    screen = resize(screen).squeeze(0)
    return screen
    
# show the screen
def show_screen(screen):
    plt.figure()
    plt.imshow(screen.cpu().numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


class SumTree(object):
    position = 0

    def __init__(self, capacity):
        self.length = 0
        self.capacity = capacity
        self.tree = np.zeros( 2 * capacity - 1 )
        self.data = np.zeros( capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.position + self.capacity - 1

        self.data[self.position] = data
        self.update(idx, priority)
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length+1, self.capacity)

    def _retrieve(self, idx, p):  # retrieve p
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # already get to the leaf
            return idx

        if p <= self.tree[left]:
            return self._retrieve(left, p)  # recursion
        else:
            return self._retrieve(right, p-self.tree[left])

    def get_leaf(self, s):
        idx = self._retrieve(0, s) # tree index
        dataIdx = idx - self.capacity + 1 # data index

        return (idx, self.tree[idx], self.data[dataIdx])

    def total(self):
        return self.tree[0]
    
    def __len__(self):
        return self.length