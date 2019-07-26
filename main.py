import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils import get_screen

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamters
NUM_EPISODES = 10

def main():

    # make environment
    env = gym.make('Acrobot-v1').unwrapped

    # # initialize memory D with capacity N
    # memory = ReplayMemory()

    # # initialize policy_net, and its parameters
    # policy_net = DQN()

    # # initialize target_net with same parameters as policy_net
    # target_net = policy_net.copy()

    for i_episode in range(NUM_EPISODES):
        # initialize initial state

        for t in range(100):
            # apply epsilon action generator getting an action
            # if prob > episilon:
            #     target_net.action()
            # else:
            #     action = env.action_space.sample()

            # execute action in env
            # get transition (state, action, reward, next_state, done)

            # store transition in memory

            # sample random minibatch of transitions from memory

            # get target value
            # if transition is done, then target = reward
            # if transition is not done, 
            # then target = reward + gamma * maximum of target_net(state,action)

            # gradient descent on loss (target - policy_net(state,action))


            print("Episode finished after {} timesteps".format(t+1))
            break

    pass


if __name__ == '__main__':
    main()
