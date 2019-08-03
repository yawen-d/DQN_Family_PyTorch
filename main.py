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

from utils import *
from DQN import *

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Hyperparamters
NUM_EPISODES = 1000
MEMORY_CAPA = 10000
MAX_EPS = 0.9
MIN_EPS = 0.1
EPS_DECAY = 200
BATCH_SIZE = 32
GAMMA = 0.99
UPDATE_FREQ = 5
SAVE_FREQ = 50
MODEL_PATH = './checkpoints/'

episode_durations = []



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # sample random minibatch of transitions from memory
    batch = Transition(*zip(*transitions))

    # print(batch.state)#, batch.action.shape, batch.reward.shape, batch.done.shape)

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    done_batch = torch.stack(batch.done)
    next_state_batch = torch.stack(batch.next_state)

    state_action_values = policy_net(state_batch).gather(1,action_batch)
    
    not_done_mask = [k for k,v in enumerate(done_batch) if v == 0]

    not_done_next_states = next_state_batch[not_done_mask]

    next_state_values = torch.zeros_like(state_action_values)

    next_state_values[not_done_mask] = target_net(not_done_next_states).max(1)[0].view(-1,1).detach()
    
    # Compute the expected Q values
    target_values = reward_batch + (GAMMA * next_state_values)

    assert state_action_values.shape == target_values.shape

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, target_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main():

    # make environment
    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    initial_screen = get_screen(env)
    # show_screen(initial_screen)

    screen_h, screen_w = initial_screen.shape
    num_actions = env.action_space.n

    # initialize policy_net, and its parameters
    policy_net = DQN(screen_h, screen_w, num_actions).to(device)

    # initialize target_net with same parameters as policy_net
    target_net = DQN(screen_h, screen_w, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    # initialize memory D with capacity N
    memory = ReplayMemory(MEMORY_CAPA)

    update_counts = 0

    for i_episode in range(NUM_EPISODES):
        
        env.reset()
        # initialize initial state
        ini_screen = get_screen(env)
        state = torch.stack([ini_screen]*4)

        for t in range(200):
            # select and perform an action
            eps = MIN_EPS + (MAX_EPS - MIN_EPS) \
                * math.exp(-1. * update_counts / EPS_DECAY)

            if random.random() > eps:
                with torch.no_grad():
                    # print(state.unsqueeze(0).shape)
                    action = target_net(state.unsqueeze(0)).max(1)[1].view(1)
            else:
                action = torch.tensor([env.action_space.sample()],
                                        device=device, dtype=torch.long)

            # execute action in env
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)

            # Observe new states
            if not done:
                current_screen = get_screen(env)
                next_state = torch.stack(list(torch.unbind(state, dim=0)[1:]) + [current_screen])
            else:
                next_state = torch.zeros([4,84,84])
            
            # get transition (state, action, reward, next_state, done) 
            # and push to the memory
            memory.push_one(state, action, next_state, reward, done)

            # move to the next state
            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)

            update_counts += 1

            if done: 
                episode_durations.append(t + 1)
                plot_durations()
                break
        
        if i_episode % UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print('Target Net updated!')
        print("Episode {} finished after {} timesteps".format(i_episode,t+1))
        if i_episode % SAVE_FREQ == 0:
            torch.save(policy_net.state_dict(), '{}{}.pth'.format(MODEL_PATH,i_episode))
            print('Model saved as {}{}.pth'.format(MODEL_PATH,i_episode))


    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
