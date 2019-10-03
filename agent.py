import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import numpy as np

import matplotlib.pyplot as plt

import random

from memory import ReplayMemory, Transition
from networks import DQN
from config import AgentConfig, EnvConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(AgentConfig, EnvConfig):
    def __init__(self):
        """initialize the agent and the environments"""
        self.env = gym.make(self.ENV)
        self.memory = ReplayMemory(capacity = self.MEMORY_CAPA, per = self.PER)
        self._build_nets()
        self.epsilon = self.MAX_EPS

    def _build_nets(self):
        self.num_actions = self.env.action_space.n
        self.policy_net = DQN(self.num_actions, input_size = 4, 
                                hidden_size = 32).to(device)
        self.target_net = DQN(self.num_actions, input_size = 4, 
                                hidden_size = 32).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def eps_decay(self):
        self.epsilon = max(self.epsilon * self.DECAY_RATE, self.MIN_EPS)

    def greedy_action(self, state, eps):
        if random.random() > eps:
            with torch.no_grad():
                q_value = self.target_net(state.unsqueeze(0).float())
                action = q_value.max(1)[1].view(1)
        else:
            action = torch.tensor([self.env.action_space.sample()],
                                    device=device, dtype=torch.long)
        return action

    def policy_action(self, state):
        with torch.no_grad():
            q_value = self.policy_net(state.unsqueeze(0).float())
            action = q_value.max(1)[1].view(1)
        return action

    def train(self):
        # define the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.LR)
        self.episode_durations = []
        self.policy_net_scores = []
        self.eps_list = []

        # train the agent for designated number of episodes
        for i_episode in range(self.START_EPISODE, self.NUM_EPISODES):
            # initialize initial state
            state = self.env.reset()
            state = torch.from_numpy(state)
            
            # decay the epsilon
            self.eps_decay()

            for t in range(501):
                # get the action based on state by greedy policy
                action = self.greedy_action(state, self.epsilon)

                # execute action in environment
                obs, reward, done, _ = self.env.step(action.item())
                if done:
                    reward = -1.
                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device=device)

                # observe new states
                if not done:
                    next_state = torch.from_numpy(obs)
                else:
                    next_state = torch.zeros_like(state)
                
                # get transition (state, action, reward, next_state, done) 
                # and push to the memory
                self.memory.push_one(state, action, next_state, reward, done)

                # move to the next state
                state = next_state
                
                # optimize the model
                self._optimize_model()

                if done: 
                    self.episode_durations.append(t + 1)
                    break
            
            print("Episode {} finished after {} timesteps -- EPS: {}" \
                                    .format(i_episode, t+1, self.epsilon))

            self.policy_net_scores.append(self.demo())
            self.eps_list.append(self.epsilon)
            if (i_episode + 1) % self.UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print('Target Net updated!')

        print('Complete!')

    def _optimize_model(self, verbose = False):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        # sample random minibatch of transitions from memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)
        next_state_batch = torch.stack(batch.next_state)

        state_action_values = self.policy_net(state_batch.float()).gather(1,action_batch)
        if verbose:
            print(self.policy_net(state_batch.float()))
            print(action_batch)
            print(state_action_values)
        not_done_mask = [k for k, v in enumerate(done_batch) if v == 0]
        not_done_next_states = next_state_batch[not_done_mask]
        next_state_values = torch.zeros_like(state_action_values)

        # DQN
        next_state_values[not_done_mask] = self.target_net(not_done_next_states.float()).max(1)[0].view(-1,1).detach()
        if verbose:
            print("--")
            print(done_batch)
            print(not_done_mask)
            print(next_state_batch)
            print(next_state_values)

        # Compute the expected Q values
        target_values = reward_batch + (self.GAMMA * next_state_values)

        assert state_action_values.shape == target_values.shape

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, target_values)
        if verbose:
            print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def save_results(self):
        PATH = self.RES_PATH # + str(self.EXPERIMENT_NO) + "/"
        # os.mkdir(PATH)

        # plot and save figure
        plt.figure(0)
        durations_t = torch.tensor(self.policy_net_scores, dtype = torch.float)
        plt.title("DQN Experiment %d" % self.EXPERIMENT_NO)
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
        plt.plot(np.array(self.episode_durations, dtype = np.float))
        # Take 10 episode policy net score averages and plot them too
        if len(durations_t) >= 10:
            means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy())
        plt.savefig(PATH + "%d-result.png" % self.EXPERIMENT_NO)
        # plt.show()
        self._write_results(PATH)
    
    def _write_results(self, PATH):
        # save the text results
        attr_dict = {
            "EXPERIMENT_NO" : self.EXPERIMENT_NO,
            "START_EPISODE" : self.START_EPISODE,
            "NUM_EPISODES" : self.NUM_EPISODES,
            "MEMORY_CAPA" : self.MEMORY_CAPA,
            "MAX_EPS" : self.MAX_EPS,
            "MIN_EPS" : self.MIN_EPS,
            "LR" : self.LR,
            "DECAY_RATE" : self.DECAY_RATE,
            "BATCH_SIZE" : self.BATCH_SIZE,
            "GAMMA" : self.GAMMA,
            "UPDATE_FREQ" : self.UPDATE_FREQ,
            "RES_PATH" : self.RES_PATH
        }
        with open(PATH + "%d-log.txt" % self.EXPERIMENT_NO, 'w') as f:
            for k,v in attr_dict.items():
                f.write("{} = {}\n".format(k, v))
            f.write("------------------\n")
            for i in range(len(self.episode_durations)):
                f.write("Ep %d finished after %d steps -- EPS: %.4f -- policy net score: %.2f\n"
                    % (i + 1, self.episode_durations[i], self.eps_list[i], self.policy_net_scores[i]))

    def demo(self, verbose = False):
        scores = []
        for i_episode in range(100):
            # initialize initial state
            state = self.env.reset()
            state = torch.from_numpy(state)

            for t in range(501):
                # get the action based on state by greedy policy
                action = self.policy_action(state)

                # execute action in environment
                obs, _, done, _ = self.env.step(action.item())
                state = torch.from_numpy(obs)

                if done: 
                    scores.append(t + 1)
                    if verbose:
                        print("Episode {} finished after {} timesteps" \
                                        .format(i_episode, t+1))
                    break
        net_score = np.array(scores, dtype = float)
        
        if verbose:
            print("policy net scores -- mean: {}, var: {}, max: {}, min: {}".format(
                            net_score.mean(), net_score.var(), net_score.max(), net_score.min()
                            ))
        else:
            print("policy net scores -- mean:", net_score.mean())
        return net_score.mean()

            

