import gymnasium as gym
import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import os
import gym
import sys
import math
import copy
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from gym import wrappers
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=None):
        """
        The right side of the deque contains the most recent experiences
        The buffer stores a number of past experiences to stochastically sample from
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=self.buffer_size)
        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)

    def add(self, state, action, reward, t, s2):
        experience = (state, action, reward, t, s2)
        self.buffer.append(experience)
        self.count += 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(batch_size, -1)
        t_batch = np.array([_[3] for _ in batch]).reshape(batch_size, -1)
        s2_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class QNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Use the output of the last LSTM cell
        return out, hidden


class LSTMAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = env.action_space.n
        self.n_observations = len(env.reset()[0])

        self.policy_net = LSTMPolicy(self.n_observations, config["hidden_size"], self.n_actions,
                                     config["num_layers"]).to(self.device)
        self.target_net = LSTMPolicy(self.n_observations, config["hidden_size"], self.n_actions,
                                     config["num_layers"]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config["lr"], amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.hidden = None
        self.seq_length = config["seq_length"]
        self.state_buffer = deque(maxlen=self.seq_length)

    def get_state(self):
        """
        Retrieves the current state of the environment.
        The state is a sequence of data points up to the current step, padded with zeros if necessary.
        Returns:
            torch.Tensor: The current state of the environment.
        """
        if len(self.state_buffer) < self.seq_length:
            padding = torch.zeros((self.seq_length - len(self.state_buffer), self.n_observations), device=self.device)
            state = torch.cat((padding, torch.stack(list(self.state_buffer))), dim=0)
        else:
            state = torch.stack(list(self.state_buffer))
        return state.unsqueeze(0)  # Add batch dimension

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.config["eps_end"] + (self.config["eps_start"] - self.config["eps_end"]) * \
                        math.exp(-1. * self.steps_done / self.config["eps_decay"])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action_values, self.hidden = self.policy_net(state, self.hidden)
                return action_values.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.config["batch_size"]:
            return
        transitions = self.memory.sample(self.config["batch_size"])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values, _ = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.config["batch_size"], device=self.device)
        with torch.no_grad():
            next_action_values, _ = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = next_action_values.max(1)[0]
        expected_state_action_values = (next_state_values * self.config["gamma"]) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            self.state_buffer.clear()
            self.state_buffer.append(torch.tensor(state, dtype=torch.float32, device=self.device))

            episode_reward = 0
            episode_loss = 0
            self.hidden = None  # Reset hidden state at the start of each episode

            for t in range(self.config["max_steps_per_episode"]):
                state = self.get_state()
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                episode_reward += reward.item()

                self.state_buffer.append(torch.tensor(observation, dtype=torch.float32, device=self.device))
                next_state = self.get_state()

                self.memory.push(state, action, next_state, reward)

                loss = self.optimize_model()
                if loss is not None:
                    episode_loss += loss

                self.update_target_network()

                if done:
                    break

            wandb.log({
                "episode": i_episode,
                "episode_reward": episode_reward,
                "episode_loss": episode_loss,
                "epsilon": self.config["eps_end"] + (self.config["eps_start"] - self.config["eps_end"]) *
                           math.exp(-1. * self.steps_done / self.config["eps_decay"])
            })

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config["tau"] + target_net_state_dict[
                key] * (1 - self.config["tau"])
        self.target_net.load_state_dict(target_net_state_dict)

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = env.action_space.n
        self.n_observations = len(env.reset()[0])

        self.policy_net = QNet(self.n_observations, self.n_actions).to(self.device)
        self.target_net = QNet(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config['lr'], amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * \
                        math.exp(-1. * self.steps_done / self.config['eps_decay'])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return
        transitions = self.memory.sample(self.config['batch_size'])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.config['batch_size'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config['tau'] + target_net_state_dict[
                key] * (1 - self.config['tau'])
        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            state_stack = deque([torch.zeros(self.n_observations) for _ in range(self.config["stack_size"])],
                                maxlen=self.config["stack_size"])
            state_stack.append(torch.tensor(state, dtype=torch.float32))
            state_stack_tensor = torch.stack(list(state_stack)).unsqueeze(0).to(self.device)

            episode_reward = 0
            episode_loss = 0
            self.hidden = None  # Reset hidden state at the start of each episode

            for t in range(self.config["max_steps_per_episode"]):
                action = self.select_action(state_stack_tensor)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                episode_reward += reward.item()

                if terminated:
                    next_state_stack = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32)
                    next_state_stack = state_stack.copy()
                    next_state_stack.append(next_state)
                    next_state_stack_tensor = torch.stack(list(next_state_stack)).unsqueeze(0).to(self.device)

                self.memory.push(state_stack_tensor, action, reward,
                                 next_state_stack_tensor if not terminated else None, done)

                state_stack = next_state_stack if next_state_stack is not None else state_stack
                state_stack_tensor = next_state_stack_tensor if next_state_stack is not None else state_stack_tensor

                loss = self.optimize_model()
                if loss is not None:
                    episode_loss += loss

                self.update_target_network()

                if done:
                    break

            wandb.log({
                "episode": i_episode,
                "episode_reward": episode_reward,
                "episode_loss": episode_loss,
                "epsilon": self.config["eps_end"] + (self.config["eps_start"] - self.config["eps_end"]) *
                           math.exp(-1. * self.steps_done / self.config["eps_decay"])
            })


def main():
    # config = {
    #     'batch_size': 128,
    #     'gamma': 0.99,
    #     'eps_start': 0.9,
    #     'eps_end': 0.05,
    #     'eps_decay': 1000,
    #     'tau': 0.005,
    #     'lr': 1e-4,
    #     'max_steps_per_episode': 1000,
    #     'num_episodes': 600
    # }
    config = {
        'batch_size': 128,
        'gamma': 0.99,
        'eps_start': 0.9,
        'eps_end': 0.05,
        'eps_decay': 1000,
        'tau': 0.005,
        'lr': 1e-4,
        'max_steps_per_episode': 1000,
        'num_episodes': 600,
        'hidden_size': 128,
        'num_layers': 1,
        'seq_length': 100
    }
    wandb.init(project="dqn-cartpole", name="DQN_CartPole", config=config)
    env = gym.make("CartPole-v1")
    agent = LSTMAgent(env, config)
    agent.train(config['num_episodes'])

    print('Complete')
    wandb.finish()
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG working code for classical control tasks')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=1234')
    parser.add_argument('--tau', type=float, default=0.001, help='adaptability')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='critic learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='critic learning rate')
    parser.add_argument('--bufferlength', type=float, default=2000, help='buffer size in replay buffer')
    parser.add_argument('--l2_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--episode_len', type=int, default=1000, help='episodic lengths')
    parser.add_argument('--episode_steps', type=int, default=1000, help='steps per episode')
    parser.add_argument('--epsilon', type=float, default=0.01, help='noide standard deviation')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='noide standard deviation')
    parser.add_argument('--is_train', type=bool, default=False, help='train mode or test mode. Default is test mode')
    parser.add_argument('--actor_weights', type=str, default='ddqn_cartpole',
                        help='Filename of actor weights. Default is actor_pendulum')
    args = parser.parse_args()

    main(args)