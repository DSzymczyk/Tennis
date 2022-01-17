import numpy as np
import random

from ReplayBuffer import ReplayBuffer
from model import PolicyModel, ValueModel

import torch
import torch.nn.functional as F
import torch.optim as optim

from OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:

    def __init__(self, nS, nA, random_seed, load_checkpoint, checkpoint_prefix='', buffer_size=int(1e6), batch_size=512,
                 gamma=0.99, tau=1e-3, policy_lr=1e-4, value_lr=1e-3, weight_decay=0, learn_every=4, agent_no=0):
        self.state_size = nS
        self.action_size = nA
        self._checkpoint_prefix = checkpoint_prefix
        self.seed = random.seed(random_seed)

        self._online_policy_model = PolicyModel(self.state_size, self.action_size, random_seed).to(device)
        self._target_policy_model = PolicyModel(self.state_size, self.action_size, random_seed).to(device)
        self._online_value_model = ValueModel(self.state_size, self.action_size, random_seed).to(device)
        self._target_value_model = ValueModel(self.state_size, self.action_size, random_seed).to(device)

        if load_checkpoint:
            self.load_weights(agent_no)

        self._policy_optimizer = optim.Adam(self._online_policy_model.parameters(), lr=policy_lr)
        self._value_optimizer = optim.Adam(self._online_value_model.parameters(), lr=value_lr, weight_decay=weight_decay)

        self._replay_buffer = ReplayBuffer(buffer_size, batch_size, random_seed)
        self._noise = OrnsteinUhlenbeckProcess(nA, random_seed)

        self.soft_update(self._online_value_model, self._target_value_model, 1.0)
        self.soft_update(self._online_policy_model, self._target_policy_model, 1.0)

        self._batch_size = batch_size
        self._gamma = gamma
        self._tau = tau
        self._learn_every = learn_every
        self._step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Add current step to replay buffer and learn every `_learn_every` episode.
        :param state: current state
        :param action: current action
        :param reward: current reward
        :param next_state: next state
        :param done: boolean flag if episode is done
        """
        self._replay_buffer.add(state, action, reward, next_state, done)
        self._step = (self._step + 1) % self._learn_every
        if (len(self._replay_buffer) > self._batch_size) and (self._step == 0):
                experiences = self._replay_buffer.get_sample()
                self.learn(experiences, self._gamma)

    def act(self, state, add_noise):
        """
        Selecting actions according to agent and clipping them to -1 to 1 borders.
        :param state: current state
        :param add_noise: boolean value whenever to add noise to action
        :return: Actions selected by agent
        """
        state = torch.from_numpy(state).float().to(device)
        self._online_policy_model.eval()
        with torch.no_grad():
            action = self._online_policy_model(state).cpu().data.numpy()
        self._online_policy_model.train()
        if add_noise:
            action += self._noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """
        Training model using DDPG.
        """
        states, actions, rewards, next_states, dones = experiences

        target_policy_model_action = self._target_policy_model(next_states)
        target_model_q_values = self._target_value_model(next_states, target_policy_model_action)

        target_q_values = rewards + (gamma * target_model_q_values * (1 - dones))
        expected_q_values = self._online_value_model(states, actions)
        value_loss = F.mse_loss(expected_q_values, target_q_values)

        self._value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online_value_model.parameters(), 1)
        self._value_optimizer.step()

        online_policy_model_action = self._online_policy_model(states)
        policy_loss = -self._online_value_model(states, online_policy_model_action).mean()
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        self.soft_update(self._online_value_model, self._target_value_model, self._tau)
        self.soft_update(self._online_policy_model, self._target_policy_model, self._tau)

    def soft_update(self, online_model, target_model, tau):
        """
        Updating target model using Polyak averaging.
        :param online_model: agents online model
        :param target_model: agents target model
        :param tau: copy weight parameter, greater the tau is more will be copied from online model
        """
        for target_param, local_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset_noise(self):
        self._noise.reset()

    def save_weights(self, agent_no):
        """
        Save weights of policy and value models.
        """
        torch.save(self._online_policy_model.state_dict(), f'weights/{self._checkpoint_prefix}policy_checkpoint{agent_no}.pth')
        torch.save(self._online_value_model.state_dict(), f'weights/{self._checkpoint_prefix}value_checkpoint{agent_no}.pth')

    def load_weights(self, agent_no):
        """
        Load weights of policy and value models.
        """
        self._online_policy_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}policy_checkpoint{agent_no}.pth'))
        self._online_policy_model.eval()
        self._target_policy_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}policy_checkpoint{agent_no}.pth'))
        self._target_policy_model.eval()
        self._online_value_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}value_checkpoint{agent_no}.pth'))
        self._online_value_model.eval()
        self._target_value_model.load_state_dict(torch.load(f'weights/{self._checkpoint_prefix}value_checkpoint{agent_no}.pth'))
        self._target_value_model.eval()
