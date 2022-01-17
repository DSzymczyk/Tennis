import random
from collections import namedtuple, deque

import numpy as np
import torch

experience = namedtuple("Experience", field_names="state action reward next_state done")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._deque = deque(maxlen=self._buffer_size)
        self.seed = random.seed(seed)

    def __iter__(self):
        """
        :return: Iterator of ReplayBuffer with size equal to `_batch_size`.
        """
        return ReplayBufferIterator(self._deque, self._batch_size)

    def __len__(self):
        """
        :return: Current length of ReplayBuffer.
        """
        return self._deque.__len__()

    def __repr__(self):
        result = ''
        for el in self._deque:
            result += str(el) + '\n'
        return result

    def add(self, state, action, reward, next_state, done):
        """
        Adding new experiance to ReplayBuffer
        :param state: current state
        :param action: current action
        :param reward: current reward
        :param next_state: next state
        :param done: boolean flag if episode is done
        """
        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        exp = experience(state, action, reward, next_state, done)
        self._deque.append(exp)

    def get_sample(self):
        """
        If deque size is greater or equal batch size return tuple of tensors with random selected experiances
        otherwise return None
        :return: tuple of tensors with states, actions, rewards, next_states, dones or None
        """
        if self._deque.__len__() < self._batch_size:
            return None
        experiences = list(iter(self))
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones


class ReplayBufferIterator:

    def __init__(self, buffer_deque, batch_size):
        self._batch_size = batch_size
        self._buffer_sample = random.sample(buffer_deque, batch_size)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._batch_size:
            raise StopIteration
        element = self._buffer_sample[self._index-1]
        self._index += 1
        return element

    def __len__(self):
        return self._batch_size
