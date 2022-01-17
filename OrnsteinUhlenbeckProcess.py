import copy
import random

import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, random_seed, mu=0., theta=1e-3, sigma=2e-3):
        """
        Ornstein-Uhlenbeck process
        :param size: size of noise vector
        :param random_seed: random seed
        :param mu: represents a long-term mean of the OU process
        :param theta: mean reverting speed value
        :param sigma: deviation of stochastic factor
        """
        self._state = None
        self._mu = mu * np.ones(size)
        self._theta = theta
        self._sigma = sigma
        self._seed = random.seed(random_seed)
        self.reset()

    def reset(self):
        """
        Reset internal state of the process
        """
        self._state = copy.copy(self._mu)

    def sample(self):
        """
        Calculate internal state of process and return as noise sample
        :return: internal state as noise  sample
        """
        x = self._state
        dx = self._theta * (self._mu - x) + self._sigma * np.array([random.random() for _ in range(len(x))])
        self._state = x + dx
        return self._state
