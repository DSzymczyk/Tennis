import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyModel(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_dims=(300, 200)):
        """
        Initialize policy model
        :param state_size: Input layer size
        :param action_size: Output layer size
        :param seed: Random seed
        :param hidden_dims: Tuple of hidden layer sizes
        """
        super(PolicyModel, self).__init__()
        self._seed = torch.manual_seed(seed)
        self._input_layer = nn.Linear(state_size, hidden_dims[0])
        self._hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self._hidden_layers.append(hidden_layer)
        self._output_layer = nn.Linear(hidden_dims[-1], action_size)

    def forward(self, state):
        """
        Transform state -> action probabilities
        :param state: current state
        :return: action that should be performed according to model
        """
        x = F.relu(self._input_layer(state))
        for layer in self._hidden_layers:
            x = F.relu(layer(x))
        return torch.tanh(self._output_layer(x))


class ValueModel(nn.Module):

    def __init__(self, state_size, action_size, seed, hidden_dims=(300, 200)):
        """
        Initialize value model
        :param state_size: Input layer size
        :param action_size: Output layer size
        :param seed: Random seed
        :param hidden_dims: Tuple of hidden layer sizes
        """
        super(ValueModel, self).__init__()
        self._seed = torch.manual_seed(seed)
        self._input_layer = nn.Linear(state_size, hidden_dims[0])
        self._hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0:
                in_dim = in_dim + action_size
            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1])
            self._hidden_layers.append(hidden_layer)
        self._output_layer = nn.Linear(hidden_dims[-1], action_size)

    def forward(self, state, action):
        """
        Transforms (state, action) pair into Q-values for each action
        :param state: Current state
        :param action: Q-values corresponding to state
        :return:
        """
        x = F.relu(self._input_layer(state))
        for i, layer in enumerate(self._hidden_layers):
            if i == 0:
                x = torch.cat((x, action), dim=1)
            x = F.relu(layer(x))
        return self._output_layer(x)
