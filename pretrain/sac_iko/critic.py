import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import utils


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=16, input_height=8, input_width=8):
        super(CNNBase, self).__init__()
        
        self._hidden_size = hidden_size

        input_shape = (1, num_inputs, input_height, input_width)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 3, stride=1)), nn.ReLU(), Flatten())

        n_size = self._get_conv_output(input_shape)
        print("CNNBase conv output size: ", n_size)

        self.main = nn.Sequential(
            self.conv,
            init_(nn.Linear(n_size, hidden_size)), nn.ReLU())

        self.train()
    
    @property
    def output_size(self):
        return self._hidden_size

    def _get_conv_output(self, shape):
        input = torch.rand(*shape)
        output_feat = self.conv(input)
        n_size = output_feat.shape[-1]
        return n_size

    def forward(self, inputs):
        # The state value of Karel is Bool (0 or 1)
        #x = self.main(inputs / 255.0)
        x = self.main(inputs / 1.0)

        return x


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, cnn_shape, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.CNN_base1 = CNNBase(cnn_shape[0], obs_dim, cnn_shape[1], cnn_shape[2])

        self.CNN_base2 = CNNBase(cnn_shape[0], obs_dim, cnn_shape[1], cnn_shape[2])

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        
        obs_q1 = self.CNN_base1(obs)
        obs_q2 = self.CNN_base2(obs)

        obs_action_1 = torch.cat([obs_q1, action], dim=-1)
        obs_action_2 = torch.cat([obs_q2, action], dim=-1)

        q1 = self.Q1(obs_action_1)
        q2 = self.Q2(obs_action_2)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

