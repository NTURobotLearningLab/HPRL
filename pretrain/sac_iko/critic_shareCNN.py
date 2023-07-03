import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_scalar(module, weight_init, bias_init):
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=16, action_dim=16, input_height=8, input_width=8):
        super(CNNBase, self).__init__()
        
        input_shape  = (1, num_inputs, input_height, input_width)

        init_        = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        #init_scalar_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                       constant_(x, 0))

        init_scalar_ = lambda m: init_scalar(m, lambda w: nn.init.normal_(w, mean=0.0, std=0.01), lambda x: nn.init.
                               constant_(x, 0))

        self.conv = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 4, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 2, stride=1)), nn.ReLU(), Flatten())

        n_size = self._get_conv_output(input_shape)

        self.main = nn.Sequential(
            self.conv,
            init_(nn.Linear(n_size, hidden_size)), nn.ReLU())

        print("CNNBase conv output size: ", n_size)
        
        n_size_with_action = hidden_size + action_dim
        print("CNNBase conv output + action size: ", n_size_with_action)

        self.critic_linear_1 = nn.Sequential(
            init_(nn.Linear(n_size_with_action, hidden_size)), nn.ReLU(),
            init_scalar_(nn.Linear(hidden_size, 1)))
            #nn.Linear(hidden_size, 1))

        self.critic_linear_2 = nn.Sequential(
            init_(nn.Linear(n_size_with_action, hidden_size)), nn.ReLU(),
            init_scalar_(nn.Linear(hidden_size, 1)))
            #nn.Linear(hidden_size, 1))

        self.train()
    
    def _get_conv_output(self, shape):
        input = torch.rand(*shape)
        output_feat = self.conv(input)
        n_size = output_feat.shape[-1]
        return n_size

    def forward(self, inputs, action):
        # The state value of Karel is Bool (0 or 1)
        #x = self.main(inputs / 255.0)
        x = self.main(inputs / 1.0)
        obs_action = torch.cat([x, action], dim=-1)

        return self.critic_linear_1(obs_action), self.critic_linear_2(obs_action)


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, cnn_shape, obs_dim, action_dim, hidden_dim_ignore, hidden_depth_ignore):
        super().__init__()

        self.CNN_base = CNNBase(cnn_shape[0], obs_dim, action_dim, cnn_shape[1], cnn_shape[2])
        self.outputs = dict()

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        
        q1, q2 = self.CNN_base(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

