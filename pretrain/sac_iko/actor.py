import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from . import utils


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=16, input_height=8, input_width=8):
        super(CNNBase, self).__init__()
        
        self._hidden_size = hidden_size

        input_shape = (1, num_inputs, input_height, input_width)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 4, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 2, stride=1)), nn.ReLU(), Flatten())

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


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, cnn_shape, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        
        self.CNN_base = CNNBase(cnn_shape[0], obs_dim, cnn_shape[1], cnn_shape[2])

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs_cnn = self.CNN_base(obs)
        mu, log_std = self.trunk(obs_cnn).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)
