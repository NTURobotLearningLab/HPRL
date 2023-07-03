import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain.ppo_iko.distributions import Bernoulli, Categorical, DiagGaussian
from pretrain.ppo_iko.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, envs, base_kwargs=None):
        super(Policy, self).__init__()
        
        obs_shape = envs.observation_space.shape
      
        assert len(obs_shape) == 3
        assert base_kwargs is not None

        if len(obs_shape) == 3:
            base = CNNBase
        elif len(obs_shape) == 1:
            base = MLPBase
        else:
            raise NotImplementedError

        print("Policy base hidden (program) dimenstion: ", base_kwargs["hidden_size"])
        self.base = base(obs_shape[0], **base_kwargs)
        
        self.dist = DiagGaussian(self.base.output_size, self.base.output_size)


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, z_logit, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(z_logit)

        if deterministic:
            z = dist.mode()
        else:
            z = dist.sample()

        z_log_probs = dist.log_probs(z)
        dist_entropy = dist.entropy().mean()
        
        return value, z, z_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, z):
        value, z_logit, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(z_logit)

        action_log_probs = dist.log_probs(z)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=16, input_height=8, input_width=8):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        
        input_shape = (1, num_inputs, input_height, input_width)
        print("input_shape: ", input_shape)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        #self.main = nn.Sequential(
        #    init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
        #    init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        #    init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
        #    init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        
        self.conv = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 4, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 2, stride=1)), nn.ReLU(), Flatten())

        n_size = self._get_conv_output(input_shape)
        print("CNNBase conv output size: ", n_size)

        self.main = nn.Sequential(
            self.conv,
            init_(nn.Linear(n_size, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def _get_conv_output(self, shape):
        input = torch.rand(*shape)
        output_feat = self.conv(input)
        n_size = output_feat.shape[-1]
        return n_size

    def forward(self, inputs, rnn_hxs, masks):
        # The state value of Karel is Bool (0 or 1)
        #x = self.main(inputs / 255.0)
        x = self.main(inputs / 1.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=16):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
