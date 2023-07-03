import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import Agent
from .import utils


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, device, cfg, critic_net, critic_target_net, actor_net):
        super().__init__()
        self.device                         = device
        self.actor                          = actor_net
        self.critic                         = critic_net
        self.critic_target                  = critic_target_net
        self.action_range                   = [-3.0, 3.0]
        self.action_dim                     = cfg['SAC']['hidden_size']
        self.alpha_lr                       = cfg['SAC']['agent']['alpha_lr']
        self.alpha_betas                    = cfg['SAC']['agent']['alpha_betas']
        self.actor_lr                       = cfg['SAC']['agent']['actor_lr']
        self.actor_betas                    = cfg['SAC']['agent']['actor_betas']
        self.critic_lr                      = cfg['SAC']['agent']['critic_lr']
        self.critic_betas                   = cfg['SAC']['agent']['critic_betas']

        self.discount                       = cfg['SAC']['agent']['discount']
        self.critic_tau                     = cfg['SAC']['agent']['critic_tau']
        self.actor_update_frequency         = cfg['SAC']['agent']['actor_update_frequency']
        self.critic_target_update_frequency = cfg['SAC']['agent']['critic_target_update_frequency']
        self.log_histogram_interval         = cfg['SAC']['agent']['log_histogram_interval']
        self.batch_size                     = cfg['SAC']['agent']['batch_size']
        self.learnable_temperature          = cfg['SAC']['agent']['learnable_temperature']

        self.log_alpha                      = torch.tensor(np.log(cfg['SAC']['agent']['init_temperature'])).to(self.device)
        
        self.log_alpha.requires_grad = True
        
        # set target entropy to -|A|
        self.target_entropy = -self.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr,
                                                betas=self.actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 betas=self.critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_lr,
                                                    betas=self.alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = obs.to(self.device)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2
        return action

    def update_critic(self, obs, action, reward, next_obs, not_done):
        update_critic_info = {}
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        update_critic_info['train_critic/loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #self.critic.log(logger, step)
        update_critic_info['q1'] = self.critic.outputs['q1']
        update_critic_info['q2'] = self.critic.outputs['q2']

        return update_critic_info

    def update_actor_and_alpha(self, obs):
        update_actor_info = {}
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        update_actor_info['train_actor/loss'] = actor_loss.item()
        update_actor_info['train_actor/target_entropy'] = self.target_entropy
        update_actor_info['train_actor/entropy'] = -log_prob.mean().item()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #self.actor.log(logger, step)
        update_actor_info['mu'] = self.actor.outputs['mu']
        update_actor_info['std'] = self.actor.outputs['std']

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            update_actor_info['train_alpha/loss'] = alpha_loss.item()
            update_actor_info['train_alpha/value'] =  self.alpha.item()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        
        return update_actor_info

    def update(self, replay_buffer, step):
        update_info_dict = {}
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)
        
        update_info_dict['train/batch_reward'] = reward.mean()


        update_info_critic = self.update_critic(obs, action, reward, next_obs, not_done_no_max)
        update_info_dict['critic_info'] = update_info_critic

        if step % self.actor_update_frequency == 0:
            update_info_actor = self.update_actor_and_alpha(obs)
            update_info_dict['actor_info'] = update_info_actor

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        return update_info_dict

