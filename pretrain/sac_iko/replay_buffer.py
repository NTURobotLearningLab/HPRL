import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, num_steps, num_processes, obs_shape, action_shape, device):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((num_steps, num_processes, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((num_steps, num_processes, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((num_steps, num_processes, action_shape), dtype=np.float32)
        self.rewards = np.empty((num_steps, num_processes, 1), dtype=np.float32)
        self.not_dones = np.empty((num_steps, num_processes, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((num_steps, num_processes, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.num_steps if self.full else self.idx

    def add(self, obs, action, reward, next_obs, n_done, n_done_no_max):
        #np.copyto(self.obses[self.idx], obs)
        #np.copyto(self.actions[self.idx], action)
        #np.copyto(self.rewards[self.idx], reward)
        #np.copyto(self.next_obses[self.idx], next_obs)
        #np.copyto(self.not_dones[self.idx], n_done)
        #np.copyto(self.not_dones_no_max[self.idx], n_done_no_max)
            
        self.obses[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obses[self.idx] = next_obs
        self.not_dones[self.idx] = n_done
        self.not_dones_no_max[self.idx] = n_done_no_max

        self.idx = (self.idx + 1) % self.num_steps
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.num_steps * self.num_processes if self.full else self.idx * self.num_processes,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses.reshape(-1, *self.obses.shape[2:])[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions.reshape(-1, *self.actions.shape[2:])[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards.reshape(-1, *self.rewards.shape[2:])[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses.reshape(-1, *self.next_obses.shape[2:])[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones.reshape(-1, *self.not_dones.shape[2:])[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max.reshape(-1, *self.not_dones_no_max.shape[2:])[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
