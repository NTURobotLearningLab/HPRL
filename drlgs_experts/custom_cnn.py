# custom_cnn.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # observation_space.shape = (8, 8, 8), but we will transpose to (C, H, W)
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),  # (C=8) -> (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (32) -> (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute flattened output size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, 8, 8, 8).permute(0, 3, 1, 2)  # (B, C, H, W)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # obs: (B, 8, 8, 8) => (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
        return self.linear(self.cnn(obs))
