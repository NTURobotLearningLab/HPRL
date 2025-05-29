import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KarelCNN(BaseFeaturesExtractor):
    """
    Very small CNN: (C,H,W) ➜ 256-D feature vector.
    """
    def __init__(self, observation_space, features_dim: int = 256):
        c, h, w = observation_space.shape
        super().__init__(observation_space, features_dim)

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 64, features_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
