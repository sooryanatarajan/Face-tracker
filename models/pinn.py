import torch
import torch.nn as nn

class MotionPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, pos):
        return self.net(pos)
