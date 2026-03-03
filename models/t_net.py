import torch
import torch.nn as nn
import torch.nn.functional as F

class InputTNet(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k * k)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, k, N)
        returns: (B, k, k)
        """
        B = x.size(0)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = torch.max(x, 2)[0]   # (B, 256)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # Initialize as identity
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + identity

        return x.view(B, self.k, self.k)
