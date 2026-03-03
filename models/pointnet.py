import torch
import torch.nn as nn
import torch.nn.functional as F
from t_net import InputTNet

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.input_tnet = InputTNet(k=5)

        self.mlp1 = nn.Conv1d(5, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x: (B, N, 5)
        """
        x = x.transpose(1, 2)        # (B, 5, N)

        # Input T-Net
        trans = self.input_tnet(x)   # (B, 5, 5)
        x = torch.bmm(trans, x)      # (B, 5, N)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))

        x = torch.max(x, 2)[0]       # (B, 256)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
