import torch
import torch.nn as nn
import numpy as np


class NNPlanner(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)

    def load(self, weight_path):
        state = torch.load(weight_path, map_location="cpu")
        self.net.load_state_dict(state)
        self.eval()

    def propose(self, features_np):
        """
        features_np: np.ndarray, shape = (in_dim,)
        return: (l_target, T)
        """
        x = torch.from_numpy(features_np).float().unsqueeze(0)
        with torch.no_grad():
            y = self.forward(x)[0]
        l_pred = float(y[0])

        return l_pred
