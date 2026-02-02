import torch
import torch.nn as nn
import numpy as np


class NNPlanner(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_need = nn.Linear(128, 1)
        self.head_delta = nn.Linear(128, 1)

    def forward(self, x):
        h = self.backbone(x)
        danger_logit = self.head_need(h)
        l_target = self.head_delta(h)
        return danger_logit, l_target

    def load(self, weight_path):
        state = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state)
        self.eval()

    def propose(self, features_np, p_thresh=0.5):
        x = torch.from_numpy(features_np).float().unsqueeze(0)
        with torch.no_grad():
            danger_logit, l_target = self.forward(x)

        p = torch.sigmoid(
            danger_logit
        ).item()  # Keep this line as is (need_logit is a tensor)
        l_target = float(l_target)  # Convert delta_l to a float directly

        print(f"[NN Output] need_logit={danger_logit.item()}, l_target={l_target}")

        return p, l_target
