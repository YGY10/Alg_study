from __future__ import annotations

from typing import List

import numpy as np
from torch.utils.data import Dataset

from data.line import Line, LineKind, SceneSample


class VectorNetToyDataset(Dataset):
    """
    临时 toy dataset。

    作用：
      只用于跑通 VectorNet pipeline。
      后面接 ArgoverseDataset 后，这个类仍然可以保留做 smoke test。

    每个 sample 包含：
      - 3 条车道线
      - 2 条路沿
      - 3 条 agent history
      - 目标 agent 的 future trajectory
    """

    def __init__(
        self,
        num_samples: int = 2000,
        history_steps: int = 6,
        future_steps: int = 5,
        dt: float = 0.5,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.lane_y_values = [-3.5, 0.0, 3.5]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        lines: List[Line] = []
        line_id = 0

        lane_x = np.linspace(-30.0, 60.0, 10, dtype=np.float32)

        # 1. Lane centerlines
        for lane_y in self.lane_y_values:
            points = np.stack(
                [lane_x, np.full_like(lane_x, lane_y)],
                axis=1,
            )

            lines.append(
                Line(
                    points=points,
                    kind=LineKind.LANE_LINE,
                    line_id=line_id,
                )
            )
            line_id += 1

        # 2. Curbs
        curb_y_values = [
            self.lane_y_values[0] - 3.5,
            self.lane_y_values[-1] + 3.5,
        ]

        for curb_y in curb_y_values:
            points = np.stack(
                [lane_x, np.full_like(lane_x, curb_y)],
                axis=1,
            )

            lines.append(
                Line(
                    points=points,
                    kind=LineKind.CURB,
                    line_id=line_id,
                )
            )
            line_id += 1

        # 3. Agent histories
        num_agents = 3
        target_agent_id = 0

        target_lane_y = None
        target_speed = None

        for agent_id in range(num_agents):
            lane_y = float(self.rng.choice(self.lane_y_values))
            speed = float(self.rng.uniform(4.0, 10.0))
            current_x = float(self.rng.uniform(-10.0, 10.0))

            if agent_id == target_agent_id:
                current_x = 0.0
                target_lane_y = lane_y
                target_speed = speed

            history_points = []

            for i in range(self.history_steps):
                t = -(self.history_steps - 1 - i) * self.dt
                x = current_x + speed * t
                y = lane_y + self.rng.normal(0.0, 0.05)
                history_points.append([x, y])

            lines.append(
                Line(
                    points=np.asarray(history_points, dtype=np.float32),
                    kind=LineKind.AGENT_HISTORY,
                    line_id=line_id,
                    agent_id=agent_id,
                    is_target=(agent_id == target_agent_id),
                )
            )
            line_id += 1

        # 4. Target future trajectory
        future = []

        for k in range(1, self.future_steps + 1):
            t = k * self.dt
            x = target_speed * t
            y = target_lane_y
            future.append([x, y])

        return SceneSample(
            lines=lines,
            target_agent_id=target_agent_id,
            future=np.asarray(future, dtype=np.float32),
        )
