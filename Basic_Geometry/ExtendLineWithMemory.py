import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import List


# ================= 数据结构 =================
@dataclass
class LinePoint:
    x: float
    y: float
    yaw: float
    k: float
    s: float


class Line(List[LinePoint]):
    def GenerateAccumulatedS(self):
        self.accumulated_s = [pt.s for pt in self]

    def ReassignYaw(self):
        if len(self) < 2:
            return
        for i in range(len(self)):
            if i == 0:
                dx = self[i + 1].x - self[i].x
                dy = self[i + 1].y - self[i].y
            elif i == len(self) - 1:
                dx = self[i].x - self[i - 1].x
                dy = self[i].y - self[i - 1].y
            else:
                dx = 0.5 * (self[i + 1].x - self[i - 1].x)
                dy = 0.5 * (self[i + 1].y - self[i - 1].y)
            self[i].yaw = math.atan2(dy, dx)


# ================= 工具函数 =================
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


before_points = [
    LinePoint(-7.54199, -2.94883, 0.0007558, math.nan, -8.33334),
    LinePoint(-4.54199, -2.94143, 0.162951, math.nan, -5.33333),
    LinePoint(-2.70361, -2.15382, 0.405029, math.nan, -3.33333),
    LinePoint(-1.54199, -1.65615, 0.405029, math.nan, -2.0696),
]


n_before = len(before_points)
bx = [pt.x for pt in before_points]
by = [pt.y for pt in before_points]
plt.figure(figsize=(8, 8))
plt.plot(bx, by, "o-", label="Before (C++ input)", linewidth=1)
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("ExtendLineWithCurve: Geometry-consistent fix")
plt.show()
