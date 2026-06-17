import math
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt

PATH_POINT_STEP = 1.0
LINE_TYPE_VIRTUAL = "VIRTUAL"


def distance_xy(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def normalize_angle(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def interp_angle(a, b, w):
    return normalize_angle(a + normalize_angle(b - a) * w)


@dataclass
class Boundary:
    x: float = 0.0
    y: float = 0.0
    type: str = "UNKNOWN"
    color: str = "UNKNOWN"
    marking: str = "UNKNOWN"


@dataclass
class DrivePathPoint:
    x: float
    y: float
    yaw: float = 0.0
    s: float = 0.0
    k: float = 0.0
    source: str = "UNKNOWN"
    transition: int = 0
    boundary_left: Boundary = field(default_factory=Boundary)
    boundary_right: Boundary = field(default_factory=Boundary)


def copy_boundary(b: Boundary) -> Boundary:
    return Boundary(
        x=b.x,
        y=b.y,
        type=b.type,
        color=b.color,
        marking=b.marking,
    )


def copy_point(p: DrivePathPoint) -> DrivePathPoint:
    return DrivePathPoint(
        x=p.x,
        y=p.y,
        yaw=p.yaw,
        s=p.s,
        k=p.k,
        source=p.source,
        transition=p.transition,
        boundary_left=copy_boundary(p.boundary_left),
        boundary_right=copy_boundary(p.boundary_right),
    )


def linear_interpolation_point(
    p0: DrivePathPoint,
    p1: DrivePathPoint,
    weight: float,
) -> DrivePathPoint:
    return DrivePathPoint(
        x=p0.x * (1.0 - weight) + p1.x * weight,
        y=p0.y * (1.0 - weight) + p1.y * weight,
        yaw=interp_angle(p0.yaw, p1.yaw, weight),
        s=p0.s * (1.0 - weight) + p1.s * weight,
        k=p0.k * (1.0 - weight) + p1.k * weight,
        source=p0.source,
        transition=p1.transition,
        boundary_left=copy_boundary(p0.boundary_left),
        boundary_right=copy_boundary(p0.boundary_right),
    )


def boundary_interpolate(
    b0: Boundary,
    b1: Boundary,
    weight: float,
) -> Boundary:
    return Boundary(
        x=b0.x * (1.0 - weight) + b1.x * weight,
        y=b0.y * (1.0 - weight) + b1.y * weight,
        type=b0.type,
        color=b0.color,
        marking=b0.marking,
    )


class Line:
    def __init__(self, points: List[Tuple[float, float]] = None):
        self.points: List[DrivePathPoint] = []

        if points is not None:
            for x, y in points:
                self.points.append(
                    DrivePathPoint(
                        x=float(x),
                        y=float(y),
                    )
                )

            self.reassign_s()
            self.reassign_yaw()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def size(self):
        return len(self.points)

    def back(self):
        return self.points[-1]

    def emplace_back(self, pt: DrivePathPoint):
        self.points.append(pt)

    def reassign_s(self):
        if len(self.points) == 0:
            return

        self.points[0].s = 0.0

        for i in range(1, len(self.points)):
            pre = self.points[i - 1]
            cur = self.points[i]

            cur.s = pre.s + distance_xy(
                pre.x,
                pre.y,
                cur.x,
                cur.y,
            )

    def reassign_yaw(self):
        if len(self.points) < 2:
            return

        for i in range(len(self.points)):
            if i == 0:
                dx = self.points[i + 1].x - self.points[i].x
                dy = self.points[i + 1].y - self.points[i].y
            elif i == len(self.points) - 1:
                dx = self.points[i].x - self.points[i - 1].x
                dy = self.points[i].y - self.points[i - 1].y
            else:
                dx = 0.5 * (self.points[i + 1].x - self.points[i - 1].x)
                dy = 0.5 * (self.points[i + 1].y - self.points[i - 1].y)

            self.points[i].yaw = math.atan2(dy, dx)

    def print(self, name="line"):
        print(f"\n{name}, size = {len(self.points)}")
        for i, p in enumerate(self.points):
            print(
                f"[{i:02d}] "
                f"x={p.x:8.3f}, "
                f"y={p.y:8.3f}, "
                f"s={p.s:8.3f}, "
                f"yaw={p.yaw:8.3f}, "
                f"source={p.source}"
            )


class PathInstance:
    def __init__(self, line: Line = None):
        self.line = Line()

        if line is not None:
            for p in line.points:
                self.line.emplace_back(copy_point(p))

    def add(self, input_line: Line, source: str):
        for i in range(input_line.size()):
            path_pt = copy_point(input_line[i])
            path_pt.s = 0.0
            path_pt.source = source

            interpolated_pts: List[DrivePathPoint] = []

            if self.line.size() != 0:
                last_pt = self.line.back()

                tmp_dis = distance_xy(
                    last_pt.x,
                    last_pt.y,
                    path_pt.x,
                    path_pt.y,
                )

                if tmp_dis < 0.5:
                    continue

                if tmp_dis > PATH_POINT_STEP + 1.0:
                    inter_num = int(tmp_dis / PATH_POINT_STEP + 0.4) - 1

                    for k in range(inter_num):
                        delta_s = PATH_POINT_STEP * (k + 1)
                        weight = delta_s / tmp_dis

                        tmp_pt = linear_interpolation_point(
                            last_pt,
                            path_pt,
                            weight,
                        )

                        tmp_pt.boundary_left = boundary_interpolate(
                            last_pt.boundary_left,
                            path_pt.boundary_left,
                            weight,
                        )
                        tmp_pt.boundary_left.type = LINE_TYPE_VIRTUAL
                        tmp_pt.boundary_left.color = path_pt.boundary_left.color
                        tmp_pt.boundary_left.marking = path_pt.boundary_left.marking

                        tmp_pt.boundary_right = boundary_interpolate(
                            last_pt.boundary_right,
                            path_pt.boundary_right,
                            weight,
                        )
                        tmp_pt.boundary_right.type = LINE_TYPE_VIRTUAL
                        tmp_pt.boundary_right.color = path_pt.boundary_right.color
                        tmp_pt.boundary_right.marking = path_pt.boundary_right.marking

                        tmp_pt.s = last_pt.s + delta_s
                        tmp_pt.source = source
                        tmp_pt.transition = path_pt.transition

                        interpolated_pts.append(tmp_pt)

                    # 和 C++ 一样：用距离判断是否补末点
                    if len(interpolated_pts) != 0:
                        tail = interpolated_pts[-1]

                        if (
                            distance_xy(
                                tail.x,
                                tail.y,
                                path_pt.x,
                                path_pt.y,
                            )
                            > 0.1
                        ):
                            path_pt.s = last_pt.s + tmp_dis
                            interpolated_pts.append(path_pt)
                    else:
                        path_pt.s = last_pt.s + tmp_dis
                        interpolated_pts.append(path_pt)

                else:
                    path_pt.s = last_pt.s + tmp_dis
                    interpolated_pts.append(path_pt)

            else:
                print("this line size 0")
                interpolated_pts.append(path_pt)

            for pt in interpolated_pts:
                self.line.emplace_back(pt)


def plot_line(line: Line, label: str, marker: str):
    xs = [p.x for p in line.points]
    ys = [p.y for p in line.points]
    plt.plot(xs, ys, marker=marker, label=label)

    for i, p in enumerate(line.points):
        plt.text(
            p.x,
            p.y,
            f"{label[0]}{i}",
            fontsize=9,
        )


def plot_result_line(line: Line):
    xs = [p.x for p in line.points]
    ys = [p.y for p in line.points]

    plt.plot(xs, ys, marker=".", linewidth=2, label="after add")

    for i, p in enumerate(line.points):
        plt.text(
            p.x,
            p.y,
            f"R{i}\ns={p.s:.1f}",
            fontsize=8,
        )


def visualize(old_line: Line, input_line: Line, result_line: Line):
    plt.figure(figsize=(9, 7))

    plot_line(old_line, "old", "o")
    plot_line(input_line, "input", "x")
    plot_result_line(result_line)

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PathInstance::Add original logic simulation")
    plt.legend()
    plt.show()


def main():
    old_points = [
        (0, 0),
        (1, 0),
        (2, 0),
    ]

    input_points = [
        (1, 1),
        (2, 1),
        (3, 1),
    ]

    old_line = Line(old_points)
    input_line = Line(input_points)

    path_instance = PathInstance(old_line)
    path_instance.add(input_line, source="NEW_SOURCE")

    old_line.print("old_line")
    input_line.print("input_line")
    path_instance.line.print("after_add")

    visualize(
        old_line=old_line,
        input_line=input_line,
        result_line=path_instance.line,
    )


if __name__ == "__main__":
    main()
