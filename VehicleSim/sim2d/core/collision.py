from __future__ import annotations

import math

import numpy as np

from sim2d.types import (
    BoxObstacle,
    CircleObstacle,
    FloatArray,
    Obstacle,
    VehicleConfig,
    VehicleState,
)


_EPS = 1e-9


def oriented_box_corners(
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
) -> FloatArray:
    """
    计算有向矩形的四个角点。

    返回角点顺序：
        front-left
        front-right
        rear-right
        rear-left

    坐标约定：
        yaw = 0 时，矩形长轴朝 +x
        yaw 正方向为逆时针

    返回：
        shape = (4, 2)
    """
    if length <= 0.0:
        raise ValueError(
            f"length must be positive, got {length}"
        )

    if width <= 0.0:
        raise ValueError(
            f"width must be positive, got {width}"
        )

    half_length = 0.5 * length
    half_width = 0.5 * width

    # 车辆局部坐标系中的四个角。
    local_corners = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    rotation = np.array(
        [
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ],
        dtype=np.float64,
    )

    center = np.array(
        [x, y],
        dtype=np.float64,
    )

    return local_corners @ rotation.T + center


def vehicle_corners(
    state: VehicleState,
    config: VehicleConfig,
) -> FloatArray:
    """
    根据车辆状态和车辆尺寸计算车身四个角点。
    """
    return oriented_box_corners(
        x=state.x,
        y=state.y,
        yaw=state.yaw,
        length=config.length,
        width=config.width,
    )


def _polygon_axes(corners: FloatArray) -> list[FloatArray]:
    """
    获取多边形各条边对应的单位法向量。

    对矩形而言，重复方向虽然存在，但不影响 SAT。
    """
    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    if corners.shape != (4, 2):
        raise ValueError(
            "corners must have shape (4, 2), "
            f"got {corners.shape}"
        )

    axes: list[FloatArray] = []

    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]

        edge = end - start
        normal = np.array(
            [-edge[1], edge[0]],
            dtype=np.float64,
        )

        norm = np.linalg.norm(normal)

        if norm <= _EPS:
            raise ValueError(
                "polygon contains a zero-length edge"
            )

        axes.append(normal / norm)

    return axes


def _project_polygon(
    corners: FloatArray,
    axis: FloatArray,
) -> tuple[float, float]:
    projections = corners @ axis

    return (
        float(np.min(projections)),
        float(np.max(projections)),
    )


def oriented_box_collision(
    corners_a: FloatArray,
    corners_b: FloatArray,
) -> bool:
    """
    使用分离轴定理 SAT 检测两个有向矩形是否相交。

    刚好接触也视为碰撞。
    """
    corners_a = np.asarray(
        corners_a,
        dtype=np.float64,
    )
    corners_b = np.asarray(
        corners_b,
        dtype=np.float64,
    )

    axes = (
        _polygon_axes(corners_a)
        + _polygon_axes(corners_b)
    )

    for axis in axes:
        min_a, max_a = _project_polygon(
            corners_a,
            axis,
        )
        min_b, max_b = _project_polygon(
            corners_b,
            axis,
        )

        if max_a < min_b - _EPS:
            return False

        if max_b < min_a - _EPS:
            return False

    return True


def _world_to_box_local(
    point_x: float,
    point_y: float,
    box_x: float,
    box_y: float,
    box_yaw: float,
) -> tuple[float, float]:
    """
    将世界坐标中的点转换到矩形局部坐标系。
    """
    dx = point_x - box_x
    dy = point_y - box_y

    cos_yaw = math.cos(box_yaw)
    sin_yaw = math.sin(box_yaw)

    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy

    return local_x, local_y


def circle_box_clearance(
    box_x: float,
    box_y: float,
    box_yaw: float,
    box_length: float,
    box_width: float,
    circle_x: float,
    circle_y: float,
    circle_radius: float,
) -> float:
    """
    计算圆与有向矩形之间的带符号间距。

    返回值：
        > 0：两者分离
        = 0：刚好接触
        < 0：发生穿透

    这里使用圆心到有向矩形的带符号距离，再减去圆半径。
    """
    if box_length <= 0.0:
        raise ValueError(
            "box_length must be positive"
        )

    if box_width <= 0.0:
        raise ValueError(
            "box_width must be positive"
        )

    if circle_radius <= 0.0:
        raise ValueError(
            "circle_radius must be positive"
        )

    local_x, local_y = _world_to_box_local(
        point_x=circle_x,
        point_y=circle_y,
        box_x=box_x,
        box_y=box_y,
        box_yaw=box_yaw,
    )

    half_length = 0.5 * box_length
    half_width = 0.5 * box_width

    distance_x = abs(local_x) - half_length
    distance_y = abs(local_y) - half_width

    outside_x = max(distance_x, 0.0)
    outside_y = max(distance_y, 0.0)

    outside_distance = math.hypot(
        outside_x,
        outside_y,
    )

    # 圆心在矩形内部时，该值为负数。
    inside_distance = min(
        max(distance_x, distance_y),
        0.0,
    )

    signed_point_box_distance = (
        outside_distance + inside_distance
    )

    return (
        signed_point_box_distance
        - circle_radius
    )


def circle_box_collision(
    box_x: float,
    box_y: float,
    box_yaw: float,
    box_length: float,
    box_width: float,
    circle_x: float,
    circle_y: float,
    circle_radius: float,
) -> bool:
    """
    检测圆与有向矩形是否碰撞。

    刚好接触也视为碰撞。
    """
    clearance = circle_box_clearance(
        box_x=box_x,
        box_y=box_y,
        box_yaw=box_yaw,
        box_length=box_length,
        box_width=box_width,
        circle_x=circle_x,
        circle_y=circle_y,
        circle_radius=circle_radius,
    )

    return clearance <= _EPS


def _point_segment_distance(
    point: FloatArray,
    segment_start: FloatArray,
    segment_end: FloatArray,
) -> float:
    segment = segment_end - segment_start
    segment_length_squared = float(
        np.dot(segment, segment)
    )

    if segment_length_squared <= _EPS:
        return float(
            np.linalg.norm(point - segment_start)
        )

    interpolation = float(
        np.dot(
            point - segment_start,
            segment,
        )
        / segment_length_squared
    )

    interpolation = max(
        0.0,
        min(1.0, interpolation),
    )

    closest_point = (
        segment_start
        + interpolation * segment
    )

    return float(
        np.linalg.norm(point - closest_point)
    )


def _polygon_distance(
    corners_a: FloatArray,
    corners_b: FloatArray,
) -> float:
    """
    计算两个不相交矩形边界之间的欧氏最小距离。
    """
    minimum = math.inf

    for point in corners_a:
        for index in range(4):
            start = corners_b[index]
            end = corners_b[(index + 1) % 4]

            minimum = min(
                minimum,
                _point_segment_distance(
                    point,
                    start,
                    end,
                ),
            )

    for point in corners_b:
        for index in range(4):
            start = corners_a[index]
            end = corners_a[(index + 1) % 4]

            minimum = min(
                minimum,
                _point_segment_distance(
                    point,
                    start,
                    end,
                ),
            )

    return float(minimum)


def oriented_box_clearance(
    corners_a: FloatArray,
    corners_b: FloatArray,
) -> float:
    """
    计算两个有向矩形之间的带符号距离。

    分离时：
        返回两个矩形边界的欧氏最小距离。

    相交时：
        返回负的 SAT 最小穿透深度。

    刚好接触时：
        返回 0。
    """
    corners_a = np.asarray(
        corners_a,
        dtype=np.float64,
    )
    corners_b = np.asarray(
        corners_b,
        dtype=np.float64,
    )

    axes = (
        _polygon_axes(corners_a)
        + _polygon_axes(corners_b)
    )

    minimum_overlap = math.inf
    separated = False

    for axis in axes:
        min_a, max_a = _project_polygon(
            corners_a,
            axis,
        )
        min_b, max_b = _project_polygon(
            corners_b,
            axis,
        )

        overlap = min(
            max_a,
            max_b,
        ) - max(
            min_a,
            min_b,
        )

        if overlap < -_EPS:
            separated = True
        else:
            minimum_overlap = min(
                minimum_overlap,
                max(overlap, 0.0),
            )

    if separated:
        return _polygon_distance(
            corners_a,
            corners_b,
        )

    if minimum_overlap is math.inf:
        return 0.0

    return -float(minimum_overlap)


def collision_with_obstacle(
    state: VehicleState,
    config: VehicleConfig,
    obstacle: Obstacle,
) -> bool:
    """
    检测车辆与指定障碍物是否碰撞。
    """
    if isinstance(obstacle, CircleObstacle):
        return circle_box_collision(
            box_x=state.x,
            box_y=state.y,
            box_yaw=state.yaw,
            box_length=config.length,
            box_width=config.width,
            circle_x=obstacle.x,
            circle_y=obstacle.y,
            circle_radius=obstacle.radius,
        )

    if isinstance(obstacle, BoxObstacle):
        ego_corners = vehicle_corners(
            state,
            config,
        )

        obstacle_corners = oriented_box_corners(
            x=obstacle.x,
            y=obstacle.y,
            yaw=obstacle.yaw,
            length=obstacle.length,
            width=obstacle.width,
        )

        return oriented_box_collision(
            ego_corners,
            obstacle_corners,
        )

    raise TypeError(
        "Unsupported obstacle type: "
        f"{type(obstacle).__name__}"
    )


def obstacle_clearance(
    state: VehicleState,
    config: VehicleConfig,
    obstacle: Obstacle,
) -> float:
    """
    计算车辆与一个障碍物的带符号间距。
    """
    if isinstance(obstacle, CircleObstacle):
        return circle_box_clearance(
            box_x=state.x,
            box_y=state.y,
            box_yaw=state.yaw,
            box_length=config.length,
            box_width=config.width,
            circle_x=obstacle.x,
            circle_y=obstacle.y,
            circle_radius=obstacle.radius,
        )

    if isinstance(obstacle, BoxObstacle):
        ego_corners = vehicle_corners(
            state,
            config,
        )

        obstacle_corners = oriented_box_corners(
            x=obstacle.x,
            y=obstacle.y,
            yaw=obstacle.yaw,
            length=obstacle.length,
            width=obstacle.width,
        )

        return oriented_box_clearance(
            ego_corners,
            obstacle_corners,
        )

    raise TypeError(
        "Unsupported obstacle type: "
        f"{type(obstacle).__name__}"
    )


def minimum_clearance(
    state: VehicleState,
    config: VehicleConfig,
    obstacles: tuple[Obstacle, ...],
) -> float | None:
    """
    计算车辆到所有障碍物的最小带符号间距。

    如果没有障碍物，返回 None。
    """
    if not obstacles:
        return None

    return min(
        obstacle_clearance(
            state=state,
            config=config,
            obstacle=obstacle,
        )
        for obstacle in obstacles
    )


def has_collision(
    state: VehicleState,
    config: VehicleConfig,
    obstacles: tuple[Obstacle, ...],
) -> bool:
    """
    检测车辆是否与任意障碍物碰撞。
    """
    return any(
        collision_with_obstacle(
            state=state,
            config=config,
            obstacle=obstacle,
        )
        for obstacle in obstacles
    )