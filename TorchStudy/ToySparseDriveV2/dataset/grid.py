from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridConfig:
    x_min: float = -50.0
    x_max: float = 65.0
    y_min: float = -60.0
    y_max: float = 60.0
    height: int = 128
    width: int = 128

    @property
    def x_resolution(self) -> float:
        return (self.x_max - self.x_min) / self.height

    @property
    def y_resolution(self) -> float:
        return (self.y_max - self.y_min) / self.width


def make_empty_grid(
    channels: int,
    config: GridConfig = GridConfig(),
) -> np.array:
    return np.zeros(
        (channels, config.height, config.width),
        dtype=np.float32,
    )


def world_to_grid(
    xy: np.ndarray,
    config: GridConfig = GridConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float32)
    x = xy[..., 0]
    y = xy[..., 1]
    row = (config.x_max - x) / (config.x_max - config.x_min) * (config.height - 1)
    col = (config.y_max - y) / (config.y_max - config.y_min) * (config.width - 1)

    row_int = np.rint(row).astype(np.int64)
    col_int = np.rint(col).astype(np.int64)

    valid = (
        (row_int >= 0)
        & (row_int < config.height)
        & (col_int >= 0)
        & (col_int < config.width)
    )
    return row_int, col_int, valid


def grid_to_world(
    row: np.ndarray,
    col: np.ndarray,
    config: GridConfig = GridConfig(),
) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32)
    col = np.asarray(col, dtype=np.float32)
    x = config.x_max - row / (config.height - 1) * (config.x_max - config.x_min)
    y = config.y_max - col / (config.width - 1) * (config.y_max - config.y_min)

    return np.stack([x, y], axis=-1)


def draw_points(
    grid: np.ndarray,
    xy: np.ndarray,
    value: float = 1.0,
    radius: int = 1,
    config: GridConfig = GridConfig(),
) -> np.ndarray:
    row, col, valid = world_to_grid(xy, config)

    row = row[valid]
    col = col[valid]

    for r, c in zip(row, col):
        r0 = max(0, r - radius)
        r1 = min(config.height, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(config.width, c + radius + 1)
        grid[r0:r1, c0:c1] = value

    return grid


def sample_polyline_points(
    xy: np.ndarray,
    samples_per_segment: int = 8,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) <= 1:
        return xy

    sampled = []
    for start, end in zip(xy[:-1], xy[1:]):
        alpha = np.linspace(
            0.0,
            1.0,
            samples_per_segment,
            endpoint=False,
            dtype=np.float32,
        )
        points = start[None] * (1.0 - alpha[:, None]) + end[None] * alpha[:, None]
        sampled.append(points)

    sampled.append(xy[-1:])
    return np.concatenate(sampled, axis=0)


def draw_polyline(
    grid: np.ndarray,
    xy: np.ndarray,
    value: float = 1.0,
    radius: int = 1,
    samples_per_segment: int = 8,
    config: GridConfig = GridConfig(),
) -> np.ndarray:
    points = sample_polyline_points(
        xy,
        samples_per_segment=samples_per_segment,
    )
    return draw_points(
        grid,
        points,
        value=value,
        radius=radius,
        config=config,
    )


def draw_rectangle(
    grid: np.ndarray,
    center_xy: np.ndarray,
    size_xy: tuple[float, float],
    value: float = 1.0,
    config: GridConfig = GridConfig(),
) -> np.ndarray:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    length_x, width_y = size_xy

    xs = np.arange(
        center_xy[0] - length_x / 2.0,
        center_xy[0] + length_x / 2.0,
        config.x_resolution,
        dtype=np.float32,
    )
    ys = np.arange(
        center_xy[1] - width_y / 2.0,
        center_xy[1] + width_y / 2.0,
        config.y_resolution,
        dtype=np.float32,
    )

    if len(xs) == 0 or len(ys) == 0:
        return grid

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    return draw_points(
        grid,
        points,
        value=value,
        radius=0,
        config=config,
    )


if __name__ == "__main__":
    config = GridConfig()

    xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 5.0],
            [50.0, -20.0],
        ],
        dtype=np.float32,
    )

    row, col, valid = world_to_grid(xy, config)
    restored_xy = grid_to_world(row, col, config)

    print("xy:")
    print(xy)
    print("row:", row)
    print("col:", col)
    print("valid:", valid)
    print("restored_xy:")
    print(restored_xy)
