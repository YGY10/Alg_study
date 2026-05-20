from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

STUDY_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "outputs" / "anchor_coverage"


@dataclass(frozen=True)
class VehicleSpec:
    length: float = 4.8
    width: float = 2.0
    safety_margin: float = 0.3


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    length: float
    width: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SparseDriveV2 path anchor coverage under simple static obstacle cases."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="SparseDriveV2 project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated coverage figures and stats.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "cone", "car", "large_vehicle", "construction"],
        help="Obstacle scenario to evaluate.",
    )

    parser.add_argument(
        "--x-min", type=float, default=8.0, help="Obstacle forward x min [m]."
    )
    parser.add_argument(
        "--x-max", type=float, default=50.0, help="Obstacle forward x max [m]."
    )
    parser.add_argument(
        "--x-num", type=int, default=43, help="Obstacle forward x samples."
    )

    parser.add_argument(
        "--y-min", type=float, default=-6.0, help="Obstacle lateral y min [m]."
    )
    parser.add_argument(
        "--y-max", type=float, default=6.0, help="Obstacle lateral y max [m]."
    )
    parser.add_argument(
        "--y-num", type=int, default=49, help="Obstacle lateral y samples."
    )

    parser.add_argument(
        "--ego-length", type=float, default=4.8, help="Ego vehicle length [m]."
    )
    parser.add_argument(
        "--ego-width", type=float, default=2.0, help="Ego vehicle width [m]."
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.3,
        help="Extra safety margin around inflated obstacle [m].",
    )

    parser.add_argument(
        "--max-case-plots",
        type=int,
        default=12,
        help="Max number of failure/challenging case plots per scenario.",
    )

    return parser.parse_args()


def load_path_vocab(project_root: Path) -> np.ndarray:
    anchor_dir = project_root / "ckpt" / "kmeans"
    path_file = anchor_dir / "path_1024.npy"

    if not path_file.exists():
        raise FileNotFoundError(f"Cannot find path anchor file: {path_file}")

    raw_path_vocab = np.load(path_file)

    if raw_path_vocab.ndim != 3:
        raise ValueError(
            "Expected path_vocab shape [num_paths, num_steps, dims], "
            f"but got {raw_path_vocab.shape}"
        )

    if raw_path_vocab.shape[-1] < 2:
        raise ValueError(
            "Expected path point dim >= 2 so we can use x/y, "
            f"but got {raw_path_vocab.shape}"
        )

    # SparseDriveV2 path_1024.npy is usually [N, T, 3].
    # We only need geometric centerline x/y for obstacle coverage analysis.
    # Convention:
    #   path[:, :, 0] = forward x
    #   path[:, :, 1] = lateral y
    #   path[:, :, 2] = extra field, likely heading/yaw, not used here.
    path_vocab_xy = raw_path_vocab[:, :, :2].astype(np.float64)

    print(
        f"[INFO] raw path_vocab shape: {raw_path_vocab.shape}, "
        f"use xy shape: {path_vocab_xy.shape}"
    )

    return path_vocab_xy


def get_obstacle_specs(scenario: str) -> list[ObstacleSpec]:
    specs = [
        ObstacleSpec(name="cone", length=0.5, width=0.5),
        ObstacleSpec(name="car", length=4.5, width=2.0),
        ObstacleSpec(name="large_vehicle", length=6.0, width=2.5),
        ObstacleSpec(name="construction", length=8.0, width=3.0),
    ]

    if scenario == "all":
        return specs

    return [spec for spec in specs if spec.name == scenario]


def setup_bev_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")
    ax.grid(True, linewidth=0.6, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")


def compute_path_ref_cost(path_vocab: np.ndarray) -> np.ndarray:
    """
    A simple reference-line deviation cost.

    Smaller value means the path stays closer to the nominal centerline.
    This is not SparseDriveV2's learned score. It is only used to select a
    visually reasonable safe path among all safe anchors.
    """
    y = path_vocab[:, :, 1]
    mean_abs_y = np.mean(np.abs(y), axis=1)
    max_abs_y = np.max(np.abs(y), axis=1)
    final_abs_y = np.abs(y[:, -1])

    return mean_abs_y + 0.2 * max_abs_y + 0.5 * final_abs_y


def path_clearance_to_inflated_box(
    path_vocab: np.ndarray,
    obs_x: float,
    obs_y: float,
    obs_spec: ObstacleSpec,
    vehicle_spec: VehicleSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check path centerline against an axis-aligned inflated obstacle box.

    Coordinate convention:
        path[:, :, 0] = forward x
        path[:, :, 1] = lateral y

    The obstacle is inflated by ego size and safety margin, so a path centerline
    entering the inflated rectangle is treated as collision.

    Returns:
        path_min_signed_clearance:
            shape [num_paths].
            > 0 means minimum centerline clearance to inflated box.
            < 0 means penetration into inflated box.
        path_collision:
            shape [num_paths], True if any path point is inside inflated box.
    """
    inflated_half_x = (
        0.5 * (obs_spec.length + vehicle_spec.length) + vehicle_spec.safety_margin
    )
    inflated_half_y = (
        0.5 * (obs_spec.width + vehicle_spec.width) + vehicle_spec.safety_margin
    )

    dx = np.abs(path_vocab[:, :, 0] - obs_x)
    dy = np.abs(path_vocab[:, :, 1] - obs_y)

    inside = (dx <= inflated_half_x) & (dy <= inflated_half_y)

    outside_x = np.maximum(dx - inflated_half_x, 0.0)
    outside_y = np.maximum(dy - inflated_half_y, 0.0)
    outside_dist = np.hypot(outside_x, outside_y)

    penetration_x = inflated_half_x - dx
    penetration_y = inflated_half_y - dy
    penetration = np.minimum(penetration_x, penetration_y)

    point_signed_clearance = np.where(inside, -penetration, outside_dist)

    path_min_signed_clearance = point_signed_clearance.min(axis=1)
    path_collision = inside.any(axis=1)

    return path_min_signed_clearance, path_collision


def evaluate_scenario(
    path_vocab: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    obs_spec: ObstacleSpec,
    vehicle_spec: VehicleSpec,
    path_ref_cost: np.ndarray,
) -> dict[str, np.ndarray]:
    nx = len(x_grid)
    ny = len(y_grid)

    safe_count = np.zeros((nx, ny), dtype=np.int32)
    safe_ratio = np.zeros((nx, ny), dtype=np.float64)
    safe_exists = np.zeros((nx, ny), dtype=bool)

    best_clearance = np.full((nx, ny), np.nan, dtype=np.float64)
    best_clearance_idx = np.full((nx, ny), -1, dtype=np.int32)

    best_ref_cost_idx = np.full((nx, ny), -1, dtype=np.int32)
    best_ref_cost_clearance = np.full((nx, ny), np.nan, dtype=np.float64)

    num_paths = path_vocab.shape[0]

    for ix, obs_x in enumerate(x_grid):
        for iy, obs_y in enumerate(y_grid):
            clearance, collision = path_clearance_to_inflated_box(
                path_vocab=path_vocab,
                obs_x=float(obs_x),
                obs_y=float(obs_y),
                obs_spec=obs_spec,
                vehicle_spec=vehicle_spec,
            )

            safe_mask = ~collision
            safe_indices = np.where(safe_mask)[0]
            count = int(safe_indices.size)

            safe_count[ix, iy] = count
            safe_ratio[ix, iy] = count / num_paths
            safe_exists[ix, iy] = count > 0

            if count > 0:
                safe_clearance = clearance[safe_indices]

                local_best_clearance_pos = int(np.argmax(safe_clearance))
                best_idx = int(safe_indices[local_best_clearance_pos])
                best_clearance_idx[ix, iy] = best_idx
                best_clearance[ix, iy] = float(clearance[best_idx])

                local_best_ref_pos = int(np.argmin(path_ref_cost[safe_indices]))
                ref_idx = int(safe_indices[local_best_ref_pos])
                best_ref_cost_idx[ix, iy] = ref_idx
                best_ref_cost_clearance[ix, iy] = float(clearance[ref_idx])

    return {
        "safe_count": safe_count,
        "safe_ratio": safe_ratio,
        "safe_exists": safe_exists,
        "best_clearance": best_clearance,
        "best_clearance_idx": best_clearance_idx,
        "best_ref_cost_idx": best_ref_cost_idx,
        "best_ref_cost_clearance": best_ref_cost_clearance,
    }


def plot_heatmap(
    output_path: Path,
    data: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[y_grid[0], y_grid[-1], x_grid[0], x_grid[-1]],
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("obstacle lateral y [m]")
    ax.set_ylabel("obstacle forward x [m]")
    ax.grid(False)

    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def add_obstacle_rectangles(
    ax: plt.Axes,
    obs_x: float,
    obs_y: float,
    obs_spec: ObstacleSpec,
    vehicle_spec: VehicleSpec,
) -> None:
    """
    Plot coordinates:
        horizontal axis = lateral y
        vertical axis = forward x
    """
    actual_rect = Rectangle(
        xy=(obs_y - 0.5 * obs_spec.width, obs_x - 0.5 * obs_spec.length),
        width=obs_spec.width,
        height=obs_spec.length,
        linewidth=1.8,
        edgecolor="red",
        facecolor="red",
        alpha=0.28,
        label="obstacle",
    )
    ax.add_patch(actual_rect)

    inflated_width = (
        obs_spec.width + vehicle_spec.width + 2.0 * vehicle_spec.safety_margin
    )
    inflated_length = (
        obs_spec.length + vehicle_spec.length + 2.0 * vehicle_spec.safety_margin
    )

    inflated_rect = Rectangle(
        xy=(obs_y - 0.5 * inflated_width, obs_x - 0.5 * inflated_length),
        width=inflated_width,
        height=inflated_length,
        linewidth=1.6,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        alpha=0.85,
        label="inflated obstacle",
    )
    ax.add_patch(inflated_rect)


def plot_single_case(
    output_path: Path,
    path_vocab: np.ndarray,
    obs_x: float,
    obs_y: float,
    obs_spec: ObstacleSpec,
    vehicle_spec: VehicleSpec,
    path_ref_cost: np.ndarray,
    title: str,
) -> None:
    clearance, collision = path_clearance_to_inflated_box(
        path_vocab=path_vocab,
        obs_x=obs_x,
        obs_y=obs_y,
        obs_spec=obs_spec,
        vehicle_spec=vehicle_spec,
    )

    safe_mask = ~collision
    safe_indices = np.where(safe_mask)[0]
    unsafe_indices = np.where(collision)[0]

    fig, ax = plt.subplots(figsize=(8, 9))

    for idx in unsafe_indices:
        path = path_vocab[idx]
        ax.plot(path[:, 1], path[:, 0], color="0.75", alpha=0.08, linewidth=0.7)

    for idx in safe_indices:
        path = path_vocab[idx]
        ax.plot(path[:, 1], path[:, 0], color="#1f77b4", alpha=0.12, linewidth=0.8)

    if safe_indices.size > 0:
        best_clearance_idx = int(safe_indices[np.argmax(clearance[safe_indices])])
        best_ref_idx = int(safe_indices[np.argmin(path_ref_cost[safe_indices])])

        best_clearance_path = path_vocab[best_clearance_idx]
        best_ref_path = path_vocab[best_ref_idx]

        ax.plot(
            best_clearance_path[:, 1],
            best_clearance_path[:, 0],
            color="#ff7f0e",
            linewidth=2.4,
            label=f"max clearance safe path {best_clearance_idx}",
        )
        ax.plot(
            best_ref_path[:, 1],
            best_ref_path[:, 0],
            color="#2ca02c",
            linewidth=2.4,
            label=f"min ref-cost safe path {best_ref_idx}",
        )

    add_obstacle_rectangles(
        ax=ax,
        obs_x=obs_x,
        obs_y=obs_y,
        obs_spec=obs_spec,
        vehicle_spec=vehicle_spec,
    )

    setup_bev_axis(ax, title)

    x_min = np.percentile(path_vocab[:, :, 0], 0.2)
    x_max = np.percentile(path_vocab[:, :, 0], 99.8)
    y_min = np.percentile(path_vocab[:, :, 1], 0.2)
    y_max = np.percentile(path_vocab[:, :, 1], 99.8)

    ax.set_xlim(y_min - 1.0, y_max + 1.0)
    ax.set_ylim(max(-2.0, x_min - 2.0), x_max + 2.0)

    ax.text(
        0.02,
        0.98,
        (
            f"obs: x={obs_x:.1f}, y={obs_y:.1f}\n"
            f"safe paths: {safe_indices.size}/{path_vocab.shape[0]}"
        ),
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def pick_evenly_spaced_cases(case_indices: np.ndarray, max_count: int) -> np.ndarray:
    if case_indices.size == 0:
        return case_indices

    if len(case_indices) <= max_count:
        return case_indices

    positions = np.linspace(0, len(case_indices) - 1, max_count).round().astype(int)
    return case_indices[positions]


def plot_case_examples(
    output_dir: Path,
    path_vocab: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    obs_spec: ObstacleSpec,
    vehicle_spec: VehicleSpec,
    path_ref_cost: np.ndarray,
    result: dict[str, np.ndarray],
    max_case_plots: int,
) -> None:
    case_dir = output_dir / obs_spec.name / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    safe_exists = result["safe_exists"]
    safe_count = result["safe_count"]

    failure_indices = np.argwhere(~safe_exists)
    selected_failures = pick_evenly_spaced_cases(failure_indices, max_case_plots)

    for k, (ix, iy) in enumerate(selected_failures):
        obs_x = float(x_grid[ix])
        obs_y = float(y_grid[iy])
        plot_single_case(
            output_path=case_dir / f"failure_{k:02d}_x{obs_x:.1f}_y{obs_y:.1f}.png",
            path_vocab=path_vocab,
            obs_x=obs_x,
            obs_y=obs_y,
            obs_spec=obs_spec,
            vehicle_spec=vehicle_spec,
            path_ref_cost=path_ref_cost,
            title=f"{obs_spec.name} failure case: no safe path",
        )

    challenging_mask = safe_exists & (safe_count > 0)
    challenging_indices = np.argwhere(challenging_mask)

    if challenging_indices.size > 0:
        counts = safe_count[challenging_mask]
        order = np.argsort(counts)
        challenging_indices = challenging_indices[order]
        selected_challenging = challenging_indices[:max_case_plots]

        for k, (ix, iy) in enumerate(selected_challenging):
            obs_x = float(x_grid[ix])
            obs_y = float(y_grid[iy])
            plot_single_case(
                output_path=case_dir
                / f"challenging_{k:02d}_x{obs_x:.1f}_y{obs_y:.1f}_safe{safe_count[ix, iy]}.png",
                path_vocab=path_vocab,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_spec=obs_spec,
                vehicle_spec=vehicle_spec,
                path_ref_cost=path_ref_cost,
                title=f"{obs_spec.name} challenging case: few safe paths",
            )


def save_scenario_outputs(
    output_dir: Path,
    path_vocab: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    obs_spec: ObstacleSpec,
    vehicle_spec: VehicleSpec,
    result: dict[str, np.ndarray],
    path_ref_cost: np.ndarray,
    max_case_plots: int,
) -> dict[str, object]:
    scenario_dir = output_dir / obs_spec.name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        scenario_dir / "coverage_arrays.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        safe_count=result["safe_count"],
        safe_ratio=result["safe_ratio"],
        safe_exists=result["safe_exists"],
        best_clearance=result["best_clearance"],
        best_clearance_idx=result["best_clearance_idx"],
        best_ref_cost_idx=result["best_ref_cost_idx"],
        best_ref_cost_clearance=result["best_ref_cost_clearance"],
    )

    num_cells = result["safe_exists"].size
    num_failure = int((~result["safe_exists"]).sum())
    num_success = int(result["safe_exists"].sum())

    summary = {
        "obstacle": asdict(obs_spec),
        "vehicle": asdict(vehicle_spec),
        "grid": {
            "x_min": float(x_grid[0]),
            "x_max": float(x_grid[-1]),
            "x_num": int(len(x_grid)),
            "y_min": float(y_grid[0]),
            "y_max": float(y_grid[-1]),
            "y_num": int(len(y_grid)),
            "num_cells": int(num_cells),
        },
        "path_shape": list(path_vocab.shape),
        "num_paths": int(path_vocab.shape[0]),
        "success_cells": num_success,
        "failure_cells": num_failure,
        "success_ratio": float(num_success / num_cells),
        "failure_ratio": float(num_failure / num_cells),
        "safe_count_min": int(result["safe_count"].min()),
        "safe_count_max": int(result["safe_count"].max()),
        "safe_count_mean": float(result["safe_count"].mean()),
    }

    valid_clearance = result["best_clearance"][np.isfinite(result["best_clearance"])]
    if valid_clearance.size > 0:
        summary["best_clearance_min_median_max"] = [
            float(valid_clearance.min()),
            float(np.median(valid_clearance)),
            float(valid_clearance.max()),
        ]
    else:
        summary["best_clearance_min_median_max"] = [None, None, None]

    with (scenario_dir / "coverage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"Scenario: {obs_spec.name}",
        "",
        f"Obstacle length/width: {obs_spec.length:.3f} / {obs_spec.width:.3f} m",
        (
            "Ego length/width/margin: "
            f"{vehicle_spec.length:.3f} / {vehicle_spec.width:.3f} / "
            f"{vehicle_spec.safety_margin:.3f} m"
        ),
        f"path_vocab: {path_vocab.shape}",
        "",
        f"grid cells: {num_cells}",
        f"success cells: {num_success} ({num_success / num_cells:.3f})",
        f"failure cells: {num_failure} ({num_failure / num_cells:.3f})",
        "",
        f"safe count min/mean/max: "
        f"{result['safe_count'].min()} / {result['safe_count'].mean():.2f} / "
        f"{result['safe_count'].max()}",
    ]

    if valid_clearance.size > 0:
        lines.extend(
            [
                "best clearance min/median/max: "
                f"{valid_clearance.min():.3f} / "
                f"{np.median(valid_clearance):.3f} / "
                f"{valid_clearance.max():.3f} m"
            ]
        )

    (scenario_dir / "coverage_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    plot_heatmap(
        output_path=scenario_dir / "safe_count_heatmap.png",
        data=result["safe_count"],
        x_grid=x_grid,
        y_grid=y_grid,
        title=f"{obs_spec.name}: number of safe path anchors",
        colorbar_label="safe path count",
        cmap="viridis",
        vmin=0,
        vmax=float(path_vocab.shape[0]),
    )

    plot_heatmap(
        output_path=scenario_dir / "safe_exists_heatmap.png",
        data=result["safe_exists"].astype(float),
        x_grid=x_grid,
        y_grid=y_grid,
        title=f"{obs_spec.name}: safe path exists",
        colorbar_label="1 = exists, 0 = no safe path",
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )

    clearance_for_plot = result["best_clearance"].copy()
    clearance_for_plot[~np.isfinite(clearance_for_plot)] = np.nan

    plot_heatmap(
        output_path=scenario_dir / "best_clearance_heatmap.png",
        data=clearance_for_plot,
        x_grid=x_grid,
        y_grid=y_grid,
        title=f"{obs_spec.name}: max clearance among safe anchors",
        colorbar_label="best safe clearance to inflated box [m]",
        cmap="magma",
    )

    plot_case_examples(
        output_dir=output_dir,
        path_vocab=path_vocab,
        x_grid=x_grid,
        y_grid=y_grid,
        obs_spec=obs_spec,
        vehicle_spec=vehicle_spec,
        path_ref_cost=path_ref_cost,
        result=result,
        max_case_plots=max_case_plots,
    )

    return summary


def save_overall_summary(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    with (output_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    lines = ["Anchor coverage overall summary", ""]

    for summary in summaries:
        obstacle = summary["obstacle"]
        lines.extend(
            [
                f"Scenario: {obstacle['name']}",
                f"  obstacle length/width: {obstacle['length']:.3f} / {obstacle['width']:.3f} m",
                f"  success ratio: {summary['success_ratio']:.3f}",
                f"  failure ratio: {summary['failure_ratio']:.3f}",
                (
                    "  safe count min/mean/max: "
                    f"{summary['safe_count_min']} / "
                    f"{summary['safe_count_mean']:.2f} / "
                    f"{summary['safe_count_max']}"
                ),
                "",
            ]
        )

    (output_dir / "overall_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle_spec = VehicleSpec(
        length=float(args.ego_length),
        width=float(args.ego_width),
        safety_margin=float(args.safety_margin),
    )

    path_vocab = load_path_vocab(args.project_root.resolve())
    path_ref_cost = compute_path_ref_cost(path_vocab)

    x_grid = np.linspace(float(args.x_min), float(args.x_max), int(args.x_num))
    y_grid = np.linspace(float(args.y_min), float(args.y_max), int(args.y_num))

    obstacle_specs = get_obstacle_specs(args.scenario)

    summaries: list[dict[str, object]] = []

    print(f"[INFO] loaded path_vocab: {path_vocab.shape}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] x_grid: {x_grid[0]:.2f} -> {x_grid[-1]:.2f}, num={len(x_grid)}")
    print(f"[INFO] y_grid: {y_grid[0]:.2f} -> {y_grid[-1]:.2f}, num={len(y_grid)}")

    for obs_spec in obstacle_specs:
        print(
            f"[INFO] evaluating scenario={obs_spec.name}, "
            f"length={obs_spec.length:.2f}, width={obs_spec.width:.2f}"
        )

        result = evaluate_scenario(
            path_vocab=path_vocab,
            x_grid=x_grid,
            y_grid=y_grid,
            obs_spec=obs_spec,
            vehicle_spec=vehicle_spec,
            path_ref_cost=path_ref_cost,
        )

        summary = save_scenario_outputs(
            output_dir=output_dir,
            path_vocab=path_vocab,
            x_grid=x_grid,
            y_grid=y_grid,
            obs_spec=obs_spec,
            vehicle_spec=vehicle_spec,
            result=result,
            path_ref_cost=path_ref_cost,
            max_case_plots=int(args.max_case_plots),
        )

        summaries.append(summary)

        print(
            f"[INFO] {obs_spec.name}: "
            f"success_ratio={summary['success_ratio']:.3f}, "
            f"failure_ratio={summary['failure_ratio']:.3f}, "
            f"safe_count_mean={summary['safe_count_mean']:.2f}"
        )

    save_overall_summary(output_dir, summaries)

    print(f"[INFO] saved coverage analysis to: {output_dir}")


if __name__ == "__main__":
    main()
