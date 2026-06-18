from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "vocab",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import TIME_STEPS, ToySparseDriveV2Dataset
from teacher import (
    EgoState,
    TeacherConfig,
    normalize_obstacles,
    score_teacher_candidates,
    score_trajectories,
)
from grid import GridConfig, draw_points, draw_polyline, draw_rectangle, make_empty_grid
from model import ToySparseDriveV2Model
from vocab import SparseDriveVocab

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_random"

# Edit this dict to test custom ego state.
# xy/yaw are kept for Teacher scoring. The current model was trained in ego frame,
# so changing speed/size is meaningful immediately; changing xy/yaw is mainly for
# future simulator work unless the model input is extended too.
CUSTOM_EGO_STATE = {
    "xy": (0.0, 0.0),
    "yaw": 0.0,
    "speed": 0.0,
    "size_xy": (4.8, 2.0),
}

# Set to None to use the random sample route endpoint.
# Otherwise use [x_forward, y_left] in meters.
CUSTOM_GOAL_XY = (45.0, 0.0)

# Edit this list to test custom dynamic obstacles.
# center_xy: [x_forward, y_left] in meters.
# size_xy: [length_along_x, width_along_y] in meters.
# velocity_xy: [vx_forward, vy_left] in meters/second.
# Set to [] to keep the randomly generated dataset obstacles.
CUSTOM_OBSTACLES = [
    {
        "center_xy": (35.0, 2.0),
        "size_xy": (5.0, 2.5),
        "velocity_xy": (1.0, 0.0),
    },
    {
        "center_xy": (20.0, -4.0),
        "size_xy": (5.0, 2.5),
        "velocity_xy": (3.0, 0.0),
    },
    # {
    #     "center_xy": (10.0, 4.0),
    #     "size_xy": (5.0, 2.5),
    #     "velocity_xy": (1.0, 0.0),
    # },
    # {
    #     "center_xy": (15.0, -6.0),
    #     "size_xy": (4.5, 2.2),
    #     "velocity_xy": (1.0, 0.0),
    # },
    # {
    #     "center_xy": (15.0, -3.0),
    #     "size_xy": (4.5, 2.2),
    #     "velocity_xy": (1.0, 0.0),
    # },
    # {
    #     "center_xy": (15.0, 0.0),
    #     "size_xy": (5.0, 2.5),
    #     "velocity_xy": (1.0, 0.0),
    # },
    {
        "center_xy": (15.0, 2.0),
        "size_xy": (5.0, 2.5),
        "velocity_xy": (1.0, 0.0),
    },
    {
        "center_xy": (15.0, 6.0),
        "size_xy": (5.0, 2.5),
        "velocity_xy": (1.0, 0.0),
    },
]


def make_custom_ego_state() -> EgoState:
    return EgoState(
        xy=tuple(CUSTOM_EGO_STATE["xy"]),
        yaw=float(CUSTOM_EGO_STATE["yaw"]),
        speed=float(CUSTOM_EGO_STATE["speed"]),
        size_xy=tuple(CUSTOM_EGO_STATE["size_xy"]),
    )


def make_ego_state_tensor() -> torch.Tensor:
    return torch.tensor(
        [
            float(CUSTOM_EGO_STATE["speed"]),
            float(CUSTOM_EGO_STATE["size_xy"][0]),
            float(CUSTOM_EGO_STATE["size_xy"][1]),
        ],
        dtype=torch.float32,
    )


def make_teacher_config() -> TeacherConfig:
    return TeacherConfig(
        num_path_candidates=8,
        num_top_trajectories=8,
        temperature=2.0,
    )


def sample_obstacles_to_dicts(
    sample: dict[str, torch.Tensor],
) -> list[dict[str, tuple[float, float]]]:
    centers = sample["obstacle_centers"].numpy()
    sizes = sample["obstacle_sizes"].numpy()
    velocities = sample["obstacle_velocities"].numpy()
    mask = sample["obstacle_mask"].numpy()

    obstacles = []
    for i in np.where(mask > 0.5)[0]:
        obstacles.append(
            {
                "center_xy": (float(centers[i, 0]), float(centers[i, 1])),
                "size_xy": (float(sizes[i, 0]), float(sizes[i, 1])),
                "velocity_xy": (float(velocities[i, 0]), float(velocities[i, 1])),
            }
        )
    return obstacles


def make_route_to_goal(
    goal_xy: np.ndarray,
    num_points: int = 50,
) -> np.ndarray:
    start_xy = np.array([0.0, 0.0], dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    route_xy = start_xy[None] * (1.0 - alpha[:, None]) + goal_xy[None] * alpha[:, None]
    yaw = np.zeros((num_points, 1), dtype=np.float32)
    return np.concatenate([route_xy, yaw], axis=-1)


def apply_custom_goal(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if CUSTOM_GOAL_XY is None:
        return sample

    goal_xy = np.asarray(CUSTOM_GOAL_XY, dtype=np.float32)
    route_path = make_route_to_goal(goal_xy)
    sample["goal_xy"] = torch.from_numpy(goal_xy).float()
    sample["route_path"] = torch.from_numpy(route_path).float()
    return sample


def draw_rectangle_world(
    ax: plt.Axes,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    size_xy = np.asarray(size_xy, dtype=np.float32)

    x = float(center_xy[0])
    y = float(center_xy[1])
    length_x = float(size_xy[0])
    width_y = float(size_xy[1])

    rect = plt.Rectangle(
        (y - width_y / 2.0, x - length_x / 2.0),
        width_y,
        length_x,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.8,
    )
    ax.add_patch(rect)


def setup_axis(ax: plt.Axes, config: GridConfig, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.set_xlim(config.y_max, config.y_min)
    ax.set_ylim(config.x_min, config.x_max)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def select_prediction(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    trajectory_scores = output["trajectory_scores"]
    best_candidate = trajectory_scores.argmax(dim=-1)
    batch_indices = torch.arange(
        trajectory_scores.shape[0],
        device=trajectory_scores.device,
    )

    return {
        "trajectory": output["candidate_trajectories"][batch_indices, best_candidate],
        "path_index": output["candidate_path_indices"][batch_indices, best_candidate],
        "velocity_index": output["candidate_velocity_indices"][
            batch_indices,
            best_candidate,
        ],
        "candidate_index": best_candidate,
    }


def compute_errors(
    pred_trajectory: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[float, float]:
    mask = target_mask.float()
    diff = pred_trajectory[..., :2] - target_trajectory[..., :2]
    point_l2 = diff.pow(2).sum(dim=-1).sqrt()
    traj_l2 = (point_l2 * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    valid_count = mask.sum(dim=-1).long().clamp(min=1)
    endpoint_index = valid_count - 1
    batch_indices = torch.arange(
        pred_trajectory.shape[0], device=pred_trajectory.device
    )
    target_endpoint = target_trajectory[batch_indices, endpoint_index, :2]
    pred_endpoint = pred_trajectory[batch_indices, endpoint_index, :2]
    endpoint_l2 = (pred_endpoint - target_endpoint).pow(2).sum(dim=-1).sqrt()

    return float(traj_l2.mean().cpu()), float(endpoint_l2.mean().cpu())


def override_obstacles(
    sample: dict[str, torch.Tensor],
    obstacles: list[dict[str, tuple[float, float]]],
) -> dict[str, torch.Tensor]:
    config = GridConfig()
    max_obstacles = sample["obstacle_mask"].shape[0]

    centers = torch.zeros_like(sample["obstacle_centers"])
    sizes = torch.zeros_like(sample["obstacle_sizes"])
    velocities = torch.zeros_like(sample["obstacle_velocities"])
    mask = torch.zeros_like(sample["obstacle_mask"])
    input_grid = make_empty_grid(channels=4, config=config)

    for i, obstacle in enumerate(obstacles[:max_obstacles]):
        center_xy = np.array(obstacle["center_xy"], dtype=np.float32)
        size_xy = tuple(obstacle["size_xy"])
        velocity_xy = np.array(obstacle["velocity_xy"], dtype=np.float32)

        centers[i] = torch.tensor(center_xy, dtype=torch.float32)
        sizes[i] = torch.tensor(size_xy, dtype=torch.float32)
        velocities[i] = torch.tensor(velocity_xy, dtype=torch.float32)
        mask[i] = 1.0

        draw_rectangle(
            input_grid[0],
            center_xy,
            size_xy,
            value=1.0,
            config=config,
        )
        for time_s in TIME_STEPS:
            future_center = center_xy + velocity_xy * float(time_s)
            draw_rectangle(
                input_grid[1],
                future_center,
                size_xy,
                value=1.0,
                config=config,
            )

    route_path = sample.get("route_path", sample["target_path"]).numpy()
    draw_polyline(
        input_grid[2],
        route_path[:, :2],
        value=1.0,
        radius=1,
        samples_per_segment=4,
        config=config,
    )

    goal_xy = sample.get("goal_xy", sample["target_trajectory"][-1, :2]).numpy()[None]
    draw_points(
        input_grid[3],
        goal_xy,
        value=1.0,
        radius=3,
        config=config,
    )

    sample["obstacle_centers"] = centers
    sample["obstacle_sizes"] = sizes
    sample["obstacle_velocities"] = velocities
    sample["obstacle_mask"] = mask
    sample["input_grid"] = torch.from_numpy(input_grid).float()

    return sample


def select_teacher_reference(
    vocab: SparseDriveVocab,
    sample: dict[str, torch.Tensor],
    obstacles: list[dict[str, tuple[float, float]]],
    ego_state: EgoState,
) -> dict[str, Any]:
    teacher_config = make_teacher_config()
    teacher_output = score_teacher_candidates(
        vocab=vocab,
        goal_xy=sample["goal_xy"].numpy(),
        obstacles=obstacles,
        ego_state=ego_state,
        route_path=sample["route_path"].numpy()[:, :2],
        config=teacher_config,
    )

    best = teacher_output.best_candidate_index
    return {
        "trajectory": teacher_output.debug["trajectories"][best],
        "trajectory_mask": teacher_output.debug["masks"][best],
        "path_index": teacher_output.best_path_index,
        "velocity_index": teacher_output.best_velocity_index,
        "flat_index": teacher_output.best_flat_index,
        "collides": bool(teacher_output.debug["collision"][best]),
        "cost": float(teacher_output.candidate_costs[best]),
        "goal_cost": float(teacher_output.debug["goal_cost"][best]),
        "route_cost": float(teacher_output.debug["route_cost"][best]),
        "clearance": float(teacher_output.debug["clearance"][best]),
        "path_costs": teacher_output.path_costs,
        "candidate_flat_indices": teacher_output.candidate_flat_indices,
        "candidate_costs": teacher_output.candidate_costs,
        "candidate_path_indices": teacher_output.candidate_path_indices,
        "candidate_velocity_indices": teacher_output.candidate_velocity_indices,
    }


def compute_rank_ascending(values: np.ndarray, index: int) -> int:
    order = np.argsort(values)
    return int(np.where(order == index)[0][0]) + 1


def compute_rank_descending(values: np.ndarray, index: int) -> int:
    order = np.argsort(-values)
    return int(np.where(order == index)[0][0]) + 1


def optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_candidate_diagnostics(
    vocab: SparseDriveVocab,
    sample: dict[str, torch.Tensor],
    obstacles: list[dict[str, tuple[float, float]]],
    ego_state: EgoState,
    output: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    reference: dict[str, Any],
) -> dict[str, Any]:
    pred_path_index = int(prediction["path_index"][0].cpu())
    pred_velocity_index = int(prediction["velocity_index"][0].cpu())
    pred_candidate_index = int(prediction["candidate_index"][0].cpu())
    pred_flat_index = pred_path_index * vocab.num_velocity + pred_velocity_index

    config = make_teacher_config()
    scored_pred_path = score_trajectories(
        vocab=vocab,
        path_indices=np.array([pred_path_index], dtype=np.int64),
        goal_xy=sample["goal_xy"].numpy(),
        obstacles=normalize_obstacles(obstacles),
        ego_state=ego_state,
        route_path=sample["route_path"].numpy()[:, :2],
        config=config,
    )
    pred_local_index = int(
        np.where(scored_pred_path["candidate_velocity_indices"] == pred_velocity_index)[
            0
        ][0]
    )

    candidate_flat_indices = reference["candidate_flat_indices"]
    candidate_costs = reference["candidate_costs"]
    matched_teacher_candidate = np.where(candidate_flat_indices == pred_flat_index)[0]
    in_teacher_candidate_set = len(matched_teacher_candidate) > 0
    teacher_candidate_rank = None
    if in_teacher_candidate_set:
        candidate_order = np.argsort(candidate_costs)
        teacher_candidate_rank = (
            int(np.where(candidate_order == matched_teacher_candidate[0])[0][0]) + 1
        )

    path_costs = reference["path_costs"]
    path_scores = output["path_scores"][0].detach().cpu().numpy()
    velocity_scores = output["velocity_scores"][0].detach().cpu().numpy()
    trajectory_scores = output["trajectory_scores"][0].detach().cpu().numpy()
    candidate_path_indices = output["candidate_path_indices"][0].detach().cpu().numpy()
    candidate_velocity_indices = (
        output["candidate_velocity_indices"][0].detach().cpu().numpy()
    )

    ref_path_index = int(reference["path_index"])
    ref_velocity_index = int(reference["velocity_index"])
    ref_model_candidate = np.where(
        (candidate_path_indices == ref_path_index)
        & (candidate_velocity_indices == ref_velocity_index)
    )[0]
    ref_model_trajectory_score = None
    if len(ref_model_candidate) > 0:
        ref_model_trajectory_score = float(trajectory_scores[ref_model_candidate[0]])

    pred_cost = float(scored_pred_path["cost"][pred_local_index])
    ref_cost = float(reference["cost"])
    return {
        "pred_flat_index": pred_flat_index,
        "pred_cost": pred_cost,
        "pred_goal_cost": float(scored_pred_path["goal_cost"][pred_local_index]),
        "pred_route_cost": float(scored_pred_path["route_cost"][pred_local_index]),
        "pred_clearance": float(scored_pred_path["clearance"][pred_local_index]),
        "pred_collides": bool(scored_pred_path["collision"][pred_local_index]),
        "cost_gap": pred_cost - ref_cost,
        "cost_ratio": pred_cost / max(abs(ref_cost), 1.0e-6),
        "pred_path_teacher_cost": float(path_costs[pred_path_index]),
        "ref_path_teacher_cost": float(path_costs[ref_path_index]),
        "pred_path_teacher_rank": compute_rank_ascending(
            path_costs,
            pred_path_index,
        ),
        "ref_path_teacher_rank": compute_rank_ascending(path_costs, ref_path_index),
        "pred_in_teacher_candidate_set": in_teacher_candidate_set,
        "pred_teacher_candidate_rank": teacher_candidate_rank,
        "pred_model_path_score": float(path_scores[pred_path_index]),
        "ref_model_path_score": float(path_scores[ref_path_index]),
        "pred_model_path_rank": compute_rank_descending(path_scores, pred_path_index),
        "ref_model_path_rank": compute_rank_descending(path_scores, ref_path_index),
        "pred_model_velocity_score": float(velocity_scores[pred_velocity_index]),
        "ref_model_velocity_score": float(velocity_scores[ref_velocity_index]),
        "pred_model_velocity_rank": compute_rank_descending(
            velocity_scores,
            pred_velocity_index,
        ),
        "ref_model_velocity_rank": compute_rank_descending(
            velocity_scores,
            ref_velocity_index,
        ),
        "pred_model_trajectory_score": float(trajectory_scores[pred_candidate_index]),
        "ref_model_trajectory_score": ref_model_trajectory_score,
    }


def make_dataset_reference(sample: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "trajectory": sample["target_trajectory"].numpy(),
        "trajectory_mask": sample["target_trajectory_mask"].numpy(),
        "path_index": int(sample["path_index"]),
        "velocity_index": int(sample["velocity_index"]),
        "flat_index": int(sample["trajectory_index"]),
        "collides": False,
        "cost": 0.0,
        "goal_cost": 0.0,
        "route_cost": 0.0,
        "clearance": 0.0,
    }


def plot_prediction(
    sample: dict[str, torch.Tensor],
    reference: dict[str, Any],
    pred_trajectory: np.ndarray,
    pred_path_index: int,
    pred_velocity_index: int,
    output_path: Path,
) -> None:
    config = GridConfig()

    target_path = sample.get("route_path", sample["target_path"]).numpy()
    old_target_trajectory = sample["target_trajectory"].numpy()
    old_target_mask = sample["target_trajectory_mask"].numpy()
    valid_old_target = old_target_trajectory[old_target_mask > 0.5]

    reference_trajectory = reference["trajectory"]
    reference_mask = reference["trajectory_mask"]
    valid_reference = reference_trajectory[reference_mask > 0.5]

    obstacle_centers = sample["obstacle_centers"].numpy()
    obstacle_sizes = sample["obstacle_sizes"].numpy()
    obstacle_velocities = sample["obstacle_velocities"].numpy()
    obstacle_mask = sample["obstacle_mask"].numpy()

    fig, ax = plt.subplots(figsize=(9, 8))

    for obstacle_index in np.where(obstacle_mask > 0.5)[0]:
        center_xy = obstacle_centers[obstacle_index]
        size_xy = obstacle_sizes[obstacle_index]
        velocity_xy = obstacle_velocities[obstacle_index]

        draw_rectangle_world(
            ax,
            center_xy,
            size_xy,
            color="#666666",
            alpha=0.45,
        )
        for time_s in TIME_STEPS:
            future_center = center_xy + velocity_xy * float(time_s)
            draw_rectangle_world(
                ax,
                future_center,
                size_xy,
                color="#ff9900",
                alpha=0.07,
            )

    ax.plot(
        target_path[:, 1],
        target_path[:, 0],
        color="#1f77b4",
        linewidth=1.2,
        alpha=0.75,
        label="route path",
    )
    ax.plot(
        valid_old_target[:, 1],
        valid_old_target[:, 0],
        color="#7f7f7f",
        linewidth=1.4,
        linestyle="--",
        alpha=0.75,
        label="old dataset target",
    )
    ax.plot(
        valid_reference[:, 1],
        valid_reference[:, 0],
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="teacher reference",
    )
    ax.plot(
        pred_trajectory[:, 1],
        pred_trajectory[:, 0],
        color="#d62728",
        linewidth=1.8,
        marker="x",
        markersize=5,
        label="pred trajectory",
    )

    setup_axis(
        ax,
        config,
        (
            f"ref p{reference['path_index']} v{reference['velocity_index']} | "
            f"pred p{pred_path_index} v{pred_velocity_index}"
        ),
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    vocab_cpu = SparseDriveVocab.load()
    vocab = vocab_cpu.to(device)
    model = ToySparseDriveV2Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ego_state = make_custom_ego_state()
    dataset = ToySparseDriveV2Dataset(num_samples=1024, seed_offset=10_000)
    sample_index = random.randrange(len(dataset))
    sample = dataset[sample_index]
    sample = apply_custom_goal(sample)
    if CUSTOM_OBSTACLES:
        active_obstacles = CUSTOM_OBSTACLES
        sample = override_obstacles(sample, active_obstacles)
    else:
        active_obstacles = sample_obstacles_to_dicts(sample)
    reference = select_teacher_reference(vocab_cpu, sample, active_obstacles, ego_state)

    input_grid = sample["input_grid"].unsqueeze(0).to(device)
    ego_state_tensor = make_ego_state_tensor().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(
            input_grid=input_grid,
            path_vocab=vocab.path,
            velocity_vocab=vocab.velocity,
            trajectory_vocab=vocab.trajectory,
            ego_state=ego_state_tensor,
        )
        prediction = select_prediction(output)

    pred_trajectory = prediction["trajectory"]
    reference_trajectory = (
        torch.from_numpy(reference["trajectory"]).unsqueeze(0).to(device)
    )
    reference_mask = (
        torch.from_numpy(reference["trajectory_mask"]).unsqueeze(0).to(device)
    )
    traj_l2, endpoint_l2 = compute_errors(
        pred_trajectory,
        reference_trajectory,
        reference_mask,
    )

    pred_path_index = int(prediction["path_index"][0].cpu())
    pred_velocity_index = int(prediction["velocity_index"][0].cpu())
    diagnostics = build_candidate_diagnostics(
        vocab=vocab_cpu,
        sample=sample,
        obstacles=active_obstacles,
        ego_state=ego_state,
        output=output,
        prediction=prediction,
        reference=reference,
    )

    output_path = OUTPUT_DIR / f"sample_{sample_index}.png"
    plot_prediction(
        sample=sample,
        reference=reference,
        pred_trajectory=pred_trajectory[0].cpu().numpy(),
        pred_path_index=pred_path_index,
        pred_velocity_index=pred_velocity_index,
        output_path=output_path,
    )

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"checkpoint metrics: {checkpoint['metrics']}")
    print(f"sample_index: {sample_index}")
    print(f"ego xy: {ego_state.xy}")
    print(f"ego yaw: {ego_state.yaw}")
    print(f"ego speed: {ego_state.speed}")
    print(f"ego size_xy: {ego_state.size_xy}")
    print(f"goal_xy: {sample['goal_xy'].tolist()}")
    print(f"old target path: {int(sample['path_index'])}")
    print(f"old target velocity: {int(sample['velocity_index'])}")
    print(f"reference path: {reference['path_index']}")
    print(f"reference velocity: {reference['velocity_index']}")
    print(f"reference collides: {reference['collides']}")
    print(f"reference cost: {reference['cost']:.4f}")
    print(f"reference_goal_cost: {reference['goal_cost']:.4f}")
    print(f"reference_route_cost: {reference['route_cost']:.4f}")
    print(f"reference_clearance: {reference['clearance']:.4f}")
    print(f"pred path: {pred_path_index}")
    print(f"pred velocity: {pred_velocity_index}")
    print(f"pred flat_index: {diagnostics['pred_flat_index']}")
    print(f"pred teacher_cost: {diagnostics['pred_cost']:.4f}")
    print(f"pred_goal_cost: {diagnostics['pred_goal_cost']:.4f}")
    print(f"pred_route_cost: {diagnostics['pred_route_cost']:.4f}")
    print(f"pred_clearance: {diagnostics['pred_clearance']:.4f}")
    print(f"pred_collides: {diagnostics['pred_collides']}")
    print(f"pred_vs_reference_cost_gap: {diagnostics['cost_gap']:.4f}")
    print(f"pred_vs_reference_cost_ratio: {diagnostics['cost_ratio']:.4f}")
    print(
        "teacher_path_cost/rank "
        f"ref={diagnostics['ref_path_teacher_cost']:.4f}/"
        f"{diagnostics['ref_path_teacher_rank']} "
        f"pred={diagnostics['pred_path_teacher_cost']:.4f}/"
        f"{diagnostics['pred_path_teacher_rank']}"
    )
    print(
        "pred_in_teacher_top_candidate_set: "
        f"{diagnostics['pred_in_teacher_candidate_set']}"
    )
    print(
        "pred_teacher_candidate_rank_in_top_set: "
        f"{diagnostics['pred_teacher_candidate_rank']}"
    )
    print(
        "model_path_score/rank "
        f"ref={diagnostics['ref_model_path_score']:.4f}/"
        f"{diagnostics['ref_model_path_rank']} "
        f"pred={diagnostics['pred_model_path_score']:.4f}/"
        f"{diagnostics['pred_model_path_rank']}"
    )
    print(
        "model_velocity_score/rank "
        f"ref={diagnostics['ref_model_velocity_score']:.4f}/"
        f"{diagnostics['ref_model_velocity_rank']} "
        f"pred={diagnostics['pred_model_velocity_score']:.4f}/"
        f"{diagnostics['pred_model_velocity_rank']}"
    )
    print(
        "model_trajectory_score "
        f"ref={optional_float(diagnostics['ref_model_trajectory_score'])} "
        f"pred={diagnostics['pred_model_trajectory_score']:.4f}"
    )
    print(f"traj_l2_to_reference: {traj_l2:.4f}")
    print(f"endpoint_l2_to_reference: {endpoint_l2:.4f}")
    print(f"saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
