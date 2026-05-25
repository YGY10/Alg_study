from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import torch


STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"
DOWNLOAD_ROOT = PROJECT_ROOT / "download"

DEFAULT_INFERENCE_DIR = STUDY_ROOT / "outputs" / "single_inference_navsim"
DEFAULT_OUT_DIR = STUDY_ROOT / "outputs" / "candidate_coverage"
DEFAULT_LOG_ROOT = DOWNLOAD_ROOT / "mini_navsim_logs" / "mini"
DEFAULT_SENSOR_ROOT = DOWNLOAD_ROOT / "mini_sensor_blobs" / "mini"
DEFAULT_MAP_ROOT = DOWNLOAD_ROOT / "maps"
DEFAULT_ANCHOR = PROJECT_ROOT / "ckpt" / "kmeans" / "trajectory_1024_256.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether SparseDrive trajectory candidates cover the expert trajectory."
    )
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", type=Path, default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--trajectory-anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument(
        "--num-poses",
        type=int,
        default=8,
        help="Number of future poses used by SparseDriveV2.",
    )
    return parser.parse_args()


def load_scene_gt(
    log_root: Path,
    sensor_root: Path,
    map_root: Path,
    log_name: str,
    token: str,
    num_poses: int,
) -> np.ndarray:
    os.environ["NUPLAN_MAPS_ROOT"] = str(map_root.resolve())
    os.environ["OPENSCENE_DATA_ROOT"] = str(DOWNLOAD_ROOT.resolve())
    sys.path.insert(0, str(PROJECT_ROOT))

    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=max(10, num_poses),
        has_route=True,
        log_names=[log_name],
        tokens=[token],
    )
    scene_loader = SceneLoader(
        data_path=log_root,
        original_sensor_path=sensor_root,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    scene = scene_loader.get_scene_from_token(token)
    return scene.get_future_trajectory(num_poses).poses.astype(np.float32)


def compute_errors(candidates: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy_error = np.linalg.norm(candidates[:, :, :2] - gt[None, :, :2], axis=-1)
    ade = xy_error.mean(axis=1)
    fde = xy_error[:, -1]
    return ade, fde


def unravel_path_velocity(flat_index: int, num_velocity: int) -> tuple[int, int]:
    return flat_index // num_velocity, flat_index % num_velocity


def plot_candidate_coverage(
    out_dir: Path,
    final_candidates: np.ndarray,
    gt: np.ndarray,
    selected_mode: int,
    best_final_idx: int,
    prediction: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))

    for candidate in final_candidates:
        ax.plot(candidate[:, 1], candidate[:, 0], color="gray", alpha=0.18, linewidth=0.8)

    ax.plot(
        final_candidates[best_final_idx, :, 1],
        final_candidates[best_final_idx, :, 0],
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        label="best final candidate by ADE",
    )
    ax.plot(
        prediction[:, 1],
        prediction[:, 0],
        color="red",
        marker="o",
        linewidth=2.5,
        label="model selected",
    )
    ax.plot(
        gt[:, 1],
        gt[:, 0],
        color="green",
        marker="o",
        linewidth=2.5,
        label="expert / GT",
    )
    ax.scatter([0.0], [0.0], c="black", s=28, label="ego")

    ax.set_title(f"Final 200 candidates, selected mode={selected_mode}")
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")
    ax.axis("equal")
    ax.grid(True, linewidth=0.6, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_coverage_bev.png", dpi=180)
    plt.close(fig)


def plot_error_hist(out_dir: Path, final_ade: np.ndarray, full_ade: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(final_ade, bins=30, color="#1f77b4", alpha=0.8)
    axes[0].axvline(final_ade.min(), color="red", linewidth=2, label="best")
    axes[0].set_title("Final 200 ADE")
    axes[0].set_xlabel("ADE [m]")
    axes[0].set_ylabel("count")
    axes[0].grid(True, linewidth=0.6, alpha=0.45)
    axes[0].legend()

    axes[1].hist(full_ade, bins=60, color="#9467bd", alpha=0.8)
    axes[1].axvline(full_ade.min(), color="red", linewidth=2, label="best")
    axes[1].set_title("Full 1024x256 library ADE")
    axes[1].set_xlabel("ADE [m]")
    axes[1].set_ylabel("count")
    axes[1].grid(True, linewidth=0.6, alpha=0.45)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "candidate_error_hist.png", dpi=180)
    plt.close(fig)


def plot_top_final_candidates(
    out_dir: Path,
    final_candidates: np.ndarray,
    gt: np.ndarray,
    final_ade: np.ndarray,
    top_k: int = 20,
) -> None:
    top_indices = np.argsort(final_ade)[:top_k]

    fig, ax = plt.subplots(figsize=(8, 9))
    for rank, idx in enumerate(top_indices):
        alpha = 0.25 + 0.65 * (1.0 - rank / max(top_k - 1, 1))
        ax.plot(
            final_candidates[idx, :, 1],
            final_candidates[idx, :, 0],
            color="#1f77b4",
            alpha=alpha,
            linewidth=1.2,
        )
    ax.plot(gt[:, 1], gt[:, 0], color="green", marker="o", linewidth=2.5, label="GT")
    ax.scatter([0.0], [0.0], c="black", s=28)
    ax.set_title(f"Top {top_k} final candidates closest to GT")
    ax.set_xlabel("lateral y [m]")
    ax.set_ylabel("forward x [m]")
    ax.axis("equal")
    ax.grid(True, linewidth=0.6, alpha=0.55)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top_final_candidates_by_ade.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prediction_data = torch.load(args.inference_dir / "prediction.pt", map_location="cpu")
    candidate_debug = torch.load(args.inference_dir / "candidate_debug.pt", map_location="cpu")
    trajectory_data = np.load(args.trajectory_anchor)

    log_name = prediction_data["log_name"]
    token = prediction_data["token"]
    prediction = prediction_data["trajectory"].numpy().astype(np.float32)
    gt = load_scene_gt(
        args.log_root,
        args.sensor_root,
        args.map_root,
        log_name,
        token,
        args.num_poses,
    )

    trajectory_vocab = trajectory_data["trajectory"].astype(np.float32)
    final_path_indices = candidate_debug["final_path_indices"].numpy().astype(np.int64)
    final_vel_indices = candidate_debug["final_vel_indices"].numpy().astype(np.int64)
    selected_mode = int(candidate_debug["selected_mode"])

    final_candidates = trajectory_vocab[
        final_path_indices[:, None],
        final_vel_indices[None, :],
    ].reshape(-1, args.num_poses, 3)
    full_candidates = trajectory_vocab.reshape(-1, args.num_poses, 3)

    final_ade, final_fde = compute_errors(final_candidates, gt)
    full_ade, full_fde = compute_errors(full_candidates, gt)
    pred_ade, pred_fde = compute_errors(prediction[None], gt)

    best_final_idx = int(final_ade.argmin())
    best_full_idx = int(full_ade.argmin())
    best_final_path_rank, best_final_vel_rank = unravel_path_velocity(
        best_final_idx,
        len(final_vel_indices),
    )
    best_full_path_idx, best_full_vel_idx = unravel_path_velocity(
        best_full_idx,
        trajectory_vocab.shape[1],
    )

    selected_path_rank, selected_vel_rank = unravel_path_velocity(
        selected_mode,
        len(final_vel_indices),
    )

    summary = {
        "log_name": log_name,
        "token": token,
        "num_final_candidates": int(len(final_candidates)),
        "num_full_candidates": int(len(full_candidates)),
        "model_selected": {
            "mode": selected_mode,
            "path_rank": selected_path_rank,
            "velocity_rank": selected_vel_rank,
            "path_index": int(final_path_indices[selected_path_rank]),
            "velocity_index": int(final_vel_indices[selected_vel_rank]),
            "ade": float(pred_ade[0]),
            "fde": float(pred_fde[0]),
        },
        "best_final_candidate": {
            "mode": best_final_idx,
            "path_rank": best_final_path_rank,
            "velocity_rank": best_final_vel_rank,
            "path_index": int(final_path_indices[best_final_path_rank]),
            "velocity_index": int(final_vel_indices[best_final_vel_rank]),
            "ade": float(final_ade[best_final_idx]),
            "fde": float(final_fde[best_final_idx]),
        },
        "best_full_library_candidate": {
            "mode": best_full_idx,
            "path_index": best_full_path_idx,
            "velocity_index": best_full_vel_idx,
            "ade": float(full_ade[best_full_idx]),
            "fde": float(full_fde[best_full_idx]),
        },
        "final_candidate_ade_percentiles": {
            "p50": float(np.percentile(final_ade, 50)),
            "p75": float(np.percentile(final_ade, 75)),
            "p90": float(np.percentile(final_ade, 90)),
        },
        "full_library_ade_percentiles": {
            "p50": float(np.percentile(full_ade, 50)),
            "p75": float(np.percentile(full_ade, 75)),
            "p90": float(np.percentile(full_ade, 90)),
        },
    }

    (args.out_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "SparseDrive candidate coverage",
        "",
        f"log_name: {log_name}",
        f"token: {token}",
        "",
        "model selected:",
        f"  mode={selected_mode}",
        f"  path index={summary['model_selected']['path_index']}",
        f"  velocity index={summary['model_selected']['velocity_index']}",
        f"  ADE={pred_ade[0]:.4f} m, FDE={pred_fde[0]:.4f} m",
        "",
        "best among final 200 candidates:",
        f"  mode={best_final_idx}",
        f"  path index={summary['best_final_candidate']['path_index']}",
        f"  velocity index={summary['best_final_candidate']['velocity_index']}",
        f"  ADE={final_ade[best_final_idx]:.4f} m, FDE={final_fde[best_final_idx]:.4f} m",
        "",
        "best among full 1024x256 library:",
        f"  mode={best_full_idx}",
        f"  path index={best_full_path_idx}",
        f"  velocity index={best_full_vel_idx}",
        f"  ADE={full_ade[best_full_idx]:.4f} m, FDE={full_fde[best_full_idx]:.4f} m",
    ]
    (args.out_dir / "coverage_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_candidate_coverage(
        args.out_dir,
        final_candidates,
        gt,
        selected_mode,
        best_final_idx,
        prediction,
    )
    plot_error_hist(args.out_dir, final_ade, full_ade)
    plot_top_final_candidates(args.out_dir, final_candidates, gt, final_ade)

    print("\n".join(lines))
    print(f"\nsaved: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
