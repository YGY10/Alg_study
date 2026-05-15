import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image


STUDY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_ROOT.parent / "SparseDriveV2"
DOWNLOAD_ROOT = PROJECT_ROOT / "download"

DEFAULT_MAP_ROOT = DOWNLOAD_ROOT / "maps"
DEFAULT_LOG_ROOT = DOWNLOAD_ROOT / "mini_navsim_logs" / "mini"
DEFAULT_SENSOR_ROOT = DOWNLOAD_ROOT / "mini_sensor_blobs" / "mini"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "ckpt" / "pretrained" / "sparsedrive_navsimv2_90p3.ckpt"
DEFAULT_OUT_DIR = STUDY_ROOT / "outputs" / "single_inference_navsim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SparseDriveV2 inference on a NAVSIM mini sample.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", type=Path, default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inputs-only", action="store_true", help="Save camera inputs and exit before model inference.")
    return parser.parse_args()


def choose_log_name(log_root: Path, sensor_root: Path) -> str:
    log_names = {path.stem for path in log_root.glob("*.pkl")}
    sensor_log_names = {
        path.name
        for path in sensor_root.iterdir()
        if path.is_dir() and (path / "CAM_F0").is_dir()
    }
    common = sorted(log_names & sensor_log_names)
    if not common:
        raise RuntimeError(f"No common camera logs found under {log_root} and {sensor_root}")
    return common[0]


def batchify_features(features: dict, device: torch.device) -> dict:
    camera_feature = features["camera_feature"]
    batched_camera = {}
    for key, value in camera_feature.items():
        if isinstance(value, torch.Tensor):
            batched_camera[key] = value.unsqueeze(0).to(device)
        else:
            batched_camera[key] = torch.as_tensor(value).unsqueeze(0).to(device)

    return {
        "camera_feature": batched_camera,
        "status_feature": features["status_feature"].unsqueeze(0).to(device),
    }


def plot_trajectory(trajectory: torch.Tensor, out_path: Path) -> None:
    traj = trajectory.detach().cpu().numpy()
    plt.figure(figsize=(6, 8))
    plt.plot(traj[:, 1], traj[:, 0], marker="o", linewidth=2)
    plt.scatter([0.0], [0.0], c="red", label="ego")
    plt.xlabel("lateral y [m]")
    plt.ylabel("forward x [m]")
    plt.title("SparseDrive predicted trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def attach_decoder_debug_hooks(model):
    debug = {}
    handles = []

    def save_tensor(name):
        def hook(_module, _inputs, output):
            debug[name] = output.detach().squeeze(-1).cpu()

        return hook

    for layer_idx, layer in enumerate(model._trajectory_head.decoder.layers):
        handles.append(layer.path_mlp.register_forward_hook(save_tensor(f"layer{layer_idx}_path_scores")))
        handles.append(layer.vel_mlp.register_forward_hook(save_tensor(f"layer{layer_idx}_vel_scores")))
        if hasattr(layer, "metric_heads"):
            for metric, head in layer.metric_heads.items():
                handles.append(head.register_forward_hook(save_tensor(f"metric_{metric}")))

    return debug, handles


def metric_score_from_debug(debug: dict, config) -> torch.Tensor:
    if config.dataset_version == "v1":
        return (
            debug["metric_no_at_fault_collisions"].sigmoid()
            * debug["metric_drivable_area_compliance"].sigmoid()
        ) * (
            5 * debug["metric_time_to_collision_within_bound"].sigmoid()
            + 5 * debug["metric_ego_progress"].sigmoid()
            + 2 * debug["metric_comfort"].sigmoid()
        )

    return (
        debug["metric_no_at_fault_collisions"].sigmoid()
        * debug["metric_drivable_area_compliance"].sigmoid()
        * debug["metric_driving_direction_compliance"].sigmoid()
        * debug["metric_traffic_light_compliance"].sigmoid()
    ) * (
        5 * debug["metric_time_to_collision_within_bound"].sigmoid()
        + 5 * debug["metric_ego_progress"].sigmoid()
        + 2 * debug["metric_lane_keeping"].sigmoid()
        + 2 * debug["metric_history_comfort"].sigmoid()
    )


def save_candidate_debug(model, debug: dict, config, out_dir: Path) -> dict:
    path_scores_0 = debug["layer0_path_scores"][0]
    vel_scores_0 = debug["layer0_vel_scores"][0]
    path_scores_1 = debug["layer1_path_scores"][0]
    vel_scores_1 = debug["layer1_vel_scores"][0]

    _, path_top0 = torch.topk(path_scores_0, config.path_filter_num[0])
    _, vel_top0 = torch.topk(vel_scores_0, config.velocity_filter_num[0])
    _, path_top1_rel = torch.topk(path_scores_1, config.path_filter_num[1])
    _, vel_top1_rel = torch.topk(vel_scores_1, config.velocity_filter_num[1])

    final_path_indices = path_top0[path_top1_rel]
    final_vel_indices = vel_top0[vel_top1_rel]

    metric_scores = metric_score_from_debug(debug, config)[0]
    selected_mode = int(metric_scores.argmax().item())
    selected_path_rank = selected_mode // len(final_vel_indices)
    selected_vel_rank = selected_mode % len(final_vel_indices)
    selected_path_index = int(final_path_indices[selected_path_rank].item())
    selected_vel_index = int(final_vel_indices[selected_vel_rank].item())

    path_vocab = model._trajectory_head.path_vocab.detach().cpu()
    vel_vocab = model._trajectory_head.vel_vocab.detach().cpu()
    selected_paths = path_vocab[final_path_indices]
    selected_vels = vel_vocab[final_vel_indices]

    torch.save(
        {
            "path_top0_indices": path_top0,
            "vel_top0_indices": vel_top0,
            "final_path_indices": final_path_indices,
            "final_vel_indices": final_vel_indices,
            "metric_scores": metric_scores,
            "selected_mode": selected_mode,
            "selected_path_rank": selected_path_rank,
            "selected_vel_rank": selected_vel_rank,
            "selected_path_index": selected_path_index,
            "selected_vel_index": selected_vel_index,
            "selected_paths": selected_paths,
            "selected_vels": selected_vels,
        },
        out_dir / "candidate_debug.pt",
    )

    with open(out_dir / "candidate_debug.txt", "w", encoding="utf-8") as fp:
        fp.write(f"layer0 path top-k count: {len(path_top0)}\n")
        fp.write(f"layer0 velocity top-k count: {len(vel_top0)}\n")
        fp.write(f"final path candidates: {final_path_indices.tolist()}\n")
        fp.write(f"final velocity candidates: {final_vel_indices.tolist()}\n")
        fp.write(f"selected mode: {selected_mode}\n")
        fp.write(f"selected path rank/index: {selected_path_rank} / {selected_path_index}\n")
        fp.write(f"selected velocity rank/index: {selected_vel_rank} / {selected_vel_index}\n")
        fp.write(f"selected metric score: {float(metric_scores[selected_mode]):.6f}\n")

    plt.figure(figsize=(7, 8))
    for rank, path in enumerate(selected_paths):
        color = "red" if rank == selected_path_rank else "steelblue"
        alpha = 1.0 if rank == selected_path_rank else 0.35
        linewidth = 2.5 if rank == selected_path_rank else 1.0
        plt.plot(path[:, 1], path[:, 0], color=color, alpha=alpha, linewidth=linewidth)
    plt.scatter([0.0], [0.0], c="black", s=20)
    plt.xlabel("lateral y [m]")
    plt.ylabel("forward x [m]")
    plt.title("Final 20 path candidates")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "candidate_paths.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    time = torch.arange(1, selected_vels.shape[1] + 1) * config.vel_time_interval
    for rank, velocity in enumerate(selected_vels):
        color = "red" if rank == selected_vel_rank else "steelblue"
        alpha = 1.0 if rank == selected_vel_rank else 0.35
        linewidth = 2.5 if rank == selected_vel_rank else 1.0
        plt.plot(time, velocity, marker="o", color=color, alpha=alpha, linewidth=linewidth)
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")
    plt.title("Final 10 velocity candidates")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "candidate_velocities.png")
    plt.close()

    return {
        "selected_path_index": selected_path_index,
        "selected_vel_index": selected_vel_index,
        "selected_path_rank": selected_path_rank,
        "selected_vel_rank": selected_vel_rank,
    }


def save_selected_path_projection(features: dict, model, selected_path_index: int, out_dir: Path) -> None:
    camera_feature = features["camera_feature"]
    projection_mat = camera_feature["projection_mat"].detach().cpu()
    path = model._trajectory_head.path_vocab[selected_path_index].detach().cpu()

    xy = path[:, :2]
    xyz1 = torch.cat(
        [
            xy,
            torch.zeros((xy.shape[0], 1), dtype=xy.dtype),
            torch.ones((xy.shape[0], 1), dtype=xy.dtype),
        ],
        dim=-1,
    )

    cam_names = ["left", "front", "right"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(len(path) - 1, 1)) for i in range(len(path))]

    for cam_idx, cam_name in enumerate(cam_names):
        img_path = out_dir / f"model_input_{cam_name}.jpg"
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        projected = (projection_mat[cam_idx] @ xyz1.T).T
        depth = projected[:, 2]
        uv = projected[:, :2] / depth.clamp(min=1e-5).unsqueeze(-1)
        valid = (
            (depth > 1e-5)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )

        axes[cam_idx].imshow(img)
        if valid.any():
            valid_uv = uv[valid]
            valid_indices = torch.arange(len(path))[valid]
            axes[cam_idx].scatter(
                valid_uv[:, 0],
                valid_uv[:, 1],
                c=[colors[int(i)] for i in valid_indices],
                s=20,
                edgecolors="black",
                linewidths=0.25,
            )
        axes[cam_idx].set_title(f"selected path on {cam_name}")
        axes[cam_idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "selected_path_projection.png")
    plt.close(fig)


def save_camera_inputs(agent_input, features: dict, out_dir: Path) -> None:
    cam_names = ["cam_l0", "cam_f0", "cam_r0"]
    display_names = ["left", "front", "right"]
    current_cameras = agent_input.cameras[-1]
    camera_feature = features["camera_feature"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for idx, (cam_name, display_name) in enumerate(zip(cam_names, display_names)):
        camera = getattr(current_cameras, cam_name)
        raw_img = Image.open(camera.image_path).convert("RGB")
        raw_img.save(out_dir / f"raw_{display_name}.jpg")

        axes[0, idx].imshow(raw_img)
        axes[0, idx].set_title(f"raw {display_name}")
        axes[0, idx].axis("off")

        img_tensor = camera_feature["imgs"][idx].detach().cpu()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img = (img_tensor * std + mean).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        Image.fromarray(img).save(out_dir / f"model_input_{display_name}.jpg")

        axes[1, idx].imshow(img)
        axes[1, idx].set_title(f"model input {display_name}")
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "camera_inputs.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    os.environ["NUPLAN_MAPS_ROOT"] = str(args.map_root.resolve())
    os.environ["OPENSCENE_DATA_ROOT"] = str(DOWNLOAD_ROOT.resolve())
    sys.path.insert(0, str(PROJECT_ROOT))

    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
    from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder
    from navsim.agents.sparsedrive.sparsedrive_model import SparseDriveModel
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    for anchor_name in ["path_1024.npy", "velocity_256.npy", "trajectory_1024_256.npz"]:
        anchor_path = PROJECT_ROOT / "ckpt" / "kmeans" / anchor_name
        if not anchor_path.is_file():
            raise FileNotFoundError(f"Missing anchor: {anchor_path}")

    log_name = args.log_name or choose_log_name(args.log_root, args.sensor_root)
    config = SparseDriveConfig(
        bkb_path=str((PROJECT_ROOT / "ckpt" / "resnet34.bin").resolve()),
        path_anchor=str((PROJECT_ROOT / "ckpt" / "kmeans" / "path_1024.npy").resolve()),
        velocity_anchor=str((PROJECT_ROOT / "ckpt" / "kmeans" / "velocity_256.npy").resolve()),
        trajectory_anchor=str((PROJECT_ROOT / "ckpt" / "kmeans" / "trajectory_1024_256.npz").resolve()),
    )

    sensor_config = SensorConfig(
        cam_f0=[0, 1, 2, 3],
        cam_l0=[0, 1, 2, 3],
        cam_l1=[0, 1, 2, 3],
        cam_l2=[0, 1, 2, 3],
        cam_r0=[0, 1, 2, 3],
        cam_r1=[0, 1, 2, 3],
        cam_r2=[0, 1, 2, 3],
        cam_b0=[0, 1, 2, 3],
        lidar_pc=[],
    )
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        has_route=True,
        max_scenes=1 if args.token is None else None,
        log_names=[log_name],
        tokens=[args.token] if args.token is not None else None,
    )
    scene_loader = SceneLoader(
        data_path=args.log_root,
        original_sensor_path=args.sensor_root,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    if len(scene_loader.tokens) == 0:
        raise RuntimeError(f"No scene loaded for log={log_name!r}, token={args.token!r}")

    token = scene_loader.tokens[0]
    agent_input = scene_loader.get_agent_input_from_token(token)

    feature_builder = SparseDriveFeatureBuilder(config)
    features = feature_builder.compute_features(agent_input)
    features, _, token = feature_builder.pipeline(features, {}, token, test_mode=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_camera_inputs(agent_input, features, args.out_dir)
    if args.inputs_only:
        print("log_name:", log_name)
        print("token:", token)
        print("saved camera inputs:", args.out_dir.resolve())
        return

    device = torch.device(args.device)
    model = SparseDriveModel(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model_state = {
        key.replace("agent._sparsedrive_model.", ""): value
        for key, value in state_dict.items()
        if key.startswith("agent._sparsedrive_model.")
    }
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    if unexpected_keys:
        print("unexpected checkpoint keys:", unexpected_keys)
    if missing_keys:
        print("missing checkpoint keys:", missing_keys)
    model.to(device)
    model.eval()

    batched_features = batchify_features(features, device)
    debug, hook_handles = attach_decoder_debug_hooks(model)
    with torch.no_grad():
        output, loss_dict = model(batched_features, {})
    for handle in hook_handles:
        handle.remove()

    trajectory = output["trajectory"][0]
    torch.save(
        {"token": token, "log_name": log_name, "trajectory": trajectory.detach().cpu()},
        args.out_dir / "prediction.pt",
    )
    plot_trajectory(trajectory, args.out_dir / "trajectory_bev.png")
    selected_candidate = save_candidate_debug(model, debug, config, args.out_dir)
    save_selected_path_projection(
        features,
        model,
        selected_candidate["selected_path_index"],
        args.out_dir,
    )

    print("log_name:", log_name)
    print("token:", token)
    print("trajectory shape:", tuple(trajectory.shape))
    print("first pose:", trajectory[0].detach().cpu().tolist())
    print("candidate debug:", (args.out_dir / "candidate_debug.txt").resolve())
    print("saved:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
