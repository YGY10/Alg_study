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
    with torch.no_grad():
        output, loss_dict = model(batched_features, {})

    trajectory = output["trajectory"][0]
    torch.save(
        {"token": token, "log_name": log_name, "trajectory": trajectory.detach().cpu()},
        args.out_dir / "prediction.pt",
    )
    plot_trajectory(trajectory, args.out_dir / "trajectory_bev.png")

    print("log_name:", log_name)
    print("token:", token)
    print("trajectory shape:", tuple(trajectory.shape))
    print("first pose:", trajectory[0].detach().cpu().tolist())
    print("saved:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
