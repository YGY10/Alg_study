# 轨迹预测所用的log脚本
import csv
import os


def init_motion_log(csv_path: str):
    # 这里其实还应该加一个，文件不存在或者文件存在但是第一行不是header
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(
                [
                    "episode_id",
                    "step_id",
                    "t",
                    "agent_type",
                    "agent_id",  # 0自车1障碍车
                    "x",
                    "y",
                    "v",
                    "yaw",
                ]
            )


def log_motion_frame(
    csv_path: str, episode_id: int, step_id: int, t: float, ego, obstacles
):
    # 传参类型：ego (VehicleModel) obstacle List[VehicleModel]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # 记录自车数据
        writer.writerow(
            [
                episode_id,
                step_id,
                f"{t:3f}",
                0,  # agent type: 0 自车
                0,
                f"{ego.x:.3f}",
                f"{ego.y:.3f}",
                f"{ego.v:.3f}",
                f"{ego.yaw:.3f}",
            ]
        )

        # 再记录每个obstacle
        for i, o in enumerate(obstacles):
            writer.writerow(
                [
                    episode_id,
                    step_id,
                    f"{t:.3f}",
                    1,  # agent type: 1障碍物
                    f"{o.x:.3f}",
                    f"{o.y:.3f}",
                    f"{o.v:.3f}",
                    f"{o.yaw:.3f}",
                ]
            )
