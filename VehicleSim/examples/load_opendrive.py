from __future__ import annotations

import argparse
from pathlib import Path

from sim2d.map import (
    RoadNetwork,
    load_opendrive_road_network,
)


def print_road_network(
    road_network: RoadNetwork,
) -> None:
    print("RoadNetwork")
    print(f"  source_type : {road_network.source_type}")
    print(f"  source_name : {road_network.source_name}")
    print(f"  lane_count  : {road_network.lane_count}")
    print(f"  metadata    : {dict(road_network.metadata)}")

    for lane in road_network.lanes:
        centerline = lane.centerline

        print()
        print(f"Lane {lane.lane_id}")
        print(f"  type        : {lane.lane_type.value}")
        print(f"  points      : {centerline.point_count}")
        print(f"  length      : {centerline.length:.3f} m")
        print(
            "  start       : "
            f"({centerline.start[0]:.3f}, "
            f"{centerline.start[1]:.3f})"
        )
        print(
            "  end         : " f"({centerline.end[0]:.3f}, " f"{centerline.end[1]:.3f})"
        )
        print(
            "  left start  : "
            f"({lane.left_boundary.start[0]:.3f}, "
            f"{lane.left_boundary.start[1]:.3f})"
        )
        print(
            "  right start : "
            f"({lane.right_boundary.start[0]:.3f}, "
            f"{lane.right_boundary.start[1]:.3f})"
        )
        print(f"  metadata    : {dict(lane.metadata)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an OpenDRIVE file and print the " "converted VehicleSim RoadNetwork."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to the .xodr file",
    )

    parser.add_argument(
        "--sample-step",
        type=float,
        default=0.5,
        help="Reference-line sampling interval in metres",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    road_network = load_opendrive_road_network(
        args.path,
        sample_step=args.sample_step,
    )

    print_road_network(road_network)


if __name__ == "__main__":
    main()
