from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
TOY_ROOT = CURRENT_DIR.parent
for path in (TOY_ROOT, CURRENT_DIR, TOY_ROOT / "vocab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from dataset.dataset import ToySparseDriveV2Dataset, obstacles_to_teacher_dicts
except ModuleNotFoundError:
    from dataset import ToySparseDriveV2Dataset, obstacles_to_teacher_dicts
from teacher import TeacherConfig, score_all_trajectories_chunked, softmax_from_cost


DEFAULT_CACHE_DIR = TOY_ROOT / "cache" / "teacher_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-search Teacher labels for ToySparseDriveV2."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--path-chunk-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--num-path-candidates", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_cache_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    teacher_config = TeacherConfig(
        num_path_candidates=args.num_path_candidates,
        num_top_trajectories=args.top_k,
        temperature=2.0,
    )
    dataset = ToySparseDriveV2Dataset(
        num_samples=args.num_samples,
        seed_offset=args.seed_offset,
        teacher_config=teacher_config,
    )
    config_json = json.dumps(asdict(teacher_config), sort_keys=True)

    start_index = max(0, args.start_index)
    if start_index >= args.num_samples:
        raise ValueError("start-index must be smaller than num-samples")

    built = 0
    skipped = 0
    total_start = time.perf_counter()
    for index in range(start_index, args.num_samples):
        cache_path = args.output_dir / f"sample_{index:06d}.npz"
        if cache_path.is_file() and not args.overwrite:
            skipped += 1
            continue

        sample_start = time.perf_counter()
        route_path_index, route_path, goal_xy, obstacles = dataset.generate_scene(index)
        output = score_all_trajectories_chunked(
            vocab=dataset.vocab,
            goal_xy=goal_xy,
            obstacles=obstacles_to_teacher_dicts(obstacles),
            ego_state=dataset.ego_state,
            route_path=route_path[:, :2],
            config=teacher_config,
            path_chunk_size=args.path_chunk_size,
            top_k=args.top_k,
        )
        teacher_path_probs = softmax_from_cost(
            output.path_costs,
            teacher_config.temperature,
        ).astype(np.float32)

        save_cache_atomic(
            cache_path,
            cache_version=np.asarray(2, dtype=np.int64),
            sample_index=np.asarray(index, dtype=np.int64),
            seed_offset=np.asarray(args.seed_offset, dtype=np.int64),
            route_path_index=np.asarray(route_path_index, dtype=np.int64),
            teacher_config_json=np.asarray(config_json),
            best_path_index=np.asarray(output.best_path_index, dtype=np.int64),
            best_velocity_index=np.asarray(
                output.best_velocity_index,
                dtype=np.int64,
            ),
            best_flat_index=np.asarray(output.best_flat_index, dtype=np.int64),
            teacher_path_indices=output.path_indices.astype(np.int64),
            teacher_path_costs=output.path_costs.astype(np.float32),
            teacher_path_probs=teacher_path_probs,
            teacher_topk_flat_indices=output.candidate_flat_indices.astype(np.int64),
            teacher_topk_path_indices=output.candidate_path_indices.astype(np.int64),
            teacher_topk_velocity_indices=output.candidate_velocity_indices.astype(
                np.int64
            ),
            teacher_topk_costs=output.candidate_costs.astype(np.float32),
            teacher_topk_probs=output.candidate_probs.astype(np.float32),
            teacher_topk_collision=output.debug["collision"].astype(bool),
            teacher_topk_clearance=output.debug["clearance"].astype(np.float32),
        )
        built += 1
        elapsed = time.perf_counter() - sample_start
        print(
            f"[{index + 1}/{args.num_samples}] "
            f"sample={index} best=p{output.best_path_index} "
            f"v{output.best_velocity_index} cost={output.candidate_costs[0]:.3f} "
            f"time={elapsed:.2f}s"
        )

    total_elapsed = time.perf_counter() - total_start
    print(
        f"cache complete: built={built} skipped={skipped} "
        f"elapsed={total_elapsed:.1f}s output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
