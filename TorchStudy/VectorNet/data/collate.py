from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from data.line import SceneSample
from data.vectorize import VECTOR_FEATURE_DIM, scene_to_polylines


@dataclass
class VectorNetBatch:
    """
    VectorNet batch tensor.

    x:
      [B, P, V, F]

    vector_mask:
      [B, P, V]
      1 表示该 vector segment 有效，0 表示 padding。

    polyline_mask:
      [B, P]
      1 表示该 polyline 有效，0 表示 padding。

    target_index:
      [B]
      每个 scene 中目标 agent history polyline 的 index。

    future:
      [B, T, 2]
    """

    x: torch.Tensor
    vector_mask: torch.Tensor
    polyline_mask: torch.Tensor
    target_index: torch.Tensor
    future: torch.Tensor


def vectornet_collate_fn(samples: List[SceneSample]) -> VectorNetBatch:
    """
    把 list[SceneSample] 合成一个 batch。

    输入:
      samples:
        长度为 B 的 SceneSample 列表

    输出:
      VectorNetBatch
    """

    batch_polylines = []
    target_indices = []
    futures = []

    max_num_polylines = 0
    max_num_vectors = 0

    for sample in samples:
        polylines, target_index = scene_to_polylines(sample)

        batch_polylines.append(polylines)
        target_indices.append(target_index)
        futures.append(sample.future)

        max_num_polylines = max(max_num_polylines, len(polylines))

        for polyline in polylines:
            max_num_vectors = max(max_num_vectors, polyline.shape[0])

    batch_size = len(samples)
    feature_dim = VECTOR_FEATURE_DIM

    x = torch.zeros(
        batch_size,
        max_num_polylines,
        max_num_vectors,
        feature_dim,
        dtype=torch.float32,
    )

    vector_mask = torch.zeros(
        batch_size,
        max_num_polylines,
        max_num_vectors,
        dtype=torch.float32,
    )

    polyline_mask = torch.zeros(
        batch_size,
        max_num_polylines,
        dtype=torch.float32,
    )

    for batch_idx, polylines in enumerate(batch_polylines):
        for polyline_idx, polyline in enumerate(polylines):
            num_vectors = polyline.shape[0]

            x[batch_idx, polyline_idx, :num_vectors] = torch.from_numpy(polyline)
            vector_mask[batch_idx, polyline_idx, :num_vectors] = 1.0
            polyline_mask[batch_idx, polyline_idx] = 1.0

    target_index = torch.tensor(target_indices, dtype=torch.long)
    future = torch.tensor(futures, dtype=torch.float32)

    return VectorNetBatch(
        x=x,
        vector_mask=vector_mask,
        polyline_mask=polyline_mask,
        target_index=target_index,
        future=future,
    )
