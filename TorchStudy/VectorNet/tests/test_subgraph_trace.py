from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.common import masked_max_pool
from models.subgraph import SubGraphEncoder
from models.global_graph import GlobalGraphEncoder


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"\n{name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(tensor.detach())


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(
        precision=4,
        sci_mode=False,
    )

    # 一个 batch，包含两条 polyline。
    #
    # Polyline 0:
    #   vector 0 = [1, 1]
    #   vector 1 = [2, 1]
    #   vector 2 = [3, 2]
    #
    # Polyline 1:
    #   vector 0 = [4, 1]
    #   vector 1 = [5, 2]
    #   vector 2 = padding
    x = torch.tensor(
        [
            [
                [[1.0, 1.0], [2.0, 1.0], [3.0, 2.0]],
                [[4.0, 1.0], [5.0, 2.0], [0.0, 0.0]],
            ]
        ]
    )

    vector_mask = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ]
    )

    model = SubGraphEncoder(
        input_dim=2,
        hidden_dim=2,
        num_layers=2,
    )

    print_tensor("0. Original input x", x)
    print_tensor("0. Vector mask", vector_mask)

    h = x

    for layer_index, layer in enumerate(model.layers, start=1):
        print(f"\n{'=' * 60}")
        print(f"SubGraphLayer {layer_index}")
        print("=" * 60)

        # 对应 SubGraphLayer.forward() 的第一步。
        local_feature = layer.mlp(h)
        print_tensor(
            f"{layer_index}.1 Local MLP output",
            local_feature,
        )

        # 对同一条 polyline 内部的 vector 做 pooling。
        polyline_feature = masked_max_pool(
            local_feature,
            vector_mask,
            dim=2,
        )
        print_tensor(
            f"{layer_index}.2 Polyline max-pool output",
            polyline_feature,
        )

        # 将整条 polyline 的特征复制回每个 vector。
        num_vectors = local_feature.shape[2]

        expanded_feature = polyline_feature.unsqueeze(2).expand(
            -1,
            -1,
            num_vectors,
            -1,
        )
        print_tensor(
            f"{layer_index}.3 Expanded polyline feature",
            expanded_feature,
        )

        # 拼接 vector 局部特征和 polyline 汇总特征。
        h = torch.cat(
            [local_feature, expanded_feature],
            dim=-1,
        )

        h = h * vector_mask.unsqueeze(-1)

        print_tensor(
            f"{layer_index}.4 Concatenated layer output",
            h,
        )

    print(f"\n{'=' * 60}")
    print("Final polyline encoding")
    print("=" * 60)

    final_pooled = masked_max_pool(
        h,
        vector_mask,
        dim=2,
    )
    print_tensor(
        "3.1 Final max-pool output",
        final_pooled,
    )

    final_feature = model.output_projection(final_pooled)
    print_tensor(
        "3.2 Final projected polyline feature",
        final_feature,
    )

    # 确认手动逐层计算与模型 forward 完全相同。
    model_output = model(x, vector_mask)
    print_tensor(
        "4. SubGraphEncoder forward output",
        model_output,
    )

    print(
        "\nmanual trace equals model output:",
        torch.allclose(final_feature, model_output),
    )

    print(f"\n{'=' * 60}")
    print("Global Graph")
    print("=" * 60)

    global_graph = GlobalGraphEncoder(hidden_dim=2)

    # 当前两条 polyline 都有效。
    polyline_mask = torch.tensor(
        [
            [1.0, 1.0],
        ]
    )

    print_tensor(
        "5.1 Global Graph input",
        final_feature,
    )

    layer = global_graph.global_graph

    query = layer.query_projection(final_feature)
    key = layer.key_projection(final_feature)
    value = layer.value_projection(final_feature)

    print_tensor("5.2 Query", query)
    print_tensor("5.3 Key", key)
    print_tensor("5.4 Value", value)

    attention_scores = torch.matmul(
        query,
        key.transpose(-1, -2),
    )

    print_tensor(
        "5.5 Raw attention scores: Q @ K^T",
        attention_scores,
    )

    attention_scores = attention_scores / (layer.hidden_dim**0.5)

    print_tensor(
        "5.6 Scaled attention scores",
        attention_scores,
    )

    key_mask = polyline_mask.unsqueeze(1)

    masked_attention_scores = attention_scores.masked_fill(
        key_mask == 0,
        float("-inf"),
    )

    print_tensor(
        "5.7 Masked attention scores",
        masked_attention_scores,
    )

    attention_weights = torch.softmax(
        masked_attention_scores,
        dim=-1,
    )

    print_tensor(
        "5.8 Attention weights",
        attention_weights,
    )

    global_feature = torch.matmul(
        attention_weights,
        value,
    )

    global_feature = global_feature * polyline_mask.unsqueeze(-1)

    print_tensor(
        "5.9 Global Graph output",
        global_feature,
    )

    model_global_output = global_graph(
        final_feature,
        polyline_mask,
    )

    print_tensor(
        "5.10 GlobalGraphEncoder forward output",
        model_global_output,
    )

    print(
        "\nmanual global trace equals model output:",
        torch.allclose(global_feature, model_global_output),
    )


if __name__ == "__main__":
    main()
