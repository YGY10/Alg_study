import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reference_shift_env import ReferenceShiftEnv


# =========================================================
# 1. 最小 MLP policy
# =========================================================
class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 2. 动作采样
# =========================================================
def sample_action_around_center(center_action, action_limit, noise_std=0.3):
    center_action = np.asarray(center_action, dtype=np.float64).reshape(-1)
    noise = np.random.randn(*center_action.shape) * noise_std
    action = center_action + noise
    action = np.clip(action, -1.0, 1.0) * action_limit
    return action


def sample_random_action(action_dim, action_limit):
    return np.random.uniform(-1.0, 1.0, size=action_dim) * action_limit


def get_policy_mean_action(policy, state_tensor):
    with torch.no_grad():
        mean_action = policy(state_tensor).cpu().numpy().reshape(-1)
    return mean_action


# =========================================================
# 3. 固定场景评估 + 渲染
# =========================================================
def evaluate_and_render(
    policy,
    action_limit,
    iter_idx,
    num_ref_points=64,
    num_boundary_points=64,
    num_ctrl=12,
    safe_dist=1.3,
    hard_collision_dist=0.4,
    alpha_limit=2.0,
    seed=42,
):
    eval_env = ReferenceShiftEnv(
        num_ref_points=num_ref_points,
        num_boundary_points=num_boundary_points,
        num_ctrl=num_ctrl,
        safe_dist=safe_dist,
        hard_collision_dist=hard_collision_dist,
        alpha_limit=alpha_limit,
        seed=seed,
    )

    state = eval_env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action = policy(state_tensor).cpu().numpy().reshape(-1)
    action = np.clip(action, -1.0, 1.0) * action_limit

    _, reward, _, info = eval_env.step(action)

    print(
        f"[eval] iter={iter_idx:04d} "
        f"reward={reward:.3f} "
        f"d_min={info['d_min']:.3f}"
    )

    eval_env.render(info, title_prefix=f"iter={iter_idx}, reward={reward:.2f}")
    return reward, info["d_min"], action, info


# =========================================================
# 4. 约束优先选 best
#    先满足当前阶段的安全约束，再优化 reward
# =========================================================
def find_best_candidate(
    candidate_actions, candidate_rewards, candidate_infos, safe_dist
):
    safe_indices = [
        i for i, info in enumerate(candidate_infos) if info["d_min"] > safe_dist
    ]

    if len(safe_indices) > 0:
        # 有满足当前安全约束的候选：在这些里面选 reward 最好的
        best_idx = max(safe_indices, key=lambda i: candidate_rewards[i])
    else:
        # 没有满足当前安全约束的：选 d_min 最大的
        best_idx = max(
            range(len(candidate_infos)), key=lambda i: candidate_infos[i]["d_min"]
        )

    best_action = np.asarray(candidate_actions[best_idx], dtype=np.float64)
    best_reward = float(candidate_rewards[best_idx])
    best_info = candidate_infos[best_idx]
    return best_action, best_reward, best_info, best_idx


# =========================================================
# 5. replay batch
# =========================================================
def build_replay_batch(
    current_state, current_best_action, replay_buffer, replay_sample_k=8
):
    batch_states = [current_state]
    batch_actions = [current_best_action]

    if len(replay_buffer) > 0:
        k = min(replay_sample_k, len(replay_buffer))
        idxs = np.random.choice(len(replay_buffer), size=k, replace=False)
        for i in idxs:
            s_i, a_i, _, _ = replay_buffer[i]
            batch_states.append(s_i)
            batch_actions.append(a_i)

    batch_states = np.asarray(batch_states, dtype=np.float32)
    batch_actions = np.asarray(batch_actions, dtype=np.float32)
    return batch_states, batch_actions


# =========================================================
# 6. 训练主循环
# =========================================================
def main():
    # -----------------------------
    # 场景参数
    # -----------------------------
    env_kwargs = dict(
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=64,
        safe_dist=1.3,  # 环境内部仍保留最终目标安全距离
        hard_collision_dist=0.4,
        alpha_limit=2.0,
    )

    # 固定训练 / 评估场景
    train_seed = 42
    eval_seed = 42

    # 初始化一次，拿维度
    init_env = ReferenceShiftEnv(**env_kwargs, seed=train_seed)
    init_state = init_env.reset()
    state_dim = init_state.shape[0]
    action_dim = env_kwargs["num_ctrl"]
    action_limit = env_kwargs["alpha_limit"]

    print("state_dim =", state_dim)
    print("action_dim =", action_dim)

    device = torch.device("cpu")
    policy = MLPPolicy(state_dim, action_dim, hidden_dim=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)

    # -----------------------------
    # 训练参数
    # -----------------------------
    num_iters = 100
    render_every = 1
    sleep_sec = 0.2

    # -----------------------------
    # 采样参数
    # -----------------------------
    num_candidates = 96

    # 正常情况下三采样配比
    best_center_ratio = 0.50
    policy_center_ratio = 0.20
    random_ratio = 0.30

    # 噪声退火
    noise_std_start = 0.45
    noise_std_end = 0.05

    # reset 时强制全局探索噪声
    reset_noise_std = 0.25

    # 卡住判定
    patience = 10
    no_improve_counter = 0

    # replay
    replay_buffer = []
    max_replay_size = 50
    replay_sample_k = 8

    # 历史最优动作
    best_action_so_far = None
    best_reward_so_far = -1e18
    best_info_so_far = None

    # -----------------------------
    # 渐进式安全约束
    # 从当前系统能做到的量级开始，逐步逼近最终 1.3
    # -----------------------------
    safe_dist_start = 0.20
    safe_dist_end = env_kwargs["safe_dist"]

    print("\n>>> Initial policy evaluation")
    evaluate_and_render(
        policy=policy,
        action_limit=action_limit,
        iter_idx=0,
        seed=eval_seed,
        **env_kwargs,
    )

    for it in range(1, num_iters + 1):
        # -------------------------
        # 1) 固定训练场景
        # -------------------------
        train_env = ReferenceShiftEnv(**env_kwargs, seed=train_seed)
        state = train_env.reset()

        state_np = state.copy()
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        policy_mean_action = get_policy_mean_action(policy, state_tensor)

        if best_action_so_far is None:
            best_action_so_far = policy_mean_action.copy()

        # -------------------------
        # 2) 基础噪声退火
        # -------------------------
        alpha = (it - 1) / max(1, num_iters - 1)
        base_noise_std = noise_std_start + alpha * (noise_std_end - noise_std_start)

        # -------------------------
        # 3) 当前阶段的安全约束（schedule）
        # -------------------------
        current_safe_dist = safe_dist_start + alpha * (safe_dist_end - safe_dist_start)

        # -------------------------
        # 4) 长时间没提升，触发全局探索 reset
        # -------------------------
        if no_improve_counter >= patience:
            use_global_explore = True
            no_improve_counter = 0
            noise_std = max(base_noise_std, reset_noise_std)
            print(">>> RESET TO GLOBAL EXPLORATION")
        else:
            use_global_explore = False
            noise_std = base_noise_std

        # -------------------------
        # 5) 候选采样
        # -------------------------
        candidate_actions = []
        candidate_rewards = []
        candidate_infos = []

        if use_global_explore:
            # reset 时：取消 best 精修，强化 policy + random
            num_best_center = 0
            num_policy_center = int(num_candidates * 0.40)
            num_random = num_candidates - num_policy_center
        else:
            num_best_center = int(num_candidates * best_center_ratio)
            num_policy_center = int(num_candidates * policy_center_ratio)
            num_random = num_candidates - num_best_center - num_policy_center

        # 5.1 围绕历史 best 采样
        for _ in range(num_best_center):
            sampled_action = sample_action_around_center(
                center_action=best_action_so_far,
                action_limit=action_limit,
                noise_std=noise_std,
            )
            _, reward, _, info = train_env.step(sampled_action)
            candidate_actions.append(sampled_action)
            candidate_rewards.append(reward)
            candidate_infos.append(info)

        # 5.2 围绕当前 policy 采样
        for _ in range(num_policy_center):
            sampled_action = sample_action_around_center(
                center_action=policy_mean_action,
                action_limit=action_limit,
                noise_std=noise_std,
            )
            _, reward, _, info = train_env.step(sampled_action)
            candidate_actions.append(sampled_action)
            candidate_rewards.append(reward)
            candidate_infos.append(info)

        # 5.3 完全随机采样
        for _ in range(num_random):
            sampled_action = sample_random_action(
                action_dim=action_dim,
                action_limit=action_limit,
            )
            _, reward, _, info = train_env.step(sampled_action)
            candidate_actions.append(sampled_action)
            candidate_rewards.append(reward)
            candidate_infos.append(info)

        # -------------------------
        # 6) 约束优先选 best
        # -------------------------
        (
            best_action_this_iter,
            best_reward_this_iter,
            best_info_this_iter,
            _,
        ) = find_best_candidate(
            candidate_actions,
            candidate_rewards,
            candidate_infos,
            safe_dist=current_safe_dist,
        )

        # -------------------------
        # 7) 更新历史 best
        # 这里仍然按 reward 更新历史 best，
        # 因为当前 best 已经过“约束优先筛选”
        # -------------------------
        if best_reward_this_iter > best_reward_so_far:
            best_reward_so_far = best_reward_this_iter
            best_action_so_far = best_action_this_iter.copy()
            best_info_so_far = best_info_this_iter

            replay_buffer.append(
                (
                    state_np.copy(),
                    best_action_this_iter.copy(),
                    best_reward_this_iter,
                    best_info_this_iter["d_min"],
                )
            )
            replay_buffer = replay_buffer[-max_replay_size:]

            no_improve_counter = 0
        else:
            no_improve_counter += 1

        # -------------------------
        # 8) 当前 best + replay 一起训练
        # -------------------------
        batch_states, batch_actions = build_replay_batch(
            current_state=state_np,
            current_best_action=best_action_this_iter,
            replay_buffer=replay_buffer,
            replay_sample_k=replay_sample_k,
        )

        batch_states_tensor = torch.tensor(
            batch_states, dtype=torch.float32, device=device
        )
        batch_actions_tensor = torch.tensor(
            batch_actions, dtype=torch.float32, device=device
        )

        optimizer.zero_grad()
        pred_actions = policy(batch_states_tensor)
        target_actions_norm = batch_actions_tensor / action_limit
        loss = torch.mean((pred_actions - target_actions_norm) ** 2)
        loss.backward()
        optimizer.step()

        # -------------------------
        # 9) 打印
        # -------------------------
        print(
            f"[train] iter={it:04d} "
            f"current_safe_dist={current_safe_dist:.3f} "
            f"noise_std={noise_std:.3f} "
            f"loss={loss.item():.4f} "
            f"best_reward_this_iter={best_reward_this_iter:.3f} "
            f"best_d_min_this_iter={best_info_this_iter['d_min']:.3f} "
            f"best_reward_so_far={best_reward_so_far:.3f} "
            f"best_d_min_so_far={best_info_so_far['d_min'] if best_info_so_far is not None else -1:.3f} "
            f"replay_size={len(replay_buffer)} "
            f"no_improve_counter={no_improve_counter}"
        )

        # -------------------------
        # 10) 固定场景可视化
        # -------------------------
        if it % render_every == 0:
            evaluate_and_render(
                policy=policy,
                action_limit=action_limit,
                iter_idx=it,
                seed=eval_seed,
                **env_kwargs,
            )
            time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
