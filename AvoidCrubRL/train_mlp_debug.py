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
            nn.Tanh(),  # 输出范围先压到 [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 2. 采样动作
# =========================================================
def sample_action_from_policy(policy, state_tensor, action_limit, noise_std=0.3):
    """
    policy 输出 mean action
    然后加高斯噪声做探索
    """
    with torch.no_grad():
        mean_action = policy(state_tensor).cpu().numpy().reshape(-1)

    noise = np.random.randn(*mean_action.shape) * noise_std
    action = mean_action + noise
    action = np.clip(action, -1.0, 1.0) * action_limit
    return mean_action, action


# =========================================================
# 3. 用 reward 计算 loss
# =========================================================
def policy_loss_from_reward(policy_out, target_action, reward, action_limit):
    """
    最小可用 debug 版：
    reward 越高（越不负），越值得让 policy 靠近 target_action
    """
    target_action_norm = torch.tensor(
        target_action / action_limit,
        dtype=torch.float32,
        device=policy_out.device,
    )

    reward_tensor = torch.tensor(
        reward,
        dtype=torch.float32,
        device=policy_out.device,
    )

    mse = torch.mean((policy_out.squeeze(0) - target_action_norm) ** 2)

    # 当前 reward 大致在 [-6000, -50] 左右
    weight = 1.0 + (reward_tensor + 6000.0) / 1000.0
    weight = torch.clamp(weight, min=0.05, max=5.0)

    return weight * mse


# =========================================================
# 4. 固定场景评估 + 渲染
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
    """
    每次评估都重新创建同一个 seed 的环境，
    保证看到的是固定场景。
    """
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
# 5. 从候选动作里挑 top-k elite，并做加权平均
# =========================================================
def build_elite_target_action(candidate_actions, candidate_rewards, top_k):
    """
    candidate_actions: list of np.ndarray, each shape=(action_dim,)
    candidate_rewards: list of float

    返回：
        elite_target_action: top-k 按 reward 加权平均后的动作
        elite_reward_mean: top-k 的平均 reward
        elite_best_reward: top-k 中最高 reward
        elite_best_idx: 原候选里的最佳索引
    """
    rewards = np.asarray(candidate_rewards, dtype=np.float64)
    actions = np.asarray(candidate_actions, dtype=np.float64)

    # reward 越高越好，所以降序
    elite_idx = np.argsort(rewards)[-top_k:]
    elite_rewards = rewards[elite_idx]
    elite_actions = actions[elite_idx]

    # 归一化成正权重
    # 用 elite 内部相对差异，避免数值太夸张
    elite_rewards_shift = elite_rewards - np.max(elite_rewards)
    weights = np.exp(elite_rewards_shift / 50.0)  # 温度 50，可调
    weights = weights / (np.sum(weights) + 1e-12)

    elite_target_action = np.sum(elite_actions * weights[:, None], axis=0)
    elite_reward_mean = float(np.mean(elite_rewards))
    elite_best_reward = float(np.max(elite_rewards))
    elite_best_idx = int(elite_idx[np.argmax(elite_rewards)])

    return elite_target_action, elite_reward_mean, elite_best_reward, elite_best_idx


# =========================================================
# 6. 训练主循环
# =========================================================
def main():
    # -----------------------------
    # 训练环境：每次 reset 随机场景
    # -----------------------------
    train_env = ReferenceShiftEnv(
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=12,
        safe_dist=1.3,
        hard_collision_dist=0.4,
        alpha_limit=2.0,
        seed=0,
    )

    # 先拿一次 state 确定维度
    state = train_env.reset()
    state_dim = state.shape[0]
    action_dim = train_env.num_ctrl
    action_limit = train_env.alpha_limit

    print("state_dim =", state_dim)
    print("action_dim =", action_dim)

    # policy
    device = torch.device("cpu")
    policy = MLPPolicy(state_dim, action_dim, hidden_dim=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # 训练参数
    num_iters = 100
    render_every = 1
    sleep_sec = 0.2

    # 探索参数
    noise_std_start = 0.6
    noise_std_min = 0.15
    noise_decay = 0.985
    num_candidates = 32
    top_k = 4

    print("\n>>> Initial policy evaluation")
    evaluate_and_render(
        policy=policy,
        action_limit=action_limit,
        iter_idx=0,
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=12,
        safe_dist=1.3,
        hard_collision_dist=0.4,
        alpha_limit=2.0,
        seed=42,
    )

    for it in range(1, num_iters + 1):
        noise_std_now = max(noise_std_min, noise_std_start * (noise_decay**it))
        # -------------------------
        # 1) 随机场景
        # -------------------------
        state = train_env.reset()
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # 当前 policy 输出
        policy_out = policy(state_tensor)

        # -------------------------
        # 2) 在当前 policy 附近采样很多动作
        # -------------------------
        candidate_actions = []
        candidate_rewards = []
        candidate_infos = []

        for _ in range(num_candidates):
            _, sampled_action = sample_action_from_policy(
                policy,
                state_tensor,
                action_limit=action_limit,
                noise_std=noise_std_now,
            )

            _, reward, _, info = train_env.step(sampled_action)

            candidate_actions.append(sampled_action)
            candidate_rewards.append(reward)
            candidate_infos.append(info)

        # -------------------------
        # 3) 从 top-k elite 里构造一个更稳的目标动作
        # -------------------------
        (
            elite_target_action,
            elite_reward_mean,
            elite_best_reward,
            elite_best_idx,
        ) = build_elite_target_action(
            candidate_actions=candidate_actions,
            candidate_rewards=candidate_rewards,
            top_k=top_k,
        )
        elite_best_info = candidate_infos[elite_best_idx]

        # -------------------------
        # 4) 用 elite 平均动作更新 policy
        # -------------------------
        optimizer.zero_grad()
        loss = policy_loss_from_reward(
            policy_out=policy_out,
            target_action=elite_target_action,
            reward=elite_reward_mean,
            action_limit=action_limit,
        )
        loss.backward()
        optimizer.step()

        # -------------------------
        # 5) 打印
        # -------------------------
        print(
            f"[train] iter={it:04d} "
            f"noise_std={noise_std_now:.3f} "
            f"loss={loss.item():.4f} "
            f"elite_mean_reward={elite_reward_mean:.3f} "
            f"elite_best_reward={elite_best_reward:.3f} "
            f"elite_best_d_min={elite_best_info['d_min']:.3f}"
        )

        # -------------------------
        # 6) 固定场景可视化
        # -------------------------
        if it % render_every == 0:
            evaluate_and_render(
                policy=policy,
                action_limit=action_limit,
                iter_idx=it,
                num_ref_points=64,
                num_boundary_points=64,
                num_ctrl=12,
                safe_dist=1.3,
                hard_collision_dist=0.4,
                alpha_limit=2.0,
                seed=42,
            )
            time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
