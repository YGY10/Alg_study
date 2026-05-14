import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reference_shift_env import ReferenceShiftEnv


# =========================================================
# 1. 一些基础工具
# =========================================================
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.8
EPS = 1e-6


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def gaussian_logprob(noise: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    residual = -0.5 * (noise.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
    return residual.sum(dim=-1, keepdim=True)


def squash_action_and_logprob(mu: torch.Tensor, log_std: torch.Tensor):
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

    noise = torch.randn_like(mu)
    u = mu + std * noise
    a = torch.tanh(u)

    log_prob_u = gaussian_logprob((u - mu) / std, log_std)
    correction = torch.log(1.0 - a.pow(2) + EPS).sum(dim=-1, keepdim=True)
    log_prob_a = log_prob_u - correction

    return a, log_prob_a, u


def eval_logprob_from_action(
    action: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor
):
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

    a = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
    u = atanh(a)

    noise = (u - mu) / std
    log_prob_u = gaussian_logprob(noise, log_std)
    correction = torch.log(1.0 - a.pow(2) + EPS).sum(dim=-1, keepdim=True)
    log_prob_a = log_prob_u - correction
    return log_prob_a


# =========================================================
# 2. Policy / Value
# =========================================================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)

        # 关键：全局可学习 log_std，不再 state-dependent
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.2)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 3. 评估 + 可视化
# =========================================================
def evaluate_and_render(
    policy_net,
    action_limit,
    iter_idx,
    env_kwargs,
    seed=42,
):
    eval_env = ReferenceShiftEnv(**env_kwargs, seed=seed)
    state = eval_env.reset()

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mu, _ = policy_net(state_t)
        action = torch.tanh(mu).cpu().numpy().reshape(-1)

    action = np.clip(action, -1.0, 1.0) * action_limit
    _, reward, _, info = eval_env.step(action)

    print(
        f"[eval] iter={iter_idx:04d} "
        f"reward={reward:.3f} "
        f"d_min={info['d_min']:.3f}"
    )

    eval_env.render(info, title_prefix=f"iter={iter_idx}, reward={reward:.2f}")
    return reward, info["d_min"], info


# =========================================================
# 4. 主训练
# =========================================================
def main():
    # -----------------------------------------------------
    # 环境参数
    # -----------------------------------------------------
    env_kwargs = dict(
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=5,
        safe_dist=1.3,
        hard_collision_dist=0.4,
        alpha_limit=2.0,
    )

    train_seed = 42
    eval_seed = 42

    tmp_env = ReferenceShiftEnv(**env_kwargs, seed=train_seed)
    init_state = tmp_env.reset()

    state_dim = init_state.shape[0]
    action_dim = env_kwargs["num_ctrl"]
    action_limit = env_kwargs["alpha_limit"]

    print("state_dim =", state_dim)
    print("action_dim =", action_dim)

    device = torch.device("cpu")

    policy_net = PolicyNet(state_dim, action_dim, hidden_dim=256).to(device)
    value_net = ValueNet(state_dim, hidden_dim=256).to(device)

    policy_optim = optim.Adam(policy_net.parameters(), lr=3e-4)
    value_optim = optim.Adam(value_net.parameters(), lr=5e-4)

    # -----------------------------------------------------
    # PPO 参数
    # -----------------------------------------------------
    num_iters = 300
    batch_size = 256
    ppo_epochs = 8
    mini_batch_size = 64
    clip_eps = 0.2
    entropy_coef = 0.0
    value_coef = 0.5
    max_grad_norm = 1.0

    # reward 处理
    reward_shift = 5500.0
    reward_scale = 1000.0

    render_every = 5
    sleep_sec = 0.2

    print("\n>>> Initial policy evaluation")
    evaluate_and_render(
        policy_net=policy_net,
        action_limit=action_limit,
        iter_idx=0,
        env_kwargs=env_kwargs,
        seed=eval_seed,
    )

    for it in range(1, num_iters + 1):
        # -------------------------------------------------
        # 1) 收集 batch（单步 episode）
        # -------------------------------------------------
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        raw_rewards = []
        values = []
        d_mins = []

        for _ in range(batch_size):
            env = ReferenceShiftEnv(**env_kwargs, seed=train_seed)
            state = env.reset()

            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
                0
            )

            with torch.no_grad():
                mu, log_std = policy_net(state_t)
                action_t, log_prob_t, _ = squash_action_and_logprob(mu, log_std)
                value_t = value_net(state_t)

            action_np = action_t.cpu().numpy().reshape(-1)
            action_np = np.clip(action_np, -1.0, 1.0) * action_limit

            _, reward_raw, _, info = env.step(action_np)

            # 关键：平移 + 缩放
            reward = (reward_raw + reward_shift) / reward_scale

            states.append(state)
            actions.append(action_t.cpu().numpy().reshape(-1))
            old_log_probs.append(log_prob_t.cpu().numpy().item())
            rewards.append(reward)
            raw_rewards.append(reward_raw)
            values.append(value_t.cpu().numpy().item())
            d_mins.append(info["d_min"])

        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.asarray(actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(
            np.asarray(old_log_probs), dtype=torch.float32, device=device
        ).unsqueeze(-1)
        rewards = torch.tensor(
            np.asarray(rewards), dtype=torch.float32, device=device
        ).unsqueeze(-1)
        values = torch.tensor(
            np.asarray(values), dtype=torch.float32, device=device
        ).unsqueeze(-1)

        # 单步 episode: return = reward
        returns = rewards

        # advantage
        advantages = returns - values
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + 1e-6
        advantages = (advantages - adv_mean) / adv_std

        # -------------------------------------------------
        # 2) PPO update
        # -------------------------------------------------
        batch_n = states.shape[0]
        idx_all = np.arange(batch_n)

        policy_loss_mean = 0.0
        value_loss_mean = 0.0
        entropy_mean = 0.0
        update_steps = 0

        for _ in range(ppo_epochs):
            np.random.shuffle(idx_all)

            for start in range(0, batch_n, mini_batch_size):
                mb_idx = idx_all[start : start + mini_batch_size]

                s_mb = states[mb_idx]
                a_mb = actions[mb_idx]
                old_lp_mb = old_log_probs[mb_idx]
                ret_mb = returns[mb_idx]
                adv_mb = advantages[mb_idx]

                mu, log_std = policy_net(s_mb)
                new_log_prob = eval_logprob_from_action(a_mb, mu, log_std)

                ratio = torch.exp(new_log_prob - old_lp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy = (
                    (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1).mean()
                )

                value_pred = value_net(s_mb)
                value_loss = ((value_pred - ret_mb) ** 2).mean()

                total_loss = (
                    policy_loss + value_coef * value_loss - entropy_coef * entropy
                )

                policy_optim.zero_grad()
                value_optim.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)

                policy_optim.step()
                value_optim.step()

                policy_loss_mean += policy_loss.item()
                value_loss_mean += value_loss.item()
                entropy_mean += entropy.item()
                update_steps += 1

        policy_loss_mean /= max(update_steps, 1)
        value_loss_mean /= max(update_steps, 1)
        entropy_mean /= max(update_steps, 1)

        # -------------------------------------------------
        # 3) 打印
        # -------------------------------------------------
        reward_mean = float(np.mean(raw_rewards))
        reward_max = float(np.max(raw_rewards))
        dmin_mean = float(np.mean(d_mins))
        dmin_max = float(np.max(d_mins))

        print(
            f"[train] iter={it:04d} "
            f"reward_mean={reward_mean:.3f} "
            f"reward_max={reward_max:.3f} "
            f"dmin_mean={dmin_mean:.3f} "
            f"dmin_max={dmin_max:.3f} "
            f"policy_loss={policy_loss_mean:.4f} "
            f"value_loss={value_loss_mean:.6f} "
            f"entropy={entropy_mean:.4f}"
        )

        # -------------------------------------------------
        # 4) 固定场景评估
        # -------------------------------------------------
        if it % render_every == 0:
            evaluate_and_render(
                policy_net=policy_net,
                action_limit=action_limit,
                iter_idx=it,
                env_kwargs=env_kwargs,
                seed=eval_seed,
            )
            time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
