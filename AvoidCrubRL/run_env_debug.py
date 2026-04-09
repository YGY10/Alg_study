import numpy as np

from reference_shift_env import ReferenceShiftEnv


def main():
    env = ReferenceShiftEnv(
        num_ref_points=64,
        num_boundary_points=64,
        num_ctrl=12,  # 你现在建议先用12
        safe_dist=1.3,
        hard_collision_dist=0.4,
        alpha_limit=2.0,
        seed=42,
    )

    state = env.reset()
    print("state shape =", state.shape)

    num_samples = 100

    best_reward = -1e18
    worst_reward = 1e18
    best_info = None
    worst_info = None
    best_action = None
    worst_action = None

    for _ in range(num_samples):
        action = np.random.uniform(-2.0, 2.0, size=env.num_ctrl)
        _, reward, _, info = env.step(action)

        if reward > best_reward:
            best_reward = reward
            best_info = info
            best_action = action

        if reward < worst_reward:
            worst_reward = reward
            worst_info = info
            worst_action = action

    print("\n===== BEST SAMPLE =====")
    print("reward =", best_reward)
    print("d_min =", best_info["d_min"])
    print("ctrl_values =", best_action)
    print("reward_terms =", best_info["reward_terms"])

    print("\n===== WORST SAMPLE =====")
    print("reward =", worst_reward)
    print("d_min =", worst_info["d_min"])
    print("ctrl_values =", worst_action)
    print("reward_terms =", worst_info["reward_terms"])

    print("\n>>> Visualizing BEST")
    env.render(best_info, title_prefix="BEST")

    print("\n>>> Visualizing WORST")
    env.render(worst_info, title_prefix="WORST")


if __name__ == "__main__":
    main()
