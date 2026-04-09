import numpy as np


def smooth_signal(y, num_iter=3):
    """
    一个轻量平滑器，避免控制点插值后出现折感
    """
    y = y.copy()
    for _ in range(num_iter):
        y_new = y.copy()
        y_new[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
        y = y_new
    return y


def gaussian_basis(s_ref, centers, sigma):
    basis_list = []
    for c in centers:
        phi = np.exp(-0.5 * ((s_ref - c) / sigma) ** 2)
        basis_list.append(phi)
    return np.stack(basis_list, axis=0)  # [num_basis, N]


def normalize_basis_rows(basis):
    row_sum = np.sum(basis, axis=1, keepdims=True) + 1e-8
    return basis / row_sum


def build_l_from_basis_coeffs(s_ref, coeffs):
    """
    coeffs 前4维: global coeffs
    coeffs 后8维: local coeffs

    输出:
        l: 每个 s_ref 点对应的横向偏移
        debug: 便于可视化/调试
    """
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    assert len(coeffs) == 12, "当前版本要求 action_dim=12 (4 global + 8 local)"

    global_coeffs = coeffs[:4]
    local_coeffs = coeffs[4:]

    s0 = s_ref[0]
    s1 = s_ref[-1]
    s_len = max(s1 - s0, 1e-6)

    # -------------------------
    # 1) Global basis: 4个宽基函数
    # -------------------------
    global_centers = np.linspace(s0, s1, 4)
    global_sigma = 0.28 * s_len
    global_basis = gaussian_basis(s_ref, global_centers, global_sigma)
    global_basis = normalize_basis_rows(global_basis)

    # -------------------------
    # 2) Local basis: 8个窄基函数
    # -------------------------
    local_centers = np.linspace(s0, s1, 8)
    local_sigma = 0.10 * s_len
    local_basis = gaussian_basis(s_ref, local_centers, local_sigma)
    local_basis = normalize_basis_rows(local_basis)

    # -------------------------
    # 3) local 分量缩小一点，避免乱补
    # -------------------------
    local_scale = 0.45

    l_global = np.sum(global_coeffs[:, None] * global_basis, axis=0)
    l_local = np.sum(local_coeffs[:, None] * local_basis, axis=0) * local_scale

    l = l_global + l_local

    # 再做一点整体平滑
    l = smooth_signal(l, num_iter=4)

    debug = {
        "global_coeffs": global_coeffs,
        "local_coeffs": local_coeffs,
        "global_centers": global_centers,
        "local_centers": local_centers,
        "l_global": l_global,
        "l_local": l_local,
        "l_total": l,
    }
    return l, debug


def build_l_from_control_points(s_ref, ctrl_values):
    """
    用控制点生成整条 l(s)
    改成固定“前密后疏”控制点分布
    """
    num_ctrl = len(ctrl_values)

    if num_ctrl == 8:
        ctrl_s_ratio = np.array([0.0, 0.06, 0.12, 0.22, 0.36, 0.55, 0.78, 1.0])

    elif num_ctrl == 12:
        ctrl_s_ratio = np.array(
            [0.00, 0.04, 0.08, 0.13, 0.20, 0.28, 0.38, 0.50, 0.64, 0.78, 0.90, 1.00]
        )

    elif num_ctrl == 18:
        ctrl_s_ratio = np.array(
            [
                0.00,
                0.03,
                0.06,
                0.09,
                0.12,
                0.16,
                0.20,
                0.25,
                0.30,
                0.36,
                0.43,
                0.50,
                0.58,
                0.67,
                0.76,
                0.85,
                0.93,
                1.00,
            ]
        )

    else:
        # 其它情况先退回均匀分布
        ctrl_s_ratio = np.linspace(0.0, 1.0, num_ctrl)

    ctrl_s = s_ref[0] + ctrl_s_ratio * (s_ref[-1] - s_ref[0])

    # 先把控制点本身压顺
    ctrl_values_smooth = smooth_signal(ctrl_values, num_iter=2)

    # 再插值成整条曲线
    l = np.interp(s_ref, ctrl_s, ctrl_values_smooth)

    # 再做整体平滑
    l = smooth_signal(l, num_iter=8)
    return l, ctrl_s
