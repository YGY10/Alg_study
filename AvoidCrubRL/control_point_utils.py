import numpy as np


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_parametric_offset_profile(
    s_ref,
    l_peak,
    s_rise_start,
    s_rise_end,
    s_fall_end,
):
    """
    生成一个：
    0 -> 平滑抬升 -> 平台 -> 平滑回归 -> 0
    的横向偏移曲线
    """
    s_ref = np.asarray(s_ref, dtype=float)
    l = np.zeros_like(s_ref)

    # 防止非法顺序
    s0 = s_ref[0]
    s1 = s_ref[-1]

    s_rise_start = np.clip(s_rise_start, s0, s1)
    s_rise_end = np.clip(s_rise_end, s_rise_start + 1e-3, s1)
    s_fall_end = np.clip(s_fall_end, s_rise_end + 1e-3, s1)

    # 1) rise
    mask_rise = (s_ref >= s_rise_start) & (s_ref <= s_rise_end)
    if np.any(mask_rise):
        x = (s_ref[mask_rise] - s_rise_start) / max(s_rise_end - s_rise_start, 1e-6)
        l[mask_rise] = l_peak * smoothstep(x)

    # 2) hold
    mask_hold = (s_ref > s_rise_end) & (s_ref <= s_fall_end)
    l[mask_hold] = l_peak

    # 3) fall
    # 这里把回归安排到最后 20% 路程里
    s_return_start = s_fall_end
    s_return_end = s_return_start + 0.18 * (s1 - s0)
    s_return_end = min(s_return_end, s1)

    mask_fall = (s_ref > s_return_start) & (s_ref <= s_return_end)
    if np.any(mask_fall):
        x = (s_ref[mask_fall] - s_return_start) / max(
            s_return_end - s_return_start, 1e-6
        )
        l[mask_fall] = l_peak * (1.0 - smoothstep(x))

    return l


def build_l_from_basis_coeffs(s_ref, coeffs):
    """
    coeffs = [a0, a1, a2, a3, a4]
    解释为：
      a0 -> l_peak
      a1 -> s_rise_start
      a2 -> s_rise_end
      a3 -> s_fall_end
      a4 -> hold_ratio / shape辅助
    """
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    assert len(coeffs) == 5, "parameterization 版本要求 action_dim=5"

    s_ref = np.asarray(s_ref, dtype=float)
    s0 = s_ref[0]
    s1 = s_ref[-1]
    s_len = max(s1 - s0, 1e-6)

    a0, a1, a2, a3, a4 = coeffs

    # 1) 最大偏移：允许左右，但你当前场景主要需要向左避让
    l_peak = np.clip(a0, -4.0, 4.0)

    # 2) 把几个参数映射到 s 范围
    #   这里不用直接把网络输出当绝对s，而是当比例更稳
    p1 = np.clip((a1 + 2.0) / 4.0, 0.08, 0.58)
    p2 = np.clip((a2 + 2.0) / 4.0, 0.18, 0.76)
    p3 = np.clip((a3 + 2.0) / 4.0, 0.28, 0.90)
    hold_ratio = np.clip((a4 + 2.0) / 4.0, 0.05, 0.32)

    # 排序，保证时序合理
    p_start = min(p1, p2, p3)
    p_mid = np.clip(sorted([p1, p2, p3])[1], p_start + 0.05, 0.90)
    p_end = np.clip(max(p1, p2, p3), p_mid + 0.05, 0.98)

    s_rise_start = s0 + p_start * s_len
    s_rise_end = s0 + p_mid * s_len

    # 平台长度
    hold_len = hold_ratio * s_len
    s_fall_end = min(s_rise_end + hold_len, s0 + p_end * s_len)

    l_total = build_parametric_offset_profile(
        s_ref=s_ref,
        l_peak=l_peak,
        s_rise_start=s_rise_start,
        s_rise_end=s_rise_end,
        s_fall_end=s_fall_end,
    )

    debug = {
        "l_raw": l_total.copy(),
        "l_total": l_total,
        "l_peak": l_peak,
        "s_rise_start": s_rise_start,
        "s_rise_end": s_rise_end,
        "s_fall_end": s_fall_end,
    }
    return l_total, debug
