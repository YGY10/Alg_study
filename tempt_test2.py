import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class LaneLineGap:
    lane_id: int
    s_start: float
    s_end: float
    gap_len: float
    l_mean: float


@dataclass
class LaneGapJunctionCandidate:
    valid: bool
    s_start: float
    s_end: float
    l_min: float
    l_max: float
    support_line_count: int
    confidence: float
    gaps: List[LaneLineGap]
    source: str = "unknown"

    end_support: int = 0
    start_support: int = 0
    blank_len: float = 0.0


@dataclass
class LaneSegmentFeature:
    lane_id: int
    s_min: float
    s_max: float
    l_start: float
    l_end: float
    l_mean: float
    s_span: float
    l_span: float


def detect_gaps_for_one_lane(
    lane_id: int,
    pts_sl: np.ndarray,
    min_gap_len: float = 8.0,
) -> List[LaneLineGap]:
    """
    旧逻辑：
    检测同一条车道线内部的长 gap。

    注意：
    这个函数只能检测：
      前半段车道线
      -> 中间空白
      -> 后半段车道线重新出现

    它不能检测“多条车道线集体终止”的场景。
    """
    if pts_sl is None or len(pts_sl) < 2:
        return []

    pts = np.asarray(pts_sl, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]

    gaps = []

    for i in range(len(pts) - 1):
        s0, l0 = pts[i]
        s1, l1 = pts[i + 1]

        gap_len = s1 - s0

        if gap_len >= min_gap_len:
            gaps.append(
                LaneLineGap(
                    lane_id=lane_id,
                    s_start=float(s0),
                    s_end=float(s1),
                    gap_len=float(gap_len),
                    l_mean=float(0.5 * (l0 + l1)),
                )
            )

    return gaps


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def cluster_lane_gaps(
    gaps: List[LaneLineGap],
    cluster_s_tol: float = 6.0,
    min_support_lines: int = 2,
    min_cluster_gap_len: float = 8.0,
    min_lateral_width: float = 2.5,
    s_margin_before: float = 1.0,
    s_margin_after: float = 1.0,
    l_margin: float = 1.0,
) -> List[LaneGapJunctionCandidate]:
    """
    旧逻辑：
    把多条车道线内部 gap 聚成 junction candidate。
    """
    if not gaps:
        return []

    gaps_sorted = sorted(gaps, key=lambda g: 0.5 * (g.s_start + g.s_end))
    clusters: List[List[LaneLineGap]] = []

    for gap in gaps_sorted:
        gap_mid = 0.5 * (gap.s_start + gap.s_end)
        inserted = False

        for cluster in clusters:
            c_s_start = min(g.s_start for g in cluster)
            c_s_end = max(g.s_end for g in cluster)
            c_mid = 0.5 * (c_s_start + c_s_end)

            has_overlap = (
                interval_overlap(gap.s_start, gap.s_end, c_s_start, c_s_end) > 0.0
            )
            mid_close = abs(gap_mid - c_mid) <= cluster_s_tol

            if has_overlap or mid_close:
                cluster.append(gap)
                inserted = True
                break

        if not inserted:
            clusters.append([gap])

    candidates = []

    for cluster in clusters:
        lane_ids = {g.lane_id for g in cluster}
        support_line_count = len(lane_ids)

        raw_s_start = min(g.s_start for g in cluster)
        raw_s_end = max(g.s_end for g in cluster)
        cluster_gap_len = raw_s_end - raw_s_start

        l_values = [g.l_mean for g in cluster]
        raw_l_min = min(l_values)
        raw_l_max = max(l_values)
        lateral_width = raw_l_max - raw_l_min

        confidence = 0.0

        if support_line_count >= 2:
            confidence += 0.35
        if support_line_count >= 3:
            confidence += 0.25
        if cluster_gap_len >= 12.0:
            confidence += 0.20
        if lateral_width >= 2.5:
            confidence += 0.15
        if lateral_width >= 5.0:
            confidence += 0.05

        confidence = min(confidence, 1.0)

        valid = (
            support_line_count >= min_support_lines
            and cluster_gap_len >= min_cluster_gap_len
            and lateral_width >= min_lateral_width
        )

        candidates.append(
            LaneGapJunctionCandidate(
                valid=valid,
                s_start=float(raw_s_start - s_margin_before),
                s_end=float(raw_s_end + s_margin_after),
                l_min=float(raw_l_min - l_margin),
                l_max=float(raw_l_max + l_margin),
                support_line_count=support_line_count,
                confidence=confidence,
                gaps=cluster,
                source="gap_cluster",
                end_support=support_line_count,
                start_support=support_line_count,
                blank_len=float(cluster_gap_len),
            )
        )

    return candidates


def extract_lane_segment_features(
    lane_lines: Dict[int, np.ndarray],
    min_lane_visible_len: float = 4.0,
    min_longitudinal_ratio: float = 1.2,
) -> List[LaneSegmentFeature]:
    """
    把每条输入车道线抽象成一个 segment feature。

    这里的输入 lane_lines 不一定是真实“同一条 lane boundary”。
    它可以是感知输出的若干条 polyline segment。

    我们只拿纵向结构作为路口证据：
      s_span 要足够长；
      s_span 要明显大于 l_span。
    """
    segments: List[LaneSegmentFeature] = []

    for lane_id, pts_sl in lane_lines.items():
        if pts_sl is None or len(pts_sl) < 2:
            continue

        pts = np.asarray(pts_sl, dtype=float)
        pts = pts[np.argsort(pts[:, 0])]

        s_min = float(np.min(pts[:, 0]))
        s_max = float(np.max(pts[:, 0]))

        s_span = s_max - s_min
        if s_span < min_lane_visible_len:
            continue

        l_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))

        # 过滤掉明显横向线段。
        # 路口横线、停止线等不应该作为“纵向车道线终止/开始”的证据。
        if s_span < min_longitudinal_ratio * max(l_span, 1e-3):
            continue

        # 因为 pts 已经按 s 排序，所以第一个点就是 start，最后一个点就是 end。
        l_start = float(pts[0, 1])
        l_end = float(pts[-1, 1])
        l_mean = float(np.mean(pts[:, 1]))

        segments.append(
            LaneSegmentFeature(
                lane_id=lane_id,
                s_min=s_min,
                s_max=s_max,
                l_start=l_start,
                l_end=l_end,
                l_mean=l_mean,
                s_span=s_span,
                l_span=l_span,
            )
        )

    return segments


def cluster_by_s(
    items: List[dict],
    key: str,
    tol: float,
) -> List[List[dict]]:
    """
    按某个 s 值聚类。
    """
    if not items:
        return []

    items_sorted = sorted(items, key=lambda x: x[key])
    clusters: List[List[dict]] = []

    for item in items_sorted:
        inserted = False

        for cluster in clusters:
            mid_s = float(np.mean([x[key] for x in cluster]))
            if abs(item[key] - mid_s) <= tol:
                cluster.append(item)
                inserted = True
                break

        if not inserted:
            clusters.append([item])

    return clusters


def cluster_lateral_width(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def build_collective_candidate_from_end_and_start(
    end_cluster: List[dict],
    start_cluster: Optional[List[dict]],
    observe_s_max: float,
    min_support_lines: int,
    min_start_support_lines: int,
    min_lateral_width: float,
    min_blank_len: float,
    s_margin_before: float,
    s_margin_after: float,
    l_margin: float,
) -> Optional[LaneGapJunctionCandidate]:
    """
    从一个“集体终止 cluster”和一个“后方重新开始 cluster”构造路口候选。

    如果 start_cluster 是 None，则退化成 collective_lane_end：
      只有前一组线集体结束，后方没有重新出现的线。
    """
    end_support = len({x["lane_id"] for x in end_cluster})
    end_s_values = np.array([x["end_s"] for x in end_cluster], dtype=float)
    end_l_values = np.array([x["end_l"] for x in end_cluster], dtype=float)

    end_s_mean = float(np.mean(end_s_values))
    end_s_std = float(np.std(end_s_values))

    end_l_min = float(np.min(end_l_values))
    end_l_max = float(np.max(end_l_values))
    end_lateral_width = end_l_max - end_l_min

    if start_cluster is not None:
        start_support = len({x["lane_id"] for x in start_cluster})
        start_s_values = np.array([x["start_s"] for x in start_cluster], dtype=float)
        start_l_values = np.array([x["start_l"] for x in start_cluster], dtype=float)

        start_s_mean = float(np.mean(start_s_values))
        start_s_std = float(np.std(start_s_values))

        raw_blank_len = start_s_mean - end_s_mean

        union_l_values = np.concatenate([end_l_values, start_l_values])
        raw_l_min = float(np.min(union_l_values))
        raw_l_max = float(np.max(union_l_values))

        source = "collective_end_start"
    else:
        start_support = 0
        start_s_mean = observe_s_max
        start_s_std = 999.0

        raw_blank_len = observe_s_max - end_s_mean

        raw_l_min = end_l_min
        raw_l_max = end_l_max

        source = "collective_lane_end"

    lateral_width = raw_l_max - raw_l_min

    if source == "collective_end_start":
        valid = (
            end_support >= min_support_lines
            and start_support >= min_start_support_lines
            and raw_blank_len >= min_blank_len
            and lateral_width >= min_lateral_width
        )
    else:
        # 只有集体终止，没有后方重新开始时，要求更严格一点。
        valid = (
            end_support >= max(3, min_support_lines)
            and raw_blank_len >= min_blank_len
            and end_lateral_width >= min_lateral_width
        )

    confidence = 0.0

    if end_support >= 2:
        confidence += 0.25
    if end_support >= 3:
        confidence += 0.20
    if start_support >= 1:
        confidence += 0.10
    if start_support >= 2:
        confidence += 0.15
    if raw_blank_len >= min_blank_len:
        confidence += 0.15
    if raw_blank_len >= 12.0:
        confidence += 0.10
    if lateral_width >= 2.5:
        confidence += 0.10
    if lateral_width >= 5.0:
        confidence += 0.05
    if end_s_std <= 2.0:
        confidence += 0.08
    if start_s_std <= 2.0:
        confidence += 0.07

    confidence = min(confidence, 1.0)

    # candidate 框直接覆盖空白区。
    s_start = end_s_mean - s_margin_before
    s_end = start_s_mean + s_margin_after

    l_min = raw_l_min - l_margin
    l_max = raw_l_max + l_margin

    return LaneGapJunctionCandidate(
        valid=valid,
        s_start=float(s_start),
        s_end=float(s_end),
        l_min=float(l_min),
        l_max=float(l_max),
        support_line_count=end_support + start_support,
        confidence=confidence,
        gaps=[],
        source=source,
        end_support=end_support,
        start_support=start_support,
        blank_len=float(raw_blank_len),
    )


def detect_collective_structure_junction_candidates(
    lane_lines: Dict[int, np.ndarray],
    observe_s_max: float,
    end_s_tol: float = 4.0,
    start_s_tol: float = 4.0,
    min_support_lines: int = 2,
    min_start_support_lines: int = 1,
    min_lateral_width: float = 2.5,
    min_blank_len: float = 6.0,
    min_lane_visible_len: float = 4.0,
    s_margin_before: float = 1.0,
    s_margin_after: float = 1.0,
    l_margin: float = 1.0,
) -> List[LaneGapJunctionCandidate]:
    """
    新主逻辑：
    从整体车道线结构判断路口候选。

    重点不是“同一条车道线的 gap”，而是：

      前一组纵向线在相近 s 位置集体终止
      +
      后一组纵向线在更远处相近 s 位置重新开始
      =
      中间是路口/冲突区/空白区候选

    如果没有后一组线，也可以退化为 collective_lane_end，
    但这种证据会比 collective_end_start 更保守。
    """
    segments = extract_lane_segment_features(
        lane_lines=lane_lines,
        min_lane_visible_len=min_lane_visible_len,
    )

    if not segments:
        return []

    end_items = []
    start_items = []

    for seg in segments:
        end_items.append(
            {
                "lane_id": seg.lane_id,
                "end_s": seg.s_max,
                "end_l": seg.l_end,
                "l_mean": seg.l_mean,
            }
        )

        start_items.append(
            {
                "lane_id": seg.lane_id,
                "start_s": seg.s_min,
                "start_l": seg.l_start,
                "l_mean": seg.l_mean,
            }
        )

    end_clusters = cluster_by_s(end_items, key="end_s", tol=end_s_tol)
    start_clusters = cluster_by_s(start_items, key="start_s", tol=start_s_tol)

    candidates: List[LaneGapJunctionCandidate] = []

    for end_cluster in end_clusters:
        end_support = len({x["lane_id"] for x in end_cluster})
        if end_support < min_support_lines:
            continue

        end_s_mean = float(np.mean([x["end_s"] for x in end_cluster]))
        end_l_values = [x["end_l"] for x in end_cluster]

        # end cluster 横向覆盖不足，不太像道路整体结束。
        if cluster_lateral_width(end_l_values) < min_lateral_width:
            continue

        # 找这个 end cluster 前方最近的 start cluster。
        best_start_cluster = None
        best_start_s_mean = None

        for start_cluster in start_clusters:
            start_support = len({x["lane_id"] for x in start_cluster})
            if start_support < min_start_support_lines:
                continue

            start_s_mean = float(np.mean([x["start_s"] for x in start_cluster]))
            blank_len = start_s_mean - end_s_mean

            if blank_len < min_blank_len:
                continue

            if best_start_cluster is None or start_s_mean < best_start_s_mean:
                best_start_cluster = start_cluster
                best_start_s_mean = start_s_mean

        if best_start_cluster is not None:
            cand = build_collective_candidate_from_end_and_start(
                end_cluster=end_cluster,
                start_cluster=best_start_cluster,
                observe_s_max=observe_s_max,
                min_support_lines=min_support_lines,
                min_start_support_lines=min_start_support_lines,
                min_lateral_width=min_lateral_width,
                min_blank_len=min_blank_len,
                s_margin_before=s_margin_before,
                s_margin_after=s_margin_after,
                l_margin=l_margin,
            )

            if cand is not None:
                candidates.append(cand)
        else:
            # 没有重新开始的线，则检测“集体终止 + 前方空白”。
            blank_ahead_len = observe_s_max - end_s_mean
            if blank_ahead_len < min_blank_len:
                continue

            cand = build_collective_candidate_from_end_and_start(
                end_cluster=end_cluster,
                start_cluster=None,
                observe_s_max=observe_s_max,
                min_support_lines=min_support_lines,
                min_start_support_lines=min_start_support_lines,
                min_lateral_width=min_lateral_width,
                min_blank_len=min_blank_len,
                s_margin_before=s_margin_before,
                s_margin_after=s_margin_after,
                l_margin=l_margin,
            )

            if cand is not None:
                candidates.append(cand)

    # 简单去重：
    # 多个候选重叠时保留 confidence 高的。
    candidates = suppress_overlapped_candidates(candidates)

    return candidates


def suppress_overlapped_candidates(
    candidates: List[LaneGapJunctionCandidate],
    overlap_ratio_thr: float = 0.5,
) -> List[LaneGapJunctionCandidate]:
    """
    候选框去重。
    """
    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    kept: List[LaneGapJunctionCandidate] = []

    for cand in candidates_sorted:
        if not cand.valid:
            kept.append(cand)
            continue

        duplicate = False

        for kept_cand in kept:
            if not kept_cand.valid:
                continue

            s_overlap = interval_overlap(
                cand.s_start, cand.s_end, kept_cand.s_start, kept_cand.s_end
            )
            l_overlap = interval_overlap(
                cand.l_min, cand.l_max, kept_cand.l_min, kept_cand.l_max
            )

            area_overlap = s_overlap * l_overlap
            area_cand = max(
                1e-6, (cand.s_end - cand.s_start) * (cand.l_max - cand.l_min)
            )
            area_kept = max(
                1e-6,
                (kept_cand.s_end - kept_cand.s_start)
                * (kept_cand.l_max - kept_cand.l_min),
            )

            ratio = area_overlap / min(area_cand, area_kept)

            if ratio >= overlap_ratio_thr:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return kept


def detect_lane_gap_junction_candidates(
    lane_lines: Dict[int, np.ndarray],
    min_gap_len: float = 8.0,
    cluster_s_tol: float = 6.0,
    observe_s_max: float = 45.0,
) -> Tuple[List[LaneLineGap], List[LaneGapJunctionCandidate]]:
    """
    总入口。

    输出：
      gaps:
        旧逻辑检测到的单线 gap。

      candidates:
        包含两类：
          1. gap_cluster
          2. collective_end_start / collective_lane_end
    """
    all_gaps = []

    for lane_id, pts_sl in lane_lines.items():
        gaps = detect_gaps_for_one_lane(
            lane_id=lane_id,
            pts_sl=pts_sl,
            min_gap_len=min_gap_len,
        )
        all_gaps.extend(gaps)

    gap_candidates = cluster_lane_gaps(
        gaps=all_gaps,
        cluster_s_tol=cluster_s_tol,
    )

    structure_candidates = detect_collective_structure_junction_candidates(
        lane_lines=lane_lines,
        observe_s_max=observe_s_max,
        end_s_tol=4.0,
        start_s_tol=4.0,
        min_support_lines=2,
        min_start_support_lines=1,
        min_lateral_width=2.5,
        min_blank_len=6.0,
        min_lane_visible_len=4.0,
        s_margin_before=1.0,
        s_margin_after=1.0,
        l_margin=1.0,
    )

    candidates = gap_candidates + structure_candidates
    candidates = suppress_overlapped_candidates(candidates)

    return all_gaps, candidates


def plot_result(
    lane_lines: Dict[int, np.ndarray],
    gaps: List[LaneLineGap],
    candidates: List[LaneGapJunctionCandidate],
    observe_s_max: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax_raw = axes[0]
    ax_det = axes[1]

    all_s = []
    all_l = []

    for pts in lane_lines.values():
        pts = np.asarray(pts)
        if len(pts) == 0:
            continue
        all_s.extend(pts[:, 0].tolist())
        all_l.extend(pts[:, 1].tolist())

    all_s.append(observe_s_max)

    for gap in gaps:
        all_s.extend([gap.s_start, gap.s_end])
        all_l.append(gap.l_mean)

    for cand in candidates:
        if not cand.valid:
            continue
        all_s.extend([cand.s_start, cand.s_end])
        all_l.extend([cand.l_min, cand.l_max])

    if len(all_s) > 0:
        s_min, s_max = min(all_s), max(all_s)
        l_min, l_max = min(all_l), max(all_l)

        s_margin = max(5.0, 0.05 * max(1.0, s_max - s_min))
        l_margin = max(2.0, 0.10 * max(1.0, l_max - l_min))

        xlim = (s_min - s_margin, s_max + s_margin)
        ylim = (l_min - l_margin, l_max + l_margin)
    else:
        xlim = (-10.0, 100.0)
        ylim = (-10.0, 10.0)

    def draw_lane_lines(ax):
        for lane_id, pts in lane_lines.items():
            pts = np.asarray(pts)
            if len(pts) == 0:
                continue

            pts = pts[np.argsort(pts[:, 0])]

            ax.plot(
                pts[:, 0],
                pts[:, 1],
                "-o",
                markersize=3,
                label=f"lane {lane_id}",
            )

            # start 点
            ax.plot(
                pts[0, 0],
                pts[0, 1],
                marker=">",
                markersize=7,
            )

            # end 点
            ax.plot(
                pts[-1, 0],
                pts[-1, 1],
                marker="x",
                markersize=8,
            )

    def setup_axis(ax, title: str):
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        ax.axvline(
            observe_s_max,
            color="gray",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
        )

        ax.set_title(title)
        ax.set_xlabel("s / x forward [m]")
        ax.set_ylabel("l / y lateral [m]")
        ax.grid(True)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

    draw_lane_lines(ax_raw)
    setup_axis(ax_raw, "Raw lane lines")

    draw_lane_lines(ax_det)

    for gap in gaps:
        ax_det.plot(
            [gap.s_start, gap.s_end],
            [gap.l_mean, gap.l_mean],
            linewidth=5,
            alpha=0.35,
        )

        ax_det.text(
            0.5 * (gap.s_start + gap.s_end),
            gap.l_mean + 0.15,
            f"gap L{gap.lane_id}\n{gap.gap_len:.1f}m",
            fontsize=8,
            ha="center",
        )

    for idx, cand in enumerate(candidates):
        if not cand.valid:
            continue

        rect_s = [cand.s_start, cand.s_end, cand.s_end, cand.s_start, cand.s_start]
        rect_l = [cand.l_min, cand.l_min, cand.l_max, cand.l_max, cand.l_min]

        ax_det.plot(rect_s, rect_l, "--", linewidth=2)

        ax_det.text(
            cand.s_start + 0.4,
            cand.l_max - 0.4,
            f"virtual junction {idx}\n"
            f"{cand.source}\n"
            f"end={cand.end_support}, start={cand.start_support}\n"
            f"blank={cand.blank_len:.1f}m, conf={cand.confidence:.2f}",
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    setup_axis(ax_det, "Detected virtual junction")

    fig.suptitle("Lane structure based virtual junction detection", fontsize=14)

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.10,
        top=0.88,
        wspace=0.20,
    )

    plt.show()


def make_demo_collective_end_start_lane_lines() -> Dict[int, np.ndarray]:
    """
    模拟你截图里的情况：

      左侧三条纵向车道线在 s=9~12m 附近集体结束；
      中间 s=12~28m 基本空白；
      右侧有新的车道线从 s=27~30m 左右开始。

    这应该触发 collective_end_start。
    """
    lane_lines = {
        # 前一组：集体结束
        0: np.array(
            [[0, -3.5], [2, -3.5], [4, -3.5], [8, -3.5], [10, -3.5], [12, -3.5]]
        ),
        1: np.array([[0, 0.0], [2, 0.0], [4, 0.0], [8, 0.0], [9, 0.0]]),
        2: np.array([[0, 3.5], [2, 3.5], [4, 3.5], [6, 3.5], [7, 3.5], [11, 3.5]]),
        # 后一组：重新开始
        3: np.array([[27, 1.0], [29, 1.2], [32, 1.2], [34, 1.3]]),
        4: np.array([[30, 3.0], [32, 2.9], [34, 2.8], [36, 2.8], [38, 3.0]]),
    }

    return lane_lines


def make_demo_collective_end_only_lane_lines() -> Dict[int, np.ndarray]:
    """
    只有前一组车道线集体结束，后方没有重新开始的线。
    这会触发 collective_lane_end，但比 collective_end_start 更弱。
    """
    lane_lines = {
        0: np.array(
            [[0, -3.5], [2, -3.5], [4, -3.5], [8, -3.5], [10, -3.5], [12, -3.5]]
        ),
        1: np.array([[0, 0.0], [2, 0.0], [4, 0.0], [8, 0.0], [9, 0.0]]),
        2: np.array([[0, 3.5], [2, 3.5], [4, 3.5], [6, 3.5], [7, 3.5], [11, 3.5]]),
    }

    return lane_lines


def make_demo_gap_lane_lines() -> Dict[int, np.ndarray]:
    """
    三条车道线中间有 gap，后面又重新出现。
    这个用于测试旧的 gap_cluster 逻辑。
    """
    rng = np.random.default_rng(0)

    lane_lines = {}

    for lane_id, l in enumerate([-3.5, 0.0, 3.5]):
        s_before = np.arange(0, 30, 2.0)
        s_after = np.arange(52, 85, 2.0)

        s = np.concatenate([s_before, s_after])
        l_arr = np.full_like(s, l, dtype=float)
        l_arr += rng.normal(0.0, 0.05, size=len(l_arr))

        lane_lines[lane_id] = np.column_stack([s, l_arr])

    return lane_lines


if __name__ == "__main__":
    # ============================================================
    # 选择测试输入
    # ============================================================

    # Case 1：最接近你截图的结构。
    lane_lines = make_demo_collective_end_start_lane_lines()
    observe_s_max = 45.0

    # Case 2：只有集体结束，没有后续车道线。
    # lane_lines = make_demo_collective_end_only_lane_lines()
    # observe_s_max = 25.0

    # Case 3：同一条车道线中间 gap，测试旧逻辑。
    # lane_lines = make_demo_gap_lane_lines()
    # observe_s_max = 85.0

    # ============================================================
    # 检测
    # ============================================================
    gaps, candidates = detect_lane_gap_junction_candidates(
        lane_lines=lane_lines,
        min_gap_len=8.0,
        cluster_s_tol=6.0,
        observe_s_max=observe_s_max,
    )

    print("Detected gaps:")
    for g in gaps:
        print(
            f"  lane={g.lane_id}, "
            f"s=[{g.s_start:.1f}, {g.s_end:.1f}], "
            f"gap_len={g.gap_len:.1f}, "
            f"l_mean={g.l_mean:.2f}"
        )

    print("\nJunction candidates:")
    for i, c in enumerate(candidates):
        print(
            f"  candidate={i}, "
            f"valid={c.valid}, "
            f"source={c.source}, "
            f"s=[{c.s_start:.1f}, {c.s_end:.1f}], "
            f"l=[{c.l_min:.1f}, {c.l_max:.1f}], "
            f"support={c.support_line_count}, "
            f"end_support={c.end_support}, "
            f"start_support={c.start_support}, "
            f"blank_len={c.blank_len:.1f}, "
            f"confidence={c.confidence:.2f}"
        )

    plot_result(
        lane_lines=lane_lines,
        gaps=gaps,
        candidates=candidates,
        observe_s_max=observe_s_max,
    )
