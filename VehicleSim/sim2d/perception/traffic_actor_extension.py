from __future__ import annotations

from dataclasses import replace

from sim2d.perception.ground_truth import GroundTruthLocalPerception

_INSTALLED = False


def install() -> None:
    """让现有局部真值感知输出交通参与者语义和真实速度。"""
    global _INSTALLED
    if _INSTALLED:
        return

    original = GroundTruthLocalPerception._perceive_obstacle

    def perceive(self, obstacle, ego):
        result = original(self, obstacle, ego)
        semantic_type = str(getattr(obstacle, "semantic_type", "unknown"))
        speed = float(getattr(obstacle, "speed", 0.0))

        # 圆形行人也有初始朝向。原实现对所有圆形障碍物固定输出 yaw=0。
        if result.object_type == "circle" and hasattr(obstacle, "yaw"):
            relative_yaw = self._normalize_angle(float(obstacle.yaw) - ego.yaw)
        else:
            relative_yaw = result.yaw

        return replace(
            result,
            yaw=relative_yaw,
            speed=speed,
            semantic_type=semantic_type,
        )

    GroundTruthLocalPerception._perceive_obstacle = perceive
    _INSTALLED = True


__all__ = ["install"]
