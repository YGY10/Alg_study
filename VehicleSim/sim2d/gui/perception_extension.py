from __future__ import annotations

from sim2d.gui.main_window import MainWindow
from sim2d.perception import PerceptionConfig

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_reset = MainWindow.reset_environment

    def reset_environment(self: MainWindow) -> None:
        original_reset(self)
        self.env.set_perception_map_network(self.road_network)
        snapshot = self.env.get_perception_snapshot()
        self.append_log(
            "PERCEPTION_READY "
            f"source={snapshot.source} "
            f"objects={len(snapshot.objects)} "
            f"signals={len(snapshot.traffic_signals)} "
            f"lane_lines={len(snapshot.lane_lines)} "
            f"road_segments={len(snapshot.road_segments)} "
            f"range_front={snapshot.debug['forward_range']:.1f} "
            f"range_rear={snapshot.debug['rear_range']:.1f} "
            f"range_lateral={snapshot.debug['lateral_range']:.1f}"
        )

    MainWindow.reset_environment = reset_environment
    _INSTALLED = True


__all__ = ["PerceptionConfig", "install"]
