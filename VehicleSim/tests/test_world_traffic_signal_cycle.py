from sim2d.world.traffic_signal import (
    TrafficLightState,
    WorldTrafficSignal,
)


def make_signal(**kwargs) -> WorldTrafficSignal:
    return WorldTrafficSignal(
        entity_id="signal",
        map_signal_id="map_signal",
        x=0.0,
        y=0.0,
        yaw=0.0,
        red_duration=12.0,
        green_duration=10.0,
        yellow_duration=3.0,
        **kwargs,
    )


def test_cycle_starts_at_red():
    signal = make_signal(phase_offset=0.0)

    assert signal.state is TrafficLightState.RED
    assert signal.remaining_time == 12.0


def test_cycle_advances_red_green_yellow_red():
    signal = make_signal(phase_offset=0.0)

    signal.advance(12.0)
    assert signal.state is TrafficLightState.GREEN
    assert signal.remaining_time == 10.0

    signal.advance(10.0)
    assert signal.state is TrafficLightState.YELLOW
    assert signal.remaining_time == 3.0

    signal.advance(3.0)
    assert signal.state is TrafficLightState.RED
    assert signal.remaining_time == 12.0


def test_cycle_handles_large_time_step():
    signal = make_signal(phase_offset=0.0)

    signal.advance(27.0)

    assert signal.state is TrafficLightState.RED
    assert signal.remaining_time == 10.0


def test_phase_offset_staggers_signals():
    red_signal = make_signal(phase_offset=0.0)
    green_signal = make_signal(phase_offset=12.5)
    yellow_signal = make_signal(phase_offset=22.5)

    assert red_signal.state is TrafficLightState.RED
    assert green_signal.state is TrafficLightState.GREEN
    assert yellow_signal.state is TrafficLightState.YELLOW
