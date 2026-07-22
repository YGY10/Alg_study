from __future__ import annotations

import numpy as np
import pytest

from sim2d.planning.spatiotemporal_planner.types import (
    ControlSequence,
    OptimizationResult,
    SpatiotemporalTrajectory,
)


def test_control_sequence_accepts_valid_controls() -> None:
    controls = np.array(
        [
            [1.0, 0.1],
            [0.5, -0.2],
        ],
        dtype=np.float64,
    )

    sequence = ControlSequence(controls=controls)

    assert sequence.controls.shape == (2, 2)
    assert np.allclose(sequence.controls, controls)


def test_control_sequence_copies_input_array() -> None:
    controls = np.zeros((2, 2), dtype=np.float64)

    sequence = ControlSequence(controls=controls)

    controls[0, 0] = 100.0

    assert sequence.controls[0, 0] == 0.0


@pytest.mark.parametrize(
    "controls",
    [
        np.zeros(2, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
    ],
)
def test_control_sequence_rejects_invalid_shape(
    controls: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        ControlSequence(controls=controls)


def test_control_sequence_rejects_non_finite_values() -> None:
    controls = np.array(
        [
            [0.0, 0.0],
            [np.nan, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="non-finite"):
        ControlSequence(controls=controls)


def test_spatiotemporal_trajectory_accepts_valid_data() -> None:
    times = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    states = np.array(
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.2, 0.0, 0.0, 2.0],
            [0.4, 0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    controls = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    trajectory = SpatiotemporalTrajectory(
        times=times,
        states=states,
        controls=controls,
    )

    assert trajectory.times.shape == (3,)
    assert trajectory.states.shape == (3, 4)
    assert trajectory.controls.shape == (2, 2)


def test_spatiotemporal_trajectory_copies_arrays() -> None:
    times = np.array([0.0, 0.1], dtype=np.float64)
    states = np.zeros((2, 4), dtype=np.float64)
    controls = np.zeros((1, 2), dtype=np.float64)

    trajectory = SpatiotemporalTrajectory(
        times=times,
        states=states,
        controls=controls,
    )

    times[1] = 10.0
    states[0, 0] = 10.0
    controls[0, 0] = 10.0

    assert trajectory.times[1] == pytest.approx(0.1)
    assert trajectory.states[0, 0] == pytest.approx(0.0)
    assert trajectory.controls[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("times", "states", "controls"),
    [
        (
            np.array([[0.0, 0.1]], dtype=np.float64),
            np.zeros((2, 4), dtype=np.float64),
            np.zeros((1, 2), dtype=np.float64),
        ),
        (
            np.array([0.0, 0.1], dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((1, 2), dtype=np.float64),
        ),
        (
            np.array([0.0, 0.1], dtype=np.float64),
            np.zeros((2, 4), dtype=np.float64),
            np.zeros((1, 3), dtype=np.float64),
        ),
        (
            np.array([0.0, 0.1, 0.2], dtype=np.float64),
            np.zeros((2, 4), dtype=np.float64),
            np.zeros((1, 2), dtype=np.float64),
        ),
        (
            np.array([0.0, 0.1, 0.2], dtype=np.float64),
            np.zeros((3, 4), dtype=np.float64),
            np.zeros((3, 2), dtype=np.float64),
        ),
    ],
)
def test_spatiotemporal_trajectory_rejects_invalid_shapes(
    times: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        SpatiotemporalTrajectory(
            times=times,
            states=states,
            controls=controls,
        )


def test_spatiotemporal_trajectory_requires_time_start_at_zero() -> None:
    with pytest.raises(ValueError, match="start at zero"):
        SpatiotemporalTrajectory(
            times=np.array([0.1, 0.2], dtype=np.float64),
            states=np.zeros((2, 4), dtype=np.float64),
            controls=np.zeros((1, 2), dtype=np.float64),
        )


def test_spatiotemporal_trajectory_requires_increasing_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SpatiotemporalTrajectory(
            times=np.array([0.0, 0.2, 0.1], dtype=np.float64),
            states=np.zeros((3, 4), dtype=np.float64),
            controls=np.zeros((2, 2), dtype=np.float64),
        )


def test_spatiotemporal_trajectory_rejects_non_finite_values() -> None:
    states = np.zeros((2, 4), dtype=np.float64)
    states[1, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        SpatiotemporalTrajectory(
            times=np.array([0.0, 0.1], dtype=np.float64),
            states=states,
            controls=np.zeros((1, 2), dtype=np.float64),
        )


def test_optimization_result_stores_result_metadata() -> None:
    trajectory = SpatiotemporalTrajectory(
        times=np.array([0.0, 0.1], dtype=np.float64),
        states=np.zeros((2, 4), dtype=np.float64),
        controls=np.zeros((1, 2), dtype=np.float64),
    )

    result = OptimizationResult(
        trajectory=trajectory,
        success=True,
        total_cost=12.5,
        iterations=8,
        status="converged",
        cost_terms={
            "reference": 5.0,
            "control": 7.5,
        },
    )

    assert result.success is True
    assert result.total_cost == pytest.approx(12.5)
    assert result.iterations == 8
    assert result.status == "converged"
