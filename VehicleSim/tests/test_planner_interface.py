import pytest

from sim2d.planning import Planner


def test_planner_is_abstract() -> None:
    with pytest.raises(TypeError):
        Planner()


def test_incomplete_planner_cannot_be_created() -> None:
    class IncompletePlanner(Planner):
        pass

    with pytest.raises(TypeError):
        IncompletePlanner()
