from sim2d.types import VehicleState

from sim2d.world import (
    World,
    WorldState,
)


def test_world_step():

    state = WorldState(
        time=0.0,
        ego_state=VehicleState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=0.0,
        ),
    )

    world = World(
        state=state,
    )

    world.step(0.1)

    assert world.state.time == 0.1
