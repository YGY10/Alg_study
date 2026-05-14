# perception/base_perception.py

from perception.perception_output import (
    BEVPerceptionInput,
    BEVPerceptionOutput,
)


class BasePerception:
    def process(self, perception_input: BEVPerceptionInput) -> BEVPerceptionOutput:
        raise NotImplementedError