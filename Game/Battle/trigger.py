from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from Game.Battle.action import ActionContext
from Game.Battle.events import EventName


Condition = Callable[[object, object, ActionContext], bool]


@dataclass
class Trigger:
    event_name: EventName
    conditions: list[Condition] = field(default_factory=list)
    effects: list[object] = field(default_factory=list)
    chance: float = 1.0
    priority: int = 0

    def can_fire(self, battle, owner, ctx: ActionContext) -> bool:
        if any(not condition(battle, owner, ctx) for condition in self.conditions):
            return False
        return battle.rng.random() <= self.chance

    def fire(self, battle, owner, ctx: ActionContext, skill=None) -> None:
        previous = getattr(battle, "current_trigger_skill", None)
        battle.current_trigger_skill = skill
        try:
            for effect in self.effects:
                effect.apply(battle, owner, ctx)
        finally:
            battle.current_trigger_skill = previous
