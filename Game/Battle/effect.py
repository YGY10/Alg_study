from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from Game.Battle.action import ActionContext, ActionKind, SkillKind
from Game.Battle.damage import DamageCalculator, DamageType


class Effect:
    def apply(self, battle, owner, ctx: ActionContext) -> None:
        raise NotImplementedError


@dataclass
class CancelAction(Effect):
    message: Optional[Callable[[object, ActionContext], str]] = None

    def apply(self, battle, owner, ctx: ActionContext) -> None:
        ctx.cancelled = True
        if self.message:
            battle.log.add(self.message(owner, ctx))


@dataclass
class DealDamage(Effect):
    ratio: float
    damage_type: str = DamageType.WEAPON
    target_getter: Callable[[object, ActionContext], list] = lambda owner, ctx: ctx.targets

    def apply(self, battle, owner, ctx: ActionContext) -> None:
        for target in self.target_getter(owner, ctx):
            if not target.alive:
                continue
            if self.damage_type == DamageType.STRATEGY:
                amount = DamageCalculator.strategy_damage(owner, target, self.ratio)
            elif self.damage_type == DamageType.WEAPON:
                amount = DamageCalculator.weapon_damage(owner, target, self.ratio)
            else:
                amount = max(1, int(self.ratio))
            DamageCalculator.apply_damage(battle, owner, target, amount, self.damage_type, getattr(battle, "current_trigger_skill", None) or ctx.skill)


@dataclass
class QueueAssault(Effect):
    skill: object
    ratio: float

    def apply(self, battle, owner, ctx: ActionContext) -> None:
        target = ctx.targets[0] if ctx.targets else None
        if target is None or not target.alive:
            return
        battle.log.add(f"{owner.name} 普通攻击后触发突击技能【{self.skill.name}】")
        battle.action_queue.append(
            ActionContext(
                actor=owner,
                targets=[target],
                action_kind=ActionKind.SKILL_CAST,
                skill=self.skill,
                skill_kind=SkillKind.ASSAULT,
                tags={"skill", "assault", "weapon_damage"},
                metadata={"ratio": self.ratio},
            )
        )
