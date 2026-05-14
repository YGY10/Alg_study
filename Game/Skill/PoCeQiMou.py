from Game.Battle.action import ActionKind, SkillKind
from Game.Battle.damage import DamageCalculator, DamageType
from Game.Battle.effect import Effect
from Game.Battle.events import EventName
from Game.Battle.trigger import Trigger
from Game.Skill.skill import Skill, SkillType


class InterruptAndDamage(Effect):
    def __init__(self, ratio=8.6):
        self.ratio = ratio

    def apply(self, battle, owner, ctx):
        ctx.cancelled = True
        amount = DamageCalculator.strategy_damage(owner, ctx.actor, self.ratio)
        battle.log.add(
            f"由于 {owner.name} 触发被动技能【破策奇谋】，"
            f"打断 {ctx.actor.name} 的主动技能【{ctx.skill.name}】"
        )
        DamageCalculator.apply_damage(
            battle,
            owner,
            ctx.actor,
            amount,
            DamageType.STRATEGY,
            getattr(battle, "current_trigger_skill", None),
        )


class PoCeQiMou(Skill):
    def __init__(self):
        super().__init__(name="破策奇谋", skill_type=SkillType.PASSIVE, probability=1.0)
        self.triggers = [
            Trigger(
                event_name=EventName.ACTION_BEFORE_RESOLVE,
                conditions=[
                    lambda battle, owner, ctx: owner.alive,
                    lambda battle, owner, ctx: ctx.action_kind == ActionKind.SKILL_CAST,
                    lambda battle, owner, ctx: ctx.skill_kind == SkillKind.ACTIVE,
                    lambda battle, owner, ctx: ctx.skill is not None,
                    lambda battle, owner, ctx: battle.is_enemy(owner, ctx.actor),
                    lambda battle, owner, ctx: not ctx.cancelled,
                ],
                chance=0.45,
                effects=[InterruptAndDamage()],
                priority=100,
            )
        ]
