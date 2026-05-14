from __future__ import annotations

from Game.Battle.action import ActionContext, ActionKind
from Game.Battle.events import EventName
from Game.Hero.buff import BuffType


class DamageType:
    WEAPON = "兵刃"
    STRATEGY = "谋略"
    TRUE = "真实"


class DamageCalculator:
    @staticmethod
    def weapon_damage(attacker, defender, ratio: float) -> int:
        return max(1, int(attacker.force * ratio - defender.defense * 0.5))

    @staticmethod
    def strategy_damage(attacker, defender, ratio: float) -> int:
        return max(1, int(attacker.intelligence * ratio - defender.intelligence * 0.3))

    @staticmethod
    def apply_damage(battle, attacker, defender, amount: int, damage_type: str, skill=None):
        ctx = ActionContext(
            actor=attacker,
            targets=[defender],
            action_kind=ActionKind.DAMAGE,
            skill=skill,
            skill_kind=getattr(skill, "skill_kind", None),
            tags={"damage", damage_type},
            metadata={"amount": max(1, int(amount)), "damage_type": damage_type},
        )
        battle.emit(EventName.BEFORE_DAMAGE, ctx)
        if ctx.cancelled:
            battle.log.add(f"{defender.name} 本次伤害被免疫")
            return 0

        final_amount = max(1, int(ctx.metadata["amount"]))
        defender.hp -= final_amount
        battle.record_damage(attacker, skill, final_amount)
        battle.log.add(f"{defender.name} 受到 {final_amount} 点{damage_type}伤害")

        if defender.hp <= 0:
            defender.hp = 0
            defender.alive = False
            battle.record_kill(attacker, skill)
            battle.log.add(f"{defender.name} 被击杀")

        ctx.result["damage"] = final_amount
        battle.emit(EventName.AFTER_DAMAGE, ctx)
        return final_amount

    @staticmethod
    def apply_heal(battle, healer, target, amount: int, skill=None):
        ctx = ActionContext(
            actor=healer,
            targets=[target],
            action_kind=ActionKind.HEAL,
            skill=skill,
            skill_kind=getattr(skill, "skill_kind", None),
            tags={"heal"},
            metadata={"amount": max(0, int(amount))},
        )
        battle.emit(EventName.BEFORE_HEAL, ctx)
        if ctx.cancelled:
            return 0

        actual = min(max(0, int(ctx.metadata["amount"])), target.max_hp - target.hp)
        target.hp += actual
        battle.record_heal(healer, skill, actual)
        battle.log.add(f"{healer.name} 为 {target.name} 恢复 {actual} 兵力")
        ctx.result["heal"] = actual
        battle.emit(EventName.AFTER_HEAL, ctx)
        return actual


def damage_modifier_from_buffs(ctx: ActionContext) -> None:
    amount = ctx.metadata.get("amount", 0)
    attacker = ctx.actor
    defender = ctx.targets[0] if ctx.targets else None

    if attacker is not None:
        amount *= 1 + attacker.buff_value(BuffType.DMG_UP)
        amount *= 1 - attacker.buff_value(BuffType.DMG_DOWN)

    if defender is not None:
        amount *= 1 - defender.buff_value(BuffType.DEF_UP)
        amount *= 1 + defender.buff_value(BuffType.DEF_DOWN)

    ctx.metadata["amount"] = max(1, int(amount))
