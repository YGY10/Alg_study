from enum import Enum

from Game.Battle.action import SkillKind


class SkillType(Enum):
    COMMAND = "指挥"
    ACTIVE = "主动"
    ASSAULT = "突击"
    PASSIVE = "被动"
    FORMATION = "阵法"


_SKILL_KIND_MAP = {
    SkillType.COMMAND: SkillKind.COMMAND,
    SkillType.ACTIVE: SkillKind.ACTIVE,
    SkillType.ASSAULT: SkillKind.ASSAULT,
    SkillType.PASSIVE: SkillKind.PASSIVE,
    SkillType.FORMATION: SkillKind.FORMATION,
}


class Skill:
    def __init__(self, name: str, skill_type: SkillType, probability: float = 1.0):
        self.name = name
        self.skill_type = skill_type
        self.skill_kind = _SKILL_KIND_MAP[skill_type]
        self.probability = probability
        self.owner = None
        self.triggers = []

    def bind_owner(self, owner):
        self.owner = owner

    def trigger_check(self, battle=None) -> bool:
        rng = battle.rng if battle is not None else None
        if rng is None:
            import random

            return random.random() <= self.probability
        return rng.random() <= self.probability

    # 旧技能兼容钩子。新版 BattleEngine 会把 Action 生命周期转成这些调用。
    def on_battle_start(self, caster, allies, enemies, battle=None):
        pass

    def on_turn_start(self, caster, allies, enemies, round_id=None, battle=None):
        pass

    def on_action(self, caster, allies, enemies, round_id=None, battle=None):
        pass

    def on_normal_attack(self, caster, target, allies, enemies, round_id=None, battle=None):
        pass

    def on_be_hit(self, caster, attacker, damage, round_id=None, battle=None):
        pass

    def execute(self, battle, ctx):
        # 默认技能仍走旧 on_action 实现，便于保留现有技能效果。
        allies = battle.allies_of(ctx.actor)
        enemies = battle.enemies_of(ctx.actor)
        return bool(self.on_action(ctx.actor, allies, enemies, battle.round_id, battle=battle))

    def record_trigger(self, caster, round_id):
        caster.record_skill_trigger(self.name, round_id)

    def record_damage(self, caster, amount: int):
        amount = max(0, int(amount))
        caster.total_damage += amount
        caster.record_skill_damage(self.name, amount)

    def record_heal(self, caster, amount: int):
        amount = max(0, int(amount))
        caster.total_heal += amount
        caster.record_skill_heal(self.name, amount)

    def record_kill(self, caster):
        caster.kill_count += 1
        caster.record_skill_kill(self.name)
