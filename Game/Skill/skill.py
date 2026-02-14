# Game/Skill/skill.py
from enum import Enum
import random


class SkillType(Enum):
    COMMAND = "指挥"
    ACTIVE = "主动"
    ASSAULT = "突击"
    PASSIVE = "被动"


class Skill:
    def __init__(self, name: str, skill_type: SkillType, probability: float = 1.0):
        self.name = name
        self.skill_type = skill_type
        self.probability = probability

    def trigger_check(self) -> bool:
        return random.random() <= self.probability

    # =========================
    # 事件接口
    # =========================
    def on_battle_start(self, caster, allies, enemies):
        pass

    def on_turn_start(self, caster, allies, enemies, round_id=None):
        pass

    def on_action(self, caster, allies, enemies, round_id=None):
        pass

    def on_normal_attack(self, caster, target, allies, enemies, round_id=None):
        pass

    def on_be_hit(self, caster, attacker, damage, round_id=None):
        pass

    # =========================
    # 🔥 统一记账入口
    # =========================
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
