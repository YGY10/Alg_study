from Game.Skill.skill import Skill, SkillType
import random


class LuoFeng(Skill):

    def __init__(self):
        super().__init__(
            name="落凤",
            skill_type=SkillType.ACTIVE,
            probability=0.35,
        )

        self.damage_ratio = 3.8
        self.silence_duration = 1

    def on_action(self, caster, allies, enemies, round_id=None):

        if not self.trigger_check():
            return False

        targets = [e for e in enemies if e.alive]
        if not targets:
            return False

        target = random.choice(targets)

        print(f"{caster.name} 触发【落凤】")

        self.record_trigger(caster, round_id)

        # =========================
        # ① 兵刃伤害
        # =========================
        damage = caster.force * self.damage_ratio - target.defense * 0.5
        damage = max(0, int(damage))

        target.hp -= damage
        self.record_damage(caster, damage)

        print(f"  对 {target.name} 造成 {damage} 兵刃伤害")

        # =========================
        # ② 技穷（封主动）
        # =========================
        if target.alive:
            target.add_debuff("silenced", self.silence_duration)
            print(f"  {target.name} 陷入技穷（无法发动主动技能）")

        # =========================
        # ③ 击杀判定
        # =========================
        if target.hp <= 0:
            target.hp = 0
            target.alive = False
            print(f"  {target.name} 被击杀")
            self.record_kill(caster)

        return True
