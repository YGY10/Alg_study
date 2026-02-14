# Game/Skill/DeZhaoZhiLie.py

from Game.Skill.skill import Skill, SkillType
import random


class DeZhaoZhiLie(Skill):

    def __init__(self):
        super().__init__(
            name="德昭志烈",
            skill_type=SkillType.ACTIVE,
            probability=0.5,
        )

        self.force_ratio = 1.2
        self.int_ratio = 1.0
        self.heal_ratio = 1.5

    def on_action(self, caster, allies, enemies, round_id=None):

        if not self.trigger_check():
            return False

        alive_enemies = [e for e in enemies if e.alive]
        alive_allies = [a for a in allies if a.alive]

        if not alive_enemies:
            return False

        print(f"{caster.name} 触发【德昭志烈】")

        self.record_trigger(caster, round_id)

        # =========================
        # ① 敌方双伤害
        # =========================
        targets = random.sample(alive_enemies, min(2, len(alive_enemies)))

        for target in targets:

            force_damage = caster.force * self.force_ratio - target.defense * 0.5
            force_damage = max(0, int(force_damage))

            int_damage = (
                caster.intelligence * self.int_ratio - target.intelligence * 0.4
            )
            int_damage = max(0, int(int_damage))

            total_damage = force_damage + int_damage

            target.hp -= total_damage

            self.record_damage(caster, total_damage)

            print(
                f"  对 {target.name} 造成 {force_damage}兵刃 + {int_damage}谋略 = {total_damage} 伤害"
            )

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被击杀")
                self.record_kill(caster)

        # =========================
        # ② 治疗
        # =========================
        if alive_allies:

            heal_targets = random.sample(alive_allies, min(2, len(alive_allies)))

            for ally in heal_targets:

                heal_amount = int(caster.intelligence * self.heal_ratio)

                # 实际可恢复量
                actual_heal = min(heal_amount, ally.max_hp - ally.hp)

                ally.hp += actual_heal

                print(f"  为 {ally.name} 恢复 {actual_heal} 兵力")

                # 🔥 关键：统一记账入口
                self.record_heal(caster, actual_heal)

        return True
