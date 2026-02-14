from Game.Skill.skill import Skill, SkillType
import random


class FireAttack(Skill):

    def __init__(self):
        super().__init__(name="火鼓齐鸣", skill_type=SkillType.ACTIVE, probability=0.5)

        # 伤害系数
        self.damage_ratio = 1.6
        self.def_reduce_ratio = 0.6  # 对方智力减伤比例

    def on_action(self, caster, allies, enemies, round_id=None):

        # 概率触发判断
        if not self.trigger_check():
            return

        # 选择存活目标
        targets = [e for e in enemies if e.alive]
        if not targets:
            return

        target = random.choice(targets)

        # 谋略伤害公式
        damage = (
            caster.intelligence * self.damage_ratio
            - target.intelligence * self.def_reduce_ratio
        )

        damage = max(0, int(damage))

        # 扣血
        target.hp -= damage

        print(
            f"{caster.name} 触发【{self.name}】对 {target.name} 造成 {damage} 点谋略伤害"
        )

        self.record_damage(caster, damage)

        if target.hp <= 0:
            target.hp = 0
            target.alive = False
            print(f"{target.name} 被击杀")
