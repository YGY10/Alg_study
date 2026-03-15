from Game.Skill.skill import Skill, SkillType
import random


class QianLiZouDanQi(Skill):

    def __init__(self):
        super().__init__(
            name="千里走单骑", skill_type=SkillType.ACTIVE, probability=0.55
        )

        self.damage_ratio = 5.5

    def on_action(self, caster, allies, enemies, round_id=None):

        if not self.trigger_check():
            return False

        targets = [e for e in enemies if e.alive]
        if not targets:
            return False

        print(f"{caster.name} 触发【千里走单骑】")

        self.record_trigger(caster, round_id)

        # 免疫控制一回合
        caster.stunned = 0

        # 攻击敌方攻击最高单位
        target = max(targets, key=lambda x: x.force)

        damage = caster.force * self.damage_ratio - target.defense * 0.5
        damage = max(0, int(damage))

        target.hp -= damage
        self.record_damage(caster, damage)

        print(f"  对拦路之敌 {target.name} 造成 {damage} 兵刃伤害")

        if target.hp <= 0:
            target.hp = 0
            target.alive = False
            print(f"  {target.name} 被斩杀")

            caster.kill_count += 1
            self.record_kill(caster)

        return True
