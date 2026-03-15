# Game/Skill/CuiFengDuanRen.py
from Game.Skill.skill import Skill, SkillType


class CuiFengDuanRen(Skill):

    def __init__(self):
        super().__init__(
            name="摧锋断刃",
            skill_type=SkillType.ASSAULT,
            probability=0.45,
        )

        self.second_ratio = 2.8
        self.execute_ratio = 10.2

    def on_normal_attack(self, caster, target, allies, enemies, round_id=None):

        if not target.alive:
            return

        print(f"{caster.name} 触发【摧锋断刃】")

        # ===== 第一段追加 =====
        damage = caster.force * self.second_ratio - target.defense * 0.5
        damage = max(0, int(damage))

        target.hp -= damage
        self.record_damage(caster, damage)

        print(f"  追加造成 {damage} 兵刃伤害")

        if target.hp <= 0:
            target.hp = 0
            target.alive = False
            print(f"  {target.name} 被斩杀")
            self.record_kill(caster)
            return

        # ===== 武力压制判定 =====
        if caster.force > target.force:

            extra = caster.force * self.execute_ratio - target.defense * 0.5
            extra = max(0, int(extra))

            target.hp -= extra
            self.record_damage(caster, extra)

            print(f"  武力压制！再造成 {extra} 强力兵刃伤害")

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被强击斩杀")
                self.record_kill(caster)
