# Game/Skill/ShenWeiTianJiang.py

from Game.Skill.skill import Skill, SkillType


class ShenWeiTianJiang(Skill):

    def __init__(self):
        super().__init__(
            name="神威天将",
            skill_type=SkillType.PASSIVE,
            probability=1.0,
        )

        self.splash_ratio = 0.8  # 溅射比例（60%）

    # ==========================================
    # 普攻命中后触发
    # ==========================================
    def on_normal_attack(self, caster, target, allies, enemies, round_id=None):

        # 只在普攻时生效
        if not caster.alive:
            return

        other_targets = [e for e in enemies if e.alive and e != target]

        if not other_targets:
            return

        print(f"{caster.name}【神威天将】发动，普攻化为群攻！")

        self.record_trigger(caster, round_id)

        for enemy in other_targets:

            damage = caster.force * self.splash_ratio - enemy.defense * 0.5
            damage = max(0, int(damage))

            enemy.hp -= damage
            self.record_damage(caster, damage)

            print(f"  溅射 {enemy.name} 造成 {damage} 兵刃伤害")

            if enemy.hp <= 0:
                enemy.hp = 0
                enemy.alive = False
                print(f"  {enemy.name} 被击杀")
                self.record_kill(caster)
