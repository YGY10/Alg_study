# Game/Skill/DanQiJiuZhu.py
from Game.Skill.skill import Skill, SkillType
import random


class DanQiJiuZhu(Skill):

    def __init__(self):
        super().__init__(
            name="单骑救主·七进七出",
            skill_type=SkillType.PASSIVE,
            probability=1.0,
        )

        self.hit_ratio = 1.8  # 每段伤害倍率
        self.hit_count = 7  # 七进七出
        self.trigger_hp_ratio = 0.5  # 触发血量阈值（50%）

    # =====================================================
    # 每回合行动前判定
    # =====================================================
    def on_turn_start(self, caster, allies, enemies, round_id=None):

        if not caster.alive:
            return

        # 🔥 永久免疫控制
        caster.stunned = 0

        # 判断是否存在低血量友军
        low_hp_exists = any(
            a.alive and a.hp <= a.max_hp * self.trigger_hp_ratio for a in allies
        )

        if not low_hp_exists:
            return

        print(f"{caster.name} 触发【单骑救主·七进七出】")

        self.record_trigger(caster, round_id)

        # 七段独立攻击
        for _ in range(self.hit_count):

            alive_enemies = [e for e in enemies if e.alive]
            if not alive_enemies:
                break

            target = random.choice(alive_enemies)

            damage = caster.force * self.hit_ratio - target.defense * 0.5
            damage = max(0, int(damage))

            target.hp -= damage
            self.record_damage(caster, damage)

            print(f"  突击 {target.name} 造成 {damage} 兵刃伤害")

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被斩杀")
                self.record_kill(caster)
