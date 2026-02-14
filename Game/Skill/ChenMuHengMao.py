from Game.Skill.skill import Skill, SkillType


class ChenMuHengMao(Skill):

    def __init__(self):
        super().__init__(
            name="瞋目横矛",
            skill_type=SkillType.PASSIVE,
            probability=1.0,
        )

        self.damage_ratio = 1.8

    def on_action(self, caster, allies, enemies, round_id=None):

        # 只在偶数回合触发
        if round_id is None or round_id % 2 != 0:
            return False

        targets = [e for e in enemies if e.alive]
        if not targets:
            return False

        print(f"{caster.name} 触发【瞋目横矛】（偶数回合）")

        # 🔥 记录触发
        self.record_trigger(caster, round_id)

        for target in targets:

            damage = caster.force * self.damage_ratio - target.defense * 0.5
            damage = max(0, int(damage))

            target.hp -= damage
            self.record_damage(caster, damage)
            print(f"  对 {target.name} 造成 {damage} 兵刃伤害")

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被击杀")

                caster.kill_count += 1
                self.record_kill(caster)

        return False  # 被动不消耗行动
