from Game.Skill.skill import Skill, SkillType
import random


class JuShuiDuanQiao(Skill):

    def __init__(self):
        super().__init__(name="拒水断桥", skill_type=SkillType.ACTIVE, probability=0.5)

        self.group_ratio = 1.2
        self.extra_ratio = 2.0
        self.crit_multiplier = 1.5

    def on_action(self, caster, allies, enemies, round_id=None):

        if not self.trigger_check():
            return False

        targets = [e for e in enemies if e.alive]
        if not targets:
            return False

        print(f"{caster.name} 触发【拒水断桥】")

        self.record_trigger(caster, round_id)

        # =========================
        # ① 记录“原本已震慑”目标
        # =========================
        original_stunned = [e for e in enemies if e.alive and e.has_debuff("stunned")]

        # =========================
        # ② 群体伤害
        # =========================
        for target in targets:

            damage = caster.force * self.group_ratio - target.defense * 0.5
            damage = max(0, int(damage))

            target.hp -= damage
            self.record_damage(caster, damage)

            print(f"  对 {target.name} 造成 {damage} 兵刃伤害")

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被击杀")
                self.record_kill(caster)

        # 更新存活
        targets = [e for e in enemies if e.alive]
        if not targets:
            return True

        # =========================
        # ③ 50%概率随机震慑
        # =========================
        if random.random() <= 0.5:
            stun_target = random.choice(targets)
            stun_target.add_debuff("stunned", 1)
            print(f"  {stun_target.name} 被震慑")

        # =========================
        # ④ 若存在震慑目标 → 追加伤害
        # =========================
        stunned_targets = [e for e in enemies if e.alive and e.has_debuff("stunned")]

        if stunned_targets:

            bonus_target = random.choice(stunned_targets)

            bonus_damage = caster.force * self.extra_ratio - bonus_target.defense * 0.5
            bonus_damage = max(0, int(bonus_damage))

            # 若原本已震慑 → 必定暴击
            if bonus_target in original_stunned:
                bonus_damage = int(bonus_damage * self.crit_multiplier)
                print("  触发暴击！")

            bonus_target.hp -= bonus_damage
            self.record_damage(caster, bonus_damage)

            print(f"  对震慑目标 {bonus_target.name} 追加 {bonus_damage} 兵刃伤害")

            if bonus_target.hp <= 0:
                bonus_target.hp = 0
                bonus_target.alive = False
                print(f"  {bonus_target.name} 被击杀")
                self.record_kill(caster)

        return True
