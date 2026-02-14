from Game.Skill.skill import Skill, SkillType
import random


class ShuiYanQiJun(Skill):

    def __init__(self):
        super().__init__(name="水淹七军", skill_type=SkillType.ACTIVE, probability=0.3)

        self.prepare_ratio = 12.0
        self.preparing = False
        self.prepared_target = None

    def on_action(self, caster, allies, enemies, round_id=None):

        # =========================
        # ① 释放阶段
        # =========================
        if self.preparing:

            target = self.prepared_target

            if target is None or not target.alive:
                self.preparing = False
                return False

            damage = caster.force * self.prepare_ratio - target.defense * 0.5
            damage = max(0, int(damage))

            target.hp -= damage

            print(
                f"{caster.name} 释放【水淹七军】对 {target.name} 造成 {damage} 兵刃伤害"
            )

            # 记录触发
            self.record_trigger(caster, round_id)
            self.record_damage(caster, damage)

            # ===== 使用状态容器 =====
            if target.alive:
                target.add_debuff("stunned", 1)
                print(f"{target.name} 受到震慑，无法行动")

            # 击杀判定
            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"{target.name} 被击杀")
                self.record_kill(caster)

            self.preparing = False
            self.prepared_target = None

            return True

        # =========================
        # ② 准备阶段
        # =========================
        if not self.trigger_check():
            return False

        targets = [e for e in enemies if e.alive]
        if not targets:
            return False

        target = random.choice(targets)

        self.preparing = True
        self.prepared_target = target

        print(f"{caster.name} 开始准备【水淹七军】，目标锁定 {target.name}")

        return True
