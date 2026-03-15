from Game.Skill.skill import Skill, SkillType


class DanDaoFuHui(Skill):

    def __init__(self):
        super().__init__(name="单刀赴会", skill_type=SkillType.PASSIVE, probability=1.0)

        self.damage_ratio = 0.8
        self.attack_bonus_ratio = 0.1

    def on_action(self, caster, allies, enemies, round_id=None):

        targets = [e for e in enemies if e.alive]
        if not targets:
            return False

        print(f"{caster.name} 触发【单刀赴会】")

        self.record_trigger(caster, round_id)

        # =========================
        # 群体伤害 + 低血震慑
        # =========================
        for target in targets:

            damage = caster.force * self.damage_ratio - target.defense * 0.5
            damage = max(0, int(damage))

            target.hp -= damage
            self.record_damage(caster, damage)

            print(f"  对 {target.name} 造成 {damage} 兵刃伤害")

            # 低于50% → 震慑1回合
            if target.alive and target.hp <= target.max_hp * 0.5:
                target.add_debuff("stunned", 1)
                print(f"  {target.name} 被震慑")

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被击杀")
                self.record_kill(caster)

        # =========================
        # 若存在震慑单位 → 自身攻击提升
        # =========================
        if any(e.alive and e.has_debuff("stunned") for e in enemies):

            bonus = caster.force * self.attack_bonus_ratio
            caster.force += bonus

            print(f"  威震群雄！{caster.name} 攻击提升 {int(bonus)}")

        return True
