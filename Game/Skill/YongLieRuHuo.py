# Game/Skill/YongLieRuHuo.py

from Game.Skill.skill import Skill, SkillType


class YongLieRuHuo(Skill):

    def __init__(self):
        super().__init__(name="勇烈如火", skill_type=SkillType.PASSIVE)

        self.max_stack = 10
        self.stack_damage_ratio = 0.8  # 每层倍率

    # =========================
    # ① 被普通攻击叠怒气
    # =========================
    def on_be_hit(self, caster, attacker, damage, round_id=None):

        if not caster.alive:
            return

        if not hasattr(caster, "rage_stack"):
            caster.rage_stack = 0

        if caster.rage_stack < self.max_stack:
            caster.rage_stack += 1
            print(f"{caster.name}【勇烈如火】怒气+1（当前{caster.rage_stack}层）")

    # =========================
    # ② 行动前爆发
    # =========================
    def on_turn_start(self, caster, allies, enemies, round_id=None):

        rage = getattr(caster, "rage_stack", 0)

        if rage <= 0:
            return False

        print(f"{caster.name}【勇烈如火】爆发！怒气{rage}层")

        self.record_trigger(caster, round_id)

        total_ratio = rage * self.stack_damage_ratio

        for target in [e for e in enemies if e.alive]:

            damage = caster.force * total_ratio - target.defense * 0.5
            damage = max(0, int(damage))

            target.hp -= damage
            self.record_damage(caster, damage)

            print(f"  对 {target.name} 造成 {damage} 兵刃伤害")

            if target.hp <= 0:
                target.hp = 0
                target.alive = False
                print(f"  {target.name} 被击杀")
                self.record_kill(caster)

        # 不清零怒气（成长型）
        return False
