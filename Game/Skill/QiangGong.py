# Game/Skill/QiangGong.py
from Game.Skill.skill import Skill, SkillType


class QiangGong(Skill):

    def __init__(self):
        super().__init__(
            name="强攻",
            skill_type=SkillType.ACTIVE,
            probability=0.5,
        )

    # =========================
    # 主动阶段触发
    # =========================
    def on_action(self, caster, allies, enemies, round_id=None):

        if not self.trigger_check():
            return False

        print(f"{caster.name} 触发【强攻】")

        self.record_trigger(caster, round_id)

        # 添加双连击Buff（持续本回合）
        caster.add_buff("combo", 1)

        print(f"  {caster.name} 获得【连击】状态（本回合普攻两次）")

        return False  # 不消耗行动（仍会进行普攻）
