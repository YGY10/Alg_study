from Game.Hero.hero import Hero
from Game.Skill.ShuiYanQiJun import ShuiYanQiJun


class GuanYu(Hero):

    def __init__(self, level: int, s_n: float, skill1=None, skill2=None):

        super().__init__(
            name="关羽",
            # 基础属性
            base_force=96,
            base_defense=85,
            base_intelligence=60,
            base_speed=88,
            base_charm=80,
            # 成长值
            growth_force=2.6,
            growth_defense=1.8,
            growth_intelligence=0.8,
            growth_speed=1.5,
            growth_charm=1.2,
            level=level,
            solders_num=s_n,
            # 自带技能
            self_skill=ShuiYanQiJun(),
            # 可替换技能
            skill1=skill1,
            skill2=skill2,
        )
