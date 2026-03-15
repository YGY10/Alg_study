from Game.Hero.hero import Hero
from Game.Skill.ChenMuHengMao import ChenMuHengMao


class ZhangFei(Hero):

    def __init__(self, level: int, s_n: float, skill1=None, skill2=None):

        super().__init__(
            name="张飞",
            # 基础属性
            base_force=97,
            base_defense=80,
            base_intelligence=65,
            base_speed=82,
            base_charm=60,
            # 成长值
            growth_force=2.7,
            growth_defense=1.6,
            growth_intelligence=0.5,
            growth_speed=1.4,
            growth_charm=0.5,
            level=level,
            solders_num=s_n,
            # 自带技能
            self_skill=ChenMuHengMao(),
            # 可替换技能
            skill1=skill1,
            skill2=skill2,
        )
