from Game.Hero.hero import Hero
from Game.Skill.DeZhaoZhiLie import DeZhaoZhiLie


class LiuBei(Hero):

    def __init__(self, level: int, s_n: float, skill1=None, skill2=None):

        super().__init__(
            name="刘备",
            # 基础属性
            base_force=75,
            base_defense=88,
            base_intelligence=80,
            base_speed=68,
            base_charm=90,
            # 成长值
            growth_force=1.1,
            growth_defense=1.9,
            growth_intelligence=1.3,
            growth_speed=1.1,
            growth_charm=2.2,
            level=level,
            solders_num=s_n,
            # 自带技能
            self_skill=DeZhaoZhiLie(),
            # 可替换技能
            skill1=skill1,
            skill2=skill2,
        )
