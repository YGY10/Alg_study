from Game.Hero.hero import Hero
from Game.Skill.DanQiJiuZhu import DanQiJiuZhu


class ZhaoYun(Hero):

    def __init__(self, level: int, s_n: float, skill1=None, skill2=None):

        super().__init__(
            name="赵云",
            # 基础属性
            base_force=96,
            base_defense=85,
            base_intelligence=75,
            base_speed=88,
            base_charm=70,
            # 成长值
            growth_force=2.5,
            growth_defense=1.8,
            growth_intelligence=1.0,
            growth_speed=1.7,
            growth_charm=0.7,
            level=level,
            solders_num=s_n,
            # 自带技能
            self_skill=DanQiJiuZhu(),
            # 可替换技能
            skill1=skill1,
            skill2=skill2,
        )
