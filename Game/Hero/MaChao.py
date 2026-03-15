from Game.Hero.hero import Hero
from Game.Skill.ShenWeiTianJiang import ShenWeiTianJiang


class MaChao(Hero):

    def __init__(self, level: int, s_n: float, skill1=None, skill2=None):

        super().__init__(
            name="马超",
            # 基础属性
            base_force=96.5,
            base_defense=80,
            base_intelligence=50,
            base_speed=89,
            base_charm=81,
            # 成长值
            growth_force=2.7,
            growth_defense=1.2,
            growth_intelligence=0.5,
            growth_speed=1.6,
            growth_charm=1.0,
            level=level,
            solders_num=s_n,
            # 自带技能
            self_skill=ShenWeiTianJiang(),
            # 可替换技能
            skill1=skill1,
            skill2=skill2,
        )
