from Game.Battle.battle_engine import BattleEngine
from Game.Hero.Guanyu import GuanYu
from Game.Hero.LiuBei import LiuBei
from Game.Hero.ZhangFei import ZhangFei
from Game.Hero.ZhaoYun import ZhaoYun
from Game.Hero.MaChao import MaChao
from Game.Skill.FireAttack import FireAttack
from Game.Skill.QianLiZouDanQi import QianLiZouDanQi
from Game.Skill.JuShuiDuanQiao import JuShuiDuanQiao
from Game.Skill.DanDaoFuHui import DanDaoFuHui
from Game.Skill.LuoFeng import LuoFeng
from Game.Skill.YongLieRuHuo import YongLieRuHuo
from Game.Skill.QiangGong import QiangGong
from Game.Skill.CuiFengDuanRen import CuiFengDuanRen
from Game.Skill.PoCeQiMou import PoCeQiMou
from Game.Hero.hero import Hero


class DummyHero(Hero):

    def __init__(self, level, skill1=None, skill2=None):

        super().__init__(
            name="诸葛亮",
            base_force=80,
            base_defense=75,
            base_intelligence=88,
            base_speed=85,
            base_charm=60,
            growth_force=1.8,
            growth_defense=1.5,
            growth_intelligence=3.0,
            growth_speed=1.0,
            growth_charm=1.0,
            level=level,
            solders_num=10000,
            self_skill=None,
            skill1=skill1,
            skill2=skill2,
        )


def main():

    guan_yu = GuanYu(level=50, s_n=10000, skill1=QianLiZouDanQi(), skill2=DanDaoFuHui())
    zhang_fei = ZhangFei(
        level=50, s_n=10000, skill1=JuShuiDuanQiao(), skill2=YongLieRuHuo()
    )
    liu_bei = LiuBei(level=50, s_n=10000, skill1=FireAttack())

    enemy1 = ZhaoYun(level=50, s_n=10000, skill1=LuoFeng())
    enemy2 = MaChao(level=50, s_n=10000, skill1=CuiFengDuanRen(), skill2=QiangGong())
    enemy3 = DummyHero(level=50, skill1=PoCeQiMou(), skill2=FireAttack())

    engine = BattleEngine(
        team_a=[guan_yu, zhang_fei, liu_bei], team_b=[enemy1, enemy2, enemy3], seed=None
    )

    engine.start_battle()


if __name__ == "__main__":
    main()
