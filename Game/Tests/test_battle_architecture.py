from Game.Battle.action import ActionContext, ActionKind
from Game.Battle.battle_engine import BattleEngine
from Game.Battle.damage import DamageCalculator, DamageType
from Game.Hero.buff import Buff, BuffType
from Game.Hero.hero import Hero
from Game.Skill.PoCeQiMou import PoCeQiMou
from Game.Skill.ZhuiJi import ZhuiJi
from Game.Skill.skill import Skill, SkillType


class DummyHero(Hero):
    def __init__(self, name, side_skill=None, skill1=None, hp=1000, speed=50):
        super().__init__(
            name=name,
            base_force=100,
            base_defense=50,
            base_intelligence=100,
            base_speed=speed,
            base_charm=50,
            growth_force=0,
            growth_defense=0,
            growth_intelligence=0,
            growth_speed=0,
            growth_charm=0,
            level=1,
            solders_num=hp,
            self_skill=side_skill,
            skill1=skill1,
        )


class ActiveStrike(Skill):
    def __init__(self, name="火计", ratio=1.5):
        super().__init__(name=name, skill_type=SkillType.ACTIVE, probability=1.0)
        self.ratio = ratio
        self.executed = False

    def on_action(self, caster, allies, enemies, round_id=None, battle=None):
        self.executed = True
        self.record_trigger(caster, round_id)
        target = next(e for e in enemies if e.alive)
        damage = DamageCalculator.strategy_damage(caster, target, self.ratio)
        if battle is not None:
            battle.log.add(f"{caster.name} 发动【{self.name}】")
            DamageCalculator.apply_damage(
                battle, caster, target, damage, DamageType.STRATEGY, self
            )
        else:
            target.hp -= damage
            self.record_damage(caster, damage)
        return True


class PassiveShell(Skill):
    def __init__(self):
        super().__init__(name="空被动", skill_type=SkillType.PASSIVE)


def test_normal_attack_deals_damage():
    a = DummyHero("张三", PassiveShell())
    b = DummyHero("李四", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)

    battle.resolve_action(ActionContext(a, [b], ActionKind.NORMAL_ATTACK))

    assert b.hp < b.max_hp
    assert a.total_damage > 0


def test_active_skill_executes():
    skill = ActiveStrike()
    a = DummyHero("张三", skill)
    b = DummyHero("李四", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)

    assert battle.try_active_skills(a) is True

    assert skill.executed is True
    assert b.hp < b.max_hp


def test_interrupt_cancels_enemy_active_skill_and_deals_damage():
    interrupt = PoCeQiMou()
    active = ActiveStrike()
    a = DummyHero("谋士", interrupt, speed=60)
    b = DummyHero("敌将", active, speed=50)
    battle = BattleEngine([a], [b], seed=1)

    assert battle.try_active_skills(b) is True

    assert active.executed is False
    assert b.hp < b.max_hp
    assert any("打断" in line for line in battle.log.entries)


def test_cancelled_active_skill_does_not_apply_original_damage():
    interrupt = PoCeQiMou()
    active = ActiveStrike(ratio=10.0)
    a = DummyHero("谋士", interrupt, hp=1000)
    b = DummyHero("敌将", active, hp=1000)
    battle = BattleEngine([a], [b], seed=1)

    battle.try_active_skills(b)

    assert active.executed is False
    assert a.hp == a.max_hp


def test_assault_trigger_queues_action_after_normal_attack():
    assault = ZhuiJi()
    a = DummyHero("武将", PassiveShell(), skill1=assault)
    b = DummyHero("敌将", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)

    battle.resolve_action(ActionContext(a, [b], ActionKind.NORMAL_ATTACK))

    assert battle.action_queue
    queued = battle.action_queue[0]
    assert queued.skill_kind.value == "突击"


def test_buff_duration_decreases_and_removes():
    a = DummyHero("张三", PassiveShell())
    b = DummyHero("李四", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)
    a.add_buff_obj(Buff("增伤", BuffType.DMG_UP, duration=1, value=0.2))

    battle.resolve_round_end_buffs()

    assert not a.has_buff_type(BuffType.DMG_UP)


def test_stun_blocks_action():
    a = DummyHero("张三", ActiveStrike())
    b = DummyHero("李四", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)
    a.add_buff_obj(Buff("眩晕", BuffType.STUN, duration=1))

    battle.execute_turn(a)

    assert b.hp == b.max_hp
    assert any("无法行动" in line for line in battle.log.entries)


def test_silence_blocks_active_but_allows_normal_attack():
    active = ActiveStrike()
    a = DummyHero("张三", active)
    b = DummyHero("李四", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)
    a.add_buff_obj(Buff("沉默", BuffType.SILENCE, duration=1))

    battle.execute_turn(a)

    assert active.executed is False
    assert b.hp < b.max_hp


def test_battle_log_has_key_messages():
    a = DummyHero("张三", PassiveShell())
    b = DummyHero("李四", PassiveShell())
    battle = BattleEngine([a], [b], seed=1)

    battle.start_battle(print_log=False)

    text = "\n".join(battle.log.entries)
    assert "战斗开始" in text
    assert "第 1 回合" in text
    assert "普通攻击" in text
