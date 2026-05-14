import random
import sys
from contextlib import redirect_stdout

from Game.Battle.action import ActionContext, ActionKind, SkillKind
from Game.Battle.damage import DamageCalculator, DamageType, damage_modifier_from_buffs
from Game.Battle.events import EventName
from Game.Battle.log import BattleLog
from Game.Battle.target import TargetSelector
from Game.Hero.buff import BuffType
from Game.Skill.skill import Skill, SkillType


class _LogWriter:
    def __init__(self, log):
        self.log = log
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line:
                self.log.add(line)

    def flush(self):
        if self.buffer:
            self.log.add(self.buffer)
            self.buffer = ""


class BattleEngine:
    def __init__(self, team_a, team_b, seed=None, max_rounds=8):
        self.team_a = team_a
        self.team_b = team_b
        self.max_rounds = max_rounds
        self.round_id = 1
        self.rng = random.Random(seed)
        self.log = BattleLog()
        self.action_queue = []
        self.normal_skill = Skill("普通攻击", SkillType.ACTIVE)

        for hero in self.team_a:
            hero.side = "A"
            hero.bind_skills()
        for hero in self.team_b:
            hero.side = "B"
            hero.bind_skills()

    def alive_units(self, team):
        return [h for h in team if h.alive]

    def allies_of(self, hero):
        return self.team_a if hero in self.team_a else self.team_b

    def enemies_of(self, hero):
        return self.team_b if hero in self.team_a else self.team_a

    def is_enemy(self, left, right):
        return left is not right and left.side != right.side

    def battle_over(self):
        return not self.alive_units(self.team_a) or not self.alive_units(self.team_b)

    def print_team_status(self):
        def format_team(team):
            return " | ".join(f"{h.name}:{h.hp}/{h.max_hp}({'存活' if h.alive else '阵亡'})" for h in team)

        self.log.add("我方 => " + format_team(self.team_a))
        self.log.add("敌方 => " + format_team(self.team_b))

    def start_battle(self, print_log=True):
        with redirect_stdout(_LogWriter(self.log)):
            self._run_battle()
        if print_log:
            self.log.print()
        return self.log

    def _run_battle(self):
        self.log.add("========== 战斗开始 ==========")
        self.emit(EventName.BATTLE_START, self.empty_context())
        self.print_team_status()

        while not self.battle_over() and self.round_id <= self.max_rounds:
            self.log.add(f"\n======== 第 {self.round_id} 回合 ========")
            self.emit(EventName.ROUND_START, self.empty_context())
            self.execute_round()
            self.emit(EventName.ROUND_END, self.empty_context())
            self.resolve_round_end_buffs()
            self.print_team_status()
            self.round_id += 1

        self.log.add("\n========== 战斗结束 ==========")
        self.emit(EventName.BATTLE_END, self.empty_context())
        self.log_result()

    def empty_context(self):
        actor = self.alive_units(self.team_a + self.team_b)
        actor = actor[0] if actor else (self.team_a + self.team_b)[0]
        return ActionContext(actor=actor, targets=[], action_kind=ActionKind.SKILL_CAST)

    def log_result(self):
        team_a_alive = len(self.alive_units(self.team_a))
        team_b_alive = len(self.alive_units(self.team_b))
        if team_a_alive > 0 and team_b_alive == 0:
            self.log.add("我方胜利")
        elif team_b_alive > 0 and team_a_alive == 0:
            self.log.add("敌方胜利")
        else:
            self.log.add("平局")

        self.log.add("\n========== 战斗统计 ==========")
        for hero in self.team_a + self.team_b:
            self.log.add(f"\n{hero.name}：")
            self.log.add(f"  总输出伤害：{hero.total_damage}")
            self.log.add(f"  总治疗量：{hero.total_heal}")
            if not hero.skill_stats:
                self.log.add("  无技能触发记录")
                continue
            for skill_name, stat in hero.skill_stats.items():
                self.log.add(f"  技能【{skill_name}】:")
                self.log.add(f"    触发回合：{stat['trigger_rounds']}")
                self.log.add(f"    技能输出伤害：{stat['total_damage']}")
                self.log.add(f"    技能治疗量：{stat['total_heal']}")
                self.log.add(f"    技能击杀数：{stat['kill_count']}")

    def execute_round(self):
        all_units = self.alive_units(self.team_a) + self.alive_units(self.team_b)
        self.rng.shuffle(all_units)
        all_units.sort(key=lambda h: h.speed, reverse=True)

        for hero in all_units:
            if not hero.alive:
                continue
            self.execute_turn(hero)
            if self.battle_over():
                break

    def execute_turn(self, hero):
        self.emit(EventName.TURN_START, ActionContext(hero, [], ActionKind.SKILL_CAST))
        self.run_turn_start_passives(hero)

        if not hero.alive:
            return
        if hero.has_buff_type(BuffType.STUN) or hero.has_debuff("stunned"):
            self.log.add(f"{hero.name} 受到眩晕，无法行动")
            self.emit(EventName.TURN_END, ActionContext(hero, [], ActionKind.SKILL_CAST))
            return

        acted = self.try_active_skills(hero)
        if not acted:
            if hero.has_debuff("disarmed"):
                self.log.add(f"{hero.name} 处于缴械状态，无法普通攻击")
            else:
                self.resolve_normal_attacks(hero)

        self.drain_action_queue()
        self.emit(EventName.TURN_END, ActionContext(hero, [], ActionKind.SKILL_CAST))

    def try_active_skills(self, hero):
        if hero.has_buff_type(BuffType.SILENCE) or hero.has_debuff("silenced"):
            self.log.add(f"{hero.name} 处于沉默状态，无法发动主动技能")
            return False

        for skill in hero.all_skills():
            if skill.skill_kind != SkillKind.ACTIVE:
                continue
            ctx = ActionContext(
                actor=hero,
                targets=[],
                action_kind=ActionKind.SKILL_CAST,
                skill=skill,
                skill_kind=SkillKind.ACTIVE,
                tags={"skill", "active"},
            )
            self.resolve_action(ctx)
            if ctx.result.get("consumed_action"):
                return True
        return False

    def resolve_normal_attacks(self, hero):
        attack_times = 2 if hero.has_buff_type(BuffType.COMBO) or hero.has_buff("combo") else 1
        if attack_times == 2:
            self.log.add(f"{hero.name} 触发连击，本回合普攻两次")
        for _ in range(attack_times):
            if not self.alive_units(self.enemies_of(hero)):
                break
            target = TargetSelector.random_enemy(self, hero)
            if target is None:
                break
            ctx = ActionContext(
                actor=hero,
                targets=[target],
                action_kind=ActionKind.NORMAL_ATTACK,
                tags={"normal_attack", "weapon_damage"},
            )
            self.resolve_action(ctx)
            self.drain_action_queue()
        hero.remove_buff("combo")

    def drain_action_queue(self):
        while self.action_queue:
            ctx = self.action_queue.pop(0)
            if ctx.actor.alive:
                self.resolve_action(ctx)

    def resolve_action(self, ctx):
        self.emit(EventName.ACTION_INTENT, ctx)
        self.emit(EventName.ACTION_CHECK, ctx)
        if ctx.cancelled:
            self.log.add("行动在检查阶段被取消")
            return ctx

        if (
            ctx.action_kind == ActionKind.SKILL_CAST
            and ctx.skill_kind == SkillKind.ACTIVE
            and ctx.skill is not None
        ):
            self.log.add(f"{ctx.actor.name} 尝试发动主动技能【{ctx.skill.name}】")

        self.emit(EventName.ACTION_BEFORE_RESOLVE, ctx)
        if ctx.cancelled:
            if ctx.action_kind == ActionKind.SKILL_CAST and ctx.skill_kind == SkillKind.ACTIVE:
                ctx.result["consumed_action"] = True
            self.log.add("行动发动失败")
            return ctx

        self.execute_action(ctx)
        self.emit(EventName.ACTION_AFTER_RESOLVE, ctx)
        self.compat_after_action(ctx)
        self.emit(EventName.ACTION_FINISH, ctx)
        return ctx

    def execute_action(self, ctx):
        if ctx.action_kind == ActionKind.NORMAL_ATTACK:
            self.execute_normal_attack(ctx)
        elif ctx.action_kind == ActionKind.SKILL_CAST:
            self.execute_skill_cast(ctx)
        elif ctx.action_kind == ActionKind.DAMAGE:
            amount = ctx.metadata.get("amount", 1)
            damage_type = ctx.metadata.get("damage_type", DamageType.WEAPON)
            for target in ctx.targets:
                DamageCalculator.apply_damage(self, ctx.actor, target, amount, damage_type, ctx.skill)
            ctx.result["success"] = True

    def execute_normal_attack(self, ctx):
        caster = ctx.actor
        target = ctx.targets[0]
        self.log.add(f"{caster.name} 发动普通攻击，攻击 {target.name}")
        damage = DamageCalculator.weapon_damage(caster, target, 1.0)
        dealt = DamageCalculator.apply_damage(self, caster, target, damage, DamageType.WEAPON, self.normal_skill)
        ctx.result["success"] = dealt > 0
        ctx.result["damage"] = dealt

    def execute_skill_cast(self, ctx):
        skill = ctx.skill
        if skill is None:
            return
        if ctx.skill_kind == SkillKind.ASSAULT:
            target = ctx.targets[0] if ctx.targets else None
            if target is None or not target.alive:
                return
            self.log.add(f"{ctx.actor.name} 发动【{skill.name}】，攻击 {target.name}")
            if "ratio" in ctx.metadata:
                amount = DamageCalculator.weapon_damage(ctx.actor, target, ctx.metadata["ratio"])
                dealt = DamageCalculator.apply_damage(self, ctx.actor, target, amount, DamageType.WEAPON, skill)
            else:
                before = target.hp
                allies = self.allies_of(ctx.actor)
                enemies = self.enemies_of(ctx.actor)
                try:
                    skill.on_normal_attack(ctx.actor, target, allies, enemies, self.round_id, battle=self)
                except TypeError:
                    skill.on_normal_attack(ctx.actor, target, allies, enemies, self.round_id)
                dealt = max(0, before - target.hp)
            ctx.result["success"] = dealt > 0
            ctx.result["consumed_action"] = False
            return

        before_hp = {h: h.hp for h in self.team_a + self.team_b}
        try:
            consumed = bool(skill.execute(self, ctx))
        except TypeError:
            allies = self.allies_of(ctx.actor)
            enemies = self.enemies_of(ctx.actor)
            consumed = bool(skill.on_action(ctx.actor, allies, enemies, self.round_id))
        ctx.result["success"] = any(h.hp != before_hp[h] for h in before_hp) or consumed
        ctx.result["consumed_action"] = consumed

    def emit(self, event_name, ctx):
        if event_name == EventName.BEFORE_DAMAGE:
            damage_modifier_from_buffs(ctx)
        triggers = []
        for hero in self.team_a + self.team_b:
            if not hero.alive:
                continue
            for skill in hero.all_skills():
                for trigger in getattr(skill, "triggers", []):
                    if trigger.event_name == event_name:
                        triggers.append((trigger.priority, hero, skill, trigger))
        triggers.sort(key=lambda item: item[0], reverse=True)
        for _, owner, skill, trigger in triggers:
            if trigger.can_fire(self, owner, ctx):
                owner.record_skill_trigger(skill.name, self.round_id)
                trigger.fire(self, owner, ctx, skill=skill)
                if ctx.cancelled:
                    break

    def run_turn_start_passives(self, hero):
        allies = self.allies_of(hero)
        enemies = self.enemies_of(hero)
        for skill in hero.all_skills():
            if skill.skill_kind == SkillKind.PASSIVE:
                try:
                    skill.on_turn_start(hero, allies, enemies, self.round_id, battle=self)
                except TypeError:
                    skill.on_turn_start(hero, allies, enemies, self.round_id)

    def compat_after_action(self, ctx):
        if ctx.action_kind != ActionKind.NORMAL_ATTACK or not ctx.result.get("success"):
            return
        caster = ctx.actor
        target = ctx.targets[0]
        allies = self.allies_of(caster)
        enemies = self.enemies_of(caster)

        for skill in caster.all_skills():
            if skill.skill_kind == SkillKind.ASSAULT and not getattr(skill, "triggers", []) and skill.trigger_check(self):
                caster.record_skill_trigger(skill.name, self.round_id)
                self.action_queue.append(
                    ActionContext(
                        actor=caster,
                        targets=[target],
                        action_kind=ActionKind.SKILL_CAST,
                        skill=skill,
                        skill_kind=SkillKind.ASSAULT,
                        tags={"skill", "assault", "weapon_damage"},
                    )
                )
            elif skill.skill_kind == SkillKind.PASSIVE:
                try:
                    skill.on_normal_attack(caster, target, allies, enemies, self.round_id, battle=self)
                except TypeError:
                    skill.on_normal_attack(caster, target, allies, enemies, self.round_id)

        if target.alive:
            for skill in target.all_skills():
                try:
                    skill.on_be_hit(target, caster, ctx.result.get("damage", 0), self.round_id, battle=self)
                except TypeError:
                    skill.on_be_hit(target, caster, ctx.result.get("damage", 0), self.round_id)

    def resolve_round_end_buffs(self):
        for hero in self.team_a + self.team_b:
            if not hero.alive:
                continue
            for buff in list(hero.buffs):
                if buff.buff_type == BuffType.BURN:
                    source = buff.source or hero
                    amount = max(1, int(source.intelligence * max(buff.value, 0.3)))
                    self.log.add(f"{hero.name} 受到灼烧伤害")
                    DamageCalculator.apply_damage(self, source, hero, amount, DamageType.STRATEGY)
                elif buff.buff_type == BuffType.POISON:
                    source = buff.source or hero
                    amount = max(1, int(hero.max_hp * max(buff.value, 0.03)))
                    self.log.add(f"{hero.name} 受到中毒伤害")
                    DamageCalculator.apply_damage(self, source, hero, amount, DamageType.TRUE)
            before = list(hero.buffs)
            hero.tick_status()
            for buff in before:
                if buff not in hero.buffs:
                    self.emit(EventName.BUFF_REMOVED, ActionContext(hero, [], ActionKind.REMOVE_BUFF, metadata={"buff": buff}))

    def record_damage(self, caster, skill, amount):
        if skill is None:
            caster.total_damage += amount
            return
        skill.record_damage(caster, amount)

    def record_heal(self, caster, skill, amount):
        if skill is None:
            caster.total_heal += amount
            return
        skill.record_heal(caster, amount)

    def record_kill(self, caster, skill):
        if skill is None:
            caster.kill_count += 1
            return
        skill.record_kill(caster)
