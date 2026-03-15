# Game/Battle/battle_engine.py
import random
from Game.Skill.skill import Skill, SkillType


class BattleEngine:

    def __init__(self, team_a, team_b, seed=None, max_rounds=8):
        self.team_a = team_a
        self.team_b = team_b
        self.max_rounds = max_rounds
        self.round_id = 1

        # 普攻统一记账
        self.normal_skill = Skill("普通攻击", SkillType.ACTIVE)

        if seed is not None:
            random.seed(seed)

    # =========================================================
    # 基础工具
    # =========================================================
    def alive_units(self, team):
        return [h for h in team if h.alive]

    def battle_over(self):
        return not self.alive_units(self.team_a) or not self.alive_units(self.team_b)

    def print_team_status(self):
        def format_team(team):
            return " | ".join(
                f"{h.name}:{h.hp}/{h.max_hp}({'存活' if h.alive else '阵亡'})"
                for h in team
            )

        print("我方 =>", format_team(self.team_a))
        print("敌方 =>", format_team(self.team_b))

    # =========================================================
    # 战斗主流程
    # =========================================================
    def start_battle(self):

        print("========== 战斗开始 ==========")
        self.print_team_status()

        while not self.battle_over() and self.round_id <= self.max_rounds:

            print(f"\n======== 第 {self.round_id} 回合 ========")
            self.execute_round()
            self.print_team_status()
            self.round_id += 1

        print("\n========== 战斗结束 ==========")

        team_a_alive = len(self.alive_units(self.team_a))
        team_b_alive = len(self.alive_units(self.team_b))

        if team_a_alive > 0 and team_b_alive == 0:
            print("🏆 我方胜利")
        elif team_b_alive > 0 and team_a_alive == 0:
            print("🏆 敌方胜利")
        else:
            print("⚖️ 平局")

        print("\n========== 战斗统计 ==========")

        for hero in self.team_a + self.team_b:

            print(f"\n{hero.name}：")
            print(f"  总输出伤害：{hero.total_damage}")
            print(f"  总治疗量：{hero.total_heal}")

            if not hero.skill_stats:
                print("  无技能触发记录")
                continue

            for skill_name, stat in hero.skill_stats.items():
                print(f"  技能【{skill_name}】: ")
                print(f"    触发回合：{stat['trigger_rounds']}")
                print(f"    技能输出伤害：{stat['total_damage']}")
                print(f"    技能治疗量：{stat['total_heal']}")
                print(f"    技能击杀数：{stat['kill_count']}")

    # =========================================================
    # 单回合执行
    # =========================================================
    def execute_round(self):

        all_units = self.alive_units(self.team_a) + self.alive_units(self.team_b)
        random.shuffle(all_units)
        all_units.sort(key=lambda h: h.speed, reverse=True)

        for hero in all_units:

            if not hero.alive:
                continue

            hero.tick_status()

            if not hero.can_act():
                print(f"{hero.name} 受到震慑，无法行动")
                continue

            allies = self.team_a if hero in self.team_a else self.team_b
            enemies = self.team_b if hero in self.team_a else self.team_a

            if not self.alive_units(enemies):
                break

            print(f"\n{hero.name} 行动：")

            skill_triggered = False

            # =========================
            # ① 主动技能阶段
            # =========================
            if not hero.has_debuff("silenced"):

                for skill in hero.all_skills():
                    if skill.skill_type == SkillType.ACTIVE:
                        if skill.on_action(hero, allies, enemies, self.round_id):
                            skill_triggered = True
                            break
            else:
                print(f"{hero.name} 处于【技穷】状态，无法发动主动技能")

            # =========================
            # ② 回合型被动阶段
            # =========================
            if not hero.has_debuff("pseudo_report"):

                for skill in hero.all_skills():
                    if skill.skill_type == SkillType.PASSIVE:
                        skill.on_turn_start(hero, allies, enemies, self.round_id)

            else:
                print(f"{hero.name} 处于【伪报】状态，被动技能失效")

            # =========================
            # ③ 普攻阶段（支持连击）
            # =========================
            if not skill_triggered:

                if hero.has_debuff("disarmed"):
                    print(f"{hero.name} 处于【缴械】状态，无法普通攻击")
                else:

                    attack_times = 1

                    # 🔥 强攻提供双普攻
                    if hero.has_buff("combo"):
                        attack_times = 2
                        print(f"{hero.name} 触发【连击】，本回合普攻两次")

                    for _ in range(attack_times):

                        if not self.alive_units(enemies):
                            break

                        self.normal_attack(hero, allies, enemies)

                    # 连击只持续本回合
                    if hero.has_buff("combo"):
                        hero.buffs.pop("combo", None)

    # =========================================================
    # 普通攻击
    # =========================================================
    def normal_attack(self, caster, allies, enemies):

        # ① 选择目标
        if caster.has_debuff("confused"):

            possible_targets = [h for h in allies + enemies if h.alive and h != caster]
            if not possible_targets:
                return
            target = random.choice(possible_targets)
            print(f"{caster.name} 混乱攻击 {target.name}")

        else:
            targets = [e for e in enemies if e.alive]
            if not targets:
                return
            target = random.choice(targets)

        # ② 主伤害
        damage = caster.force - target.defense * 0.5
        damage = max(0, int(damage))

        target.hp -= damage

        print(f"{caster.name} 普攻 {target.name} 造成 {damage} 伤害")

        self.normal_skill.record_damage(caster, damage)

        # =====================================================
        # ③ 普攻后触发 ASSAULT（突击）
        # =====================================================
        for skill in caster.all_skills():
            if skill.skill_type == SkillType.ASSAULT:
                if skill.trigger_check():
                    skill.record_trigger(caster, self.round_id)
                    skill.on_normal_attack(
                        caster, target, allies, enemies, self.round_id
                    )

        # =====================================================
        # ④ 普攻类被动
        # =====================================================
        for skill in caster.all_skills():
            if skill.skill_type == SkillType.PASSIVE:
                skill.on_normal_attack(caster, target, allies, enemies, self.round_id)

        # =====================================================
        # ⑤ 受击触发
        # =====================================================
        for skill in target.all_skills():
            skill.on_be_hit(target, caster, damage, self.round_id)

        # ⑥ 击杀判定
        if target.hp <= 0:
            target.hp = 0
            target.alive = False
            print(f"{target.name} 被击杀")
            self.normal_skill.record_kill(caster)
