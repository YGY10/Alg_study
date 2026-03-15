# Game/Hero/hero.py
from Game.Skill.skill import Skill


class Hero:
    def __init__(
        self,
        name: str,
        base_force: int,
        base_defense: int,
        base_intelligence: int,
        base_speed: int,
        base_charm: int,
        growth_force: float,
        growth_defense: float,
        growth_intelligence: float,
        growth_speed: float,
        growth_charm: float,
        level: int,
        solders_num: int,
        self_skill: Skill,
        skill1: Skill = None,
        skill2: Skill = None,
    ):
        self.name = name
        self.level = level

        # 属性
        self.force = base_force + growth_force * level
        self.defense = base_defense + growth_defense * level
        self.intelligence = base_intelligence + growth_intelligence * level
        self.speed = base_speed + growth_speed * level
        self.charm = base_charm + growth_charm * level

        # 兵力
        self.hp = solders_num
        self.max_hp = solders_num
        self.alive = True

        # 技能
        self.self_skill = self_skill
        self.skill1 = skill1
        self.skill2 = skill2

        # =========================
        # 🔥 状态系统
        # =========================

        # 增益状态容器
        # 例如：{"attack_up": 2, "immune_control": 1}
        self.buffs = {}

        # 负面状态容器
        # 例如：{"stunned": 1, "silenced": 2}
        self.debuffs = {}

        # =========================
        # 战斗统计
        # =========================
        self.kill_count = 0
        self.total_damage = 0
        self.total_heal = 0
        self.skill_stats = {}

    # =========================================================
    # 技能集合
    # =========================================================
    def all_skills(self):
        skills = []
        if self.self_skill:
            skills.append(self.self_skill)
        if self.skill1:
            skills.append(self.skill1)
        if self.skill2:
            skills.append(self.skill2)
        return skills

    # =========================================================
    # 状态管理接口
    # =========================================================
    def add_buff(self, name, duration):
        self.buffs[name] = max(self.buffs.get(name, 0), duration)

    def add_debuff(self, name, duration):
        # 如果有免疫控制类buff
        if (
            name in ["stunned", "silenced", "disarmed", "confused"]
            and self.buffs.get("immune_control", 0) > 0
        ):
            return
        self.debuffs[name] = max(self.debuffs.get(name, 0), duration)

    def has_buff(self, name):
        return self.buffs.get(name, 0) > 0

    def has_debuff(self, name):
        return self.debuffs.get(name, 0) > 0

    def remove_buff(self, name):
        if name in self.buffs:
            del self.buffs[name]

    def remove_debuff(self, name):
        if name in self.debuffs:
            del self.debuffs[name]

    def tick_status(self):
        # Buff 递减
        for k in list(self.buffs.keys()):
            self.buffs[k] -= 1
            if self.buffs[k] <= 0:
                del self.buffs[k]

        # Debuff 递减
        for k in list(self.debuffs.keys()):
            self.debuffs[k] -= 1
            if self.debuffs[k] <= 0:
                del self.debuffs[k]

    # =========================================================
    # 行动判定
    # =========================================================
    def can_act(self):
        return self.alive and not self.has_debuff("stunned")

    # =========================================================
    # 🔥 统一统计底层
    # =========================================================
    def _ensure_skill_stat(self, skill_name):
        if skill_name not in self.skill_stats:
            self.skill_stats[skill_name] = {}

        stat = self.skill_stats[skill_name]
        stat.setdefault("trigger_rounds", [])
        stat.setdefault("kill_count", 0)
        stat.setdefault("total_damage", 0)
        stat.setdefault("total_heal", 0)

        return stat

    def record_skill_trigger(self, skill_name, round_id):
        stat = self._ensure_skill_stat(skill_name)
        if round_id is not None:
            stat["trigger_rounds"].append(round_id)

    def record_skill_damage(self, skill_name, damage):
        stat = self._ensure_skill_stat(skill_name)
        stat["total_damage"] += damage

    def record_skill_heal(self, skill_name, amount):
        stat = self._ensure_skill_stat(skill_name)
        stat["total_heal"] += amount

    def record_skill_kill(self, skill_name):
        stat = self._ensure_skill_stat(skill_name)
        stat["kill_count"] += 1
