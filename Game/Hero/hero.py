from Game.Hero.buff import Buff, BuffType
from Game.Skill.skill import Skill


_DEBUFF_NAME_MAP = {
    "stunned": BuffType.STUN,
    "silenced": BuffType.SILENCE,
    "disarmed": None,
    "confused": None,
    "burn": BuffType.BURN,
    "poison": BuffType.POISON,
}

_BUFF_NAME_MAP = {
    "combo": BuffType.COMBO,
    "immune_control": BuffType.IMMUNE_CONTROL,
}


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
        self.side = None
        self.level = level
        self.force = base_force + growth_force * level
        self.attack = self.force
        self.defense = base_defense + growth_defense * level
        self.intelligence = base_intelligence + growth_intelligence * level
        self.speed = base_speed + growth_speed * level
        self.charm = base_charm + growth_charm * level
        self.troop_type = "步兵"

        self.hp = int(solders_num)
        self.max_hp = int(solders_num)
        self.alive = True

        self.self_skill = self_skill
        self.skill1 = skill1
        self.skill2 = skill2
        self.buffs = []
        self.debuffs = {}  # 兼容旧测试/旧技能读取。

        self.kill_count = 0
        self.total_damage = 0
        self.total_heal = 0
        self.skill_stats = {}

        self.bind_skills()

    def bind_skills(self):
        for skill in self.all_skills():
            skill.bind_owner(self)

    def all_skills(self):
        return [skill for skill in [self.self_skill, self.skill1, self.skill2] if skill]

    def add_buff_obj(self, buff: Buff):
        existing = self.get_buff(buff.buff_type)
        if existing:
            existing.duration = max(existing.duration, buff.duration)
            existing.value = max(existing.value, buff.value)
            existing.tags |= buff.tags
            return existing
        self.buffs.append(buff)
        return buff

    def add_buff(self, name, duration, value=0.0, source=None):
        buff_type = _BUFF_NAME_MAP.get(name)
        if buff_type is None:
            self.debuffs[name] = max(self.debuffs.get(name, 0), duration)
            return
        self.add_buff_obj(Buff(name=name, buff_type=buff_type, duration=duration, source=source, value=value))

    def add_debuff(self, name, duration, value=0.0, source=None):
        buff_type = _DEBUFF_NAME_MAP.get(name)
        if name in ["stunned", "silenced", "disarmed", "confused"] and self.has_buff("immune_control"):
            return
        self.debuffs[name] = max(self.debuffs.get(name, 0), duration)
        if buff_type is not None:
            self.add_buff_obj(Buff(name=name, buff_type=buff_type, duration=duration, source=source, value=value))

    def has_buff(self, name):
        buff_type = _BUFF_NAME_MAP.get(name)
        if buff_type is None:
            return False
        return self.has_buff_type(buff_type)

    def has_debuff(self, name):
        if self.debuffs.get(name, 0) > 0:
            return True
        buff_type = _DEBUFF_NAME_MAP.get(name)
        return buff_type is not None and self.has_buff_type(buff_type)

    def has_buff_type(self, buff_type: BuffType):
        return any(buff.buff_type == buff_type and buff.duration > 0 for buff in self.buffs)

    def get_buff(self, buff_type: BuffType):
        for buff in self.buffs:
            if buff.buff_type == buff_type and buff.duration > 0:
                return buff
        return None

    def buff_value(self, buff_type: BuffType):
        return sum(buff.value for buff in self.buffs if buff.buff_type == buff_type and buff.duration > 0)

    def remove_buff(self, name):
        buff_type = _BUFF_NAME_MAP.get(name)
        if buff_type is not None:
            self.buffs = [buff for buff in self.buffs if buff.buff_type != buff_type]

    def remove_debuff(self, name):
        self.debuffs.pop(name, None)
        buff_type = _DEBUFF_NAME_MAP.get(name)
        if buff_type is not None:
            self.buffs = [buff for buff in self.buffs if buff.buff_type != buff_type]

    def tick_status(self):
        for buff in list(self.buffs):
            buff.duration -= 1
            if buff.duration <= 0:
                self.buffs.remove(buff)
        for key in list(self.debuffs.keys()):
            self.debuffs[key] -= 1
            if self.debuffs[key] <= 0:
                del self.debuffs[key]

    def can_act(self):
        return self.alive and not self.has_debuff("stunned")

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
        self._ensure_skill_stat(skill_name)["total_damage"] += damage

    def record_skill_heal(self, skill_name, amount):
        self._ensure_skill_stat(skill_name)["total_heal"] += amount

    def record_skill_kill(self, skill_name):
        self._ensure_skill_stat(skill_name)["kill_count"] += 1
