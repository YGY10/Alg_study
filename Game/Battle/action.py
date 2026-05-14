from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionKind(Enum):
    NORMAL_ATTACK = "normal_attack"
    SKILL_CAST = "skill_cast"
    DAMAGE = "damage"
    HEAL = "heal"
    APPLY_BUFF = "apply_buff"
    REMOVE_BUFF = "remove_buff"
    COUNTER_ATTACK = "counter_attack"
    FOLLOW_UP_ATTACK = "follow_up_attack"


class SkillKind(Enum):
    ACTIVE = "主动"
    PASSIVE = "被动"
    ASSAULT = "突击"
    COMMAND = "指挥"
    FORMATION = "阵法"


@dataclass
class ActionContext:
    actor: "Hero"
    targets: list["Hero"]
    action_kind: ActionKind
    skill: Optional["Skill"] = None
    skill_kind: Optional[SkillKind] = None
    tags: set[str] = field(default_factory=set)
    cancelled: bool = False
    result: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Game.Hero.hero import Hero
    from Game.Skill.skill import Skill
