from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BuffType(Enum):
    STUN = "眩晕"
    SILENCE = "沉默"
    BURN = "灼烧"
    POISON = "中毒"
    DMG_UP = "增伤"
    DMG_DOWN = "减伤"
    DEF_UP = "防御提升"
    DEF_DOWN = "防御下降"
    COMBO = "连击"
    IMMUNE_CONTROL = "免疫控制"


@dataclass
class Buff:
    name: str
    buff_type: BuffType
    duration: int
    source: Optional["Hero"] = None
    value: float = 0.0
    tags: set[str] = field(default_factory=set)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Game.Hero.hero import Hero
