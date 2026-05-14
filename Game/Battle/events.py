from enum import Enum


class EventName(Enum):
    BATTLE_START = "BattleStart"
    ROUND_START = "RoundStart"
    TURN_START = "TurnStart"
    ACTION_INTENT = "ActionIntent"
    ACTION_CHECK = "ActionCheck"
    ACTION_BEFORE_RESOLVE = "ActionBeforeResolve"
    ACTION_AFTER_RESOLVE = "ActionAfterResolve"
    ACTION_FINISH = "ActionFinish"
    BEFORE_DAMAGE = "BeforeDamage"
    AFTER_DAMAGE = "AfterDamage"
    BEFORE_HEAL = "BeforeHeal"
    AFTER_HEAL = "AfterHeal"
    BUFF_ADDED = "BuffAdded"
    BUFF_REMOVED = "BuffRemoved"
    TURN_END = "TurnEnd"
    ROUND_END = "RoundEnd"
    BATTLE_END = "BattleEnd"
