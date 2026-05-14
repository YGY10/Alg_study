from Game.Battle.action import ActionContext, ActionKind, SkillKind
from Game.Battle.events import EventName
from Game.Battle.trigger import Trigger
from Game.Skill.skill import Skill, SkillType


class QueueFollowAssault:
    def __init__(self, skill, ratio=2.2):
        self.skill = skill
        self.ratio = ratio

    def apply(self, battle, owner, ctx):
        target = ctx.targets[0] if ctx.targets else None
        if target is None or not target.alive:
            return
        battle.log.add(
            f"{owner.name} 普通攻击后触发突击技能【{self.skill.name}】，追击 {target.name}"
        )
        battle.action_queue.append(
            ActionContext(
                actor=owner,
                targets=[target],
                action_kind=ActionKind.SKILL_CAST,
                skill=self.skill,
                skill_kind=SkillKind.ASSAULT,
                tags={"skill", "assault", "weapon_damage"},
                metadata={"ratio": self.ratio},
            )
        )


class ZhuiJi(Skill):
    def __init__(self):
        super().__init__(name="追击", skill_type=SkillType.ASSAULT, probability=1.0)
        self.triggers = [
            Trigger(
                event_name=EventName.ACTION_AFTER_RESOLVE,
                conditions=[
                    lambda battle, owner, ctx: ctx.action_kind == ActionKind.NORMAL_ATTACK,
                    lambda battle, owner, ctx: ctx.actor == owner,
                    lambda battle, owner, ctx: ctx.result.get("success", False),
                ],
                chance=0.45,
                effects=[QueueFollowAssault(self)],
                priority=10,
            )
        ]
