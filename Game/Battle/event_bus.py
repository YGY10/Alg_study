# Game/Battle/event_bus.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional


@dataclass
class _Listener:
    handler: Callable[..., None]
    predicate: Optional[Callable[..., bool]] = None


class BattleEventBus:
    """
    一个简单的事件总线：
    - register(event_name, handler, predicate=None)
    - dispatch(event_name, **kwargs)

    predicate 用来过滤事件归属：例如只让“actor==关羽”时才触发关羽的技能监听器
    """

    def __init__(self):
        self.listeners: Dict[str, List[_Listener]] = {}

    def register(
        self,
        event_name: str,
        handler: Callable[..., None],
        predicate: Optional[Callable[..., bool]] = None,
    ) -> None:
        if event_name not in self.listeners:
            self.listeners[event_name] = []
        self.listeners[event_name].append(
            _Listener(handler=handler, predicate=predicate)
        )

    def dispatch(self, event_name: str, **kwargs) -> None:
        lst = self.listeners.get(event_name, [])
        if not lst:
            return

        for li in lst:
            if li.predicate is None or li.predicate(**kwargs):
                li.handler(**kwargs)
