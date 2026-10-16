"""
core/event_bus.py
Minimal in-process pub/sub for decoupled modules.
"""
import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

Handler = Callable[[str, Any], Awaitable[None]]


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Handler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Handler):
        self._subs[event_type].append(handler)

    async def publish(self, event_type: str, payload: Any):
        handlers = self._subs.get(event_type, []) + self._subs.get("*", [])
        if not handlers:
            return
        await asyncio.gather(
            *[self._safe_call(h, event_type, payload) for h in handlers],
            return_exceptions=True,
        )

    @staticmethod
    async def _safe_call(handler: Handler, event_type: str, payload: Any):
        try:
            await handler(event_type, payload)
        except Exception as e:
            logger.error(f"EventBus handler error on '{event_type}': {e}")


# Event type constants
EVENT_SIGNAL_CREATED = "signal.created"
EVENT_SIGNAL_HIGH_CONF = "signal.high_confidence"
EVENT_TRADE_OPENED = "trade.opened"
EVENT_TRADE_CLOSED = "trade.closed"
EVENT_POSITION_UPDATED = "position.updated"
EVENT_EMERGENCY_STOP = "emergency.stop"
EVENT_TUNING_APPLIED = "tuning.applied"