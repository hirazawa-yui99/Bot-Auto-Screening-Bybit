"""
core/health_monitor.py
Heartbeat + auto-restart watchdog.

Tracks last successful tick of each subsystem. If a subsystem stalls
beyond its threshold, we log CRITICAL and (optionally) restart it.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubsystemState:
    name: str
    last_heartbeat: float = 0.0
    stall_threshold_sec: float = 300.0
    restart_callback: Optional[Callable] = None
    restarts: int = 0
    max_restarts: int = 10


class HealthMonitor:
    def __init__(self):
        self._subsystems: Dict[str, SubsystemState] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def register(
        self,
        name: str,
        stall_threshold_sec: float = 300.0,
        restart_callback: Optional[Callable] = None,
    ):
        self._subsystems[name] = SubsystemState(
            name=name,
            stall_threshold_sec=stall_threshold_sec,
            restart_callback=restart_callback,
            last_heartbeat=time.time(),
        )
        logger.info(f"HealthMonitor: registered '{name}'")

    def heartbeat(self, name: str):
        if name in self._subsystems:
            self._subsystems[name].last_heartbeat = time.time()

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._watchdog_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watchdog_loop(self):
        while self._running:
            now = time.time()
            for name, state in self._subsystems.items():
                elapsed = now - state.last_heartbeat
                if elapsed > state.stall_threshold_sec:
                    logger.critical(
                        f"HEALTH: '{name}' stalled for {elapsed:.0f}s "
                        f"(threshold {state.stall_threshold_sec:.0f}s)"
                    )
                    if state.restart_callback and state.restarts < state.max_restarts:
                        state.restarts += 1
                        try:
                            result = state.restart_callback()
                            if asyncio.iscoroutine(result):
                                await result
                            state.last_heartbeat = time.time()
                            logger.info(
                                f"HEALTH: restarted '{name}' "
                                f"({state.restarts}/{state.max_restarts})"
                            )
                        except Exception as e:
                            logger.error(
                                f"HEALTH: restart failed for '{name}': {e}"
                            )
            await asyncio.sleep(10)