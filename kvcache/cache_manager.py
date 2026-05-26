"""
CacheManager: polls SGLang instance load and cache stats at runtime.

Provides per-instance state to TaskAwareRouter for load-balance-aware dispatch.
Runs as a background async task inside the benchmark loop.

State per instance:
  queue_length   : number of pending requests (/get_load)
  cache_hit_rate : radix cache hit rate (/get_server_info, if available)
"""

import asyncio
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class InstanceState:
    name: str
    url: str
    queue_length: int = 0
    cache_hit_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def is_stale(self, ttl: float = 2.0) -> bool:
        return time.time() - self.last_updated > ttl


class CacheManager:
    """
    Lightweight async cache/load state monitor for SGLang instances.

    Usage:
        manager = CacheManager(router.instances)
        async with aiohttp.ClientSession() as session:
            await manager.start(session)          # start background polling
            state = manager.get_state()           # use in routing decisions
            await manager.stop()
    """

    def __init__(self, instances, poll_interval: float = 1.0):
        self._instances = instances
        self._poll_interval = poll_interval
        self._state: dict[str, InstanceState] = {
            inst.name: InstanceState(name=inst.name, url=inst.url)
            for inst in instances
        }
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self, session: aiohttp.ClientSession):
        self._session = session
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_state(self) -> dict[str, InstanceState]:
        return dict(self._state)

    async def _poll_loop(self):
        while True:
            await self._poll_all()
            await asyncio.sleep(self._poll_interval)

    async def _poll_all(self):
        tasks = [self._poll_instance(inst) for inst in self._instances]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_instance(self, inst):
        state = self._state[inst.name]
        try:
            # Queue length
            async with self._session.get(
                f"{inst.url}/get_load",
                timeout=aiohttp.ClientTimeout(total=1.0),
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    state.queue_length = int(data.get("load", 0))

            # Cache hit rate (best-effort, may not be available)
            async with self._session.get(
                f"{inst.url}/get_server_info",
                timeout=aiohttp.ClientTimeout(total=1.0),
            ) as r:
                if r.status == 200:
                    info = await r.json()
                    rate = (
                        info.get("cache_hit_rate")
                        or info.get("radix_cache_hit_rate")
                        or 0.0
                    )
                    state.cache_hit_rate = float(rate)

            state.last_updated = time.time()
        except Exception:
            pass  # Stale state is fine; router falls back to affinity
