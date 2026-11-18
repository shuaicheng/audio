from __future__ import annotations

import asyncio


class PlaybackGate:
    """限制已经交给浏览器、尚未播放完的音频字节数，并可被前端暂停。"""

    def __init__(self, max_inflight: int) -> None:
        self.max_inflight = max(1, max_inflight)
        self.inflight = 0
        self.hold = False
        self._cv = asyncio.Condition()

    async def reserve(self, nbytes: int, cancelled: asyncio.Event | None = None) -> bool:
        need = max(0, nbytes)
        async with self._cv:
            while self.hold or self.inflight + need > self.max_inflight:
                if cancelled is not None and cancelled.is_set():
                    return False
                await self._cv.wait()
            if cancelled is not None and cancelled.is_set():
                return False
            self.inflight += need
            return True

    async def release(self, nbytes: int) -> None:
        async with self._cv:
            self.inflight = max(0, self.inflight - max(0, nbytes))
            self._cv.notify_all()

    async def set_hold(self, hold: bool) -> None:
        async with self._cv:
            self.hold = hold
            self._cv.notify_all()

    async def reset(self) -> None:
        async with self._cv:
            self.inflight = 0
            self.hold = False
            self._cv.notify_all()
