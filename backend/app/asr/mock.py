from __future__ import annotations

import asyncio

from app.asr.base import AsrEvent
from app.config import Settings


class MockAsrEngine:
    """离线演示。不会把结果说成真实识别。"""

    mode = "mock"
    degraded = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._events: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self._pcm = 0
        self._next_seq = 1
        self._closed = False

    async def start(self) -> None:
        await self._events.put(AsrEvent(kind="ready", validated=False, message="离线演示识别，不是真实模型"))

    async def push_audio(self, pcm: bytes, seq: int) -> None:
        if seq != self._next_seq:
            await self._events.put(AsrEvent(kind="failed", code="audio_sequence", message=f"音频序号不连续，期望 {self._next_seq}"))
            return
        self._next_seq = seq + 1
        self._pcm += len(pcm)

    async def finish(self) -> None:
        if self._pcm <= 0:
            text = "没有收到音频。这是离线演示占位，不是真实识别。"
        else:
            await self._events.put(AsrEvent(kind="partial", text="这是离线", timestamps=False))
            await asyncio.sleep(0)
            text = "这是离线演示的识别文本，不是真实语音识别。"
            await self._events.put(AsrEvent(kind="partial", text=text, timestamps=False))
        await self._events.put(AsrEvent(kind="final", text=text if self._pcm > 0 else "没有收到音频。这是离线演示占位，不是真实识别。", timestamps=False))

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._events.put(None)

    def drain(self) -> list[AsrEvent]:
        items: list[AsrEvent] = []
        while True:
            try:
                item = self._events.get_nowait()
            except asyncio.QueueEmpty:
                return items
            if item is not None:
                items.append(item)

    async def next_event(self, timeout: float | None = None) -> AsrEvent | None:
        if timeout is None:
            return await self._events.get()
        try:
            return await asyncio.wait_for(self._events.get(), timeout)
        except TimeoutError:
            return None
