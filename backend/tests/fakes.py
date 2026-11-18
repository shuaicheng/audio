from __future__ import annotations

import asyncio

from app.asr.base import AsrEvent
from app.tts import TtsError


class ScriptedAsr:
    mode = "script"
    degraded = False

    def __init__(self, events: list[AsrEvent] | None = None, block_audio: bool = False) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.audio: list[tuple[int, bytes]] = []
        self.closed = False
        self.finished = False
        self.block = asyncio.Event()
        self.block_audio = block_audio
        for event in events or []:
            self.queue.put_nowait(event)

    async def start(self) -> None:
        await self.queue.put(AsrEvent(kind="ready", validated=True))

    async def push_audio(self, pcm: bytes, seq: int) -> None:
        if self.block_audio:
            await self.block.wait()
        self.audio.append((seq, pcm))

    async def finish(self) -> None:
        self.finished = True

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.queue.put(None)

    async def next_event(self, timeout: float | None = None):
        if timeout is None:
            return await self.queue.get()
        return await asyncio.wait_for(self.queue.get(), timeout)


class PausingLlm:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict]] = []
        self.released = asyncio.Event()
        self.saw_forbidden = False
        self.closed = False

    async def stream(self, messages, cancelled: asyncio.Event):
        self.calls += 1
        self.messages.append(messages)
        try:
            if self.calls == 1:
                yield "部分回答，"
                await self.released.wait()
                if cancelled.is_set():
                    return
                self.saw_forbidden = True
                yield "不该出现"
            else:
                yield "新的回答。"
        finally:
            self.closed = True


class ScriptLlm:
    def __init__(self, parts: list[str], delay: float = 0, error: Exception | None = None) -> None:
        self.parts = parts
        self.delay = delay
        self.error = error
        self.calls = 0
        self.messages: list[list[dict]] = []

    async def stream(self, messages, cancelled: asyncio.Event):
        self.calls += 1
        self.messages.append(messages)
        for part in self.parts:
            if cancelled.is_set():
                return
            if self.delay:
                await asyncio.sleep(self.delay)
            yield part
        if self.error:
            raise self.error


class RecordingTts:
    def __init__(self, fail: bool = False, sample_rate: int = 24000) -> None:
        self.fail = fail
        self.sample_rate = sample_rate
        self.sentences: list[str] = []
        self.started: list[float] = []

    async def stream(self, text: str, cancelled: asyncio.Event):
        self.sentences.append(text)
        self.started.append(asyncio.get_running_loop().time())
        if self.fail:
            raise TtsError("tts_http", "语音合成失败（HTTP 500）")
        if cancelled.is_set():
            return
        yield b"\x01\x00" * 8, self.sample_rate
        yield b"\x02", self.sample_rate
        yield b"\x03\x04\x00", self.sample_rate
