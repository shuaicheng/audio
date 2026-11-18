from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import AsyncIterator

from app.config import Settings


class MockLlm:
    """本地逐字输出。文案本身会标明这不是真实模型。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream(self, messages: list[dict], cancelled: asyncio.Event) -> AsyncIterator[str]:
        text = "你好。这是离线演示回答，不是真实模型生成的内容。"
        delay = self.settings.mock_token_delay_sec
        for char in text:
            if cancelled.is_set():
                return
            if delay:
                await asyncio.sleep(delay)
            yield char


class MockTts:
    """生成短促正弦波 PCM，只为演示播放链路。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream(self, text: str, cancelled: asyncio.Event) -> AsyncIterator[tuple[bytes, int]]:
        rate = self.settings.tts_sample_rate
        pcm = _sine(rate, seconds=0.18, freq=440 if len(text) % 2 == 0 else 554)
        step = max(2, rate // 25 * 2)
        for offset in range(0, len(pcm), step):
            if cancelled.is_set():
                return
            yield pcm[offset : offset + step], rate
            await asyncio.sleep(0)


def _sine(sample_rate: int, seconds: float, freq: float) -> bytes:
    count = max(1, int(sample_rate * seconds))
    out = bytearray()
    for index in range(count):
        envelope = min(1.0, index / 80, (count - index) / 80)
        sample = int(max(-1.0, min(1.0, math.sin(2 * math.pi * freq * index / sample_rate))) * 32767 * envelope)
        out += struct.pack("<h", sample)
    return bytes(out)
