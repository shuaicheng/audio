from __future__ import annotations

import asyncio
import logging

import httpx

from app.asr.base import AsrEvent
from app.config import Settings
from app.redact import scrub
from app.wavutil import pcm16_to_wav

logger = logging.getLogger(__name__)


class DegradedFileAsr:
    """显式降级：把 PCM 打成 WAV 调文件转写接口。

    这不是 vLLM Realtime，也不是 Qwen 的 streaming_transcribe。
    ASR_DEGRADED_CHUNK_SEC 大于 0 时，会按间隔重传越来越长的整段，
    事件 strategy=cumulative_file_reupload，并且 degraded=true。
    """

    mode = "degraded_file"
    degraded = True

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._pcm = bytearray()
        self._events: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self._closed = False
        self._next_seq = 1
        self._uploaded_until = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.asr_timeout_sec)
        await self._events.put(
            AsrEvent(
                kind="ready",
                degraded=True,
                strategy="file_endpoint",
                validated=False,
                message="文件转写降级已启用，这不是真流式识别",
            )
        )

    async def push_audio(self, pcm: bytes, seq: int) -> None:
        if self._closed:
            return
        if seq != self._next_seq:
            await self._events.put(AsrEvent(kind="failed", code="audio_sequence", message=f"音频序号不连续，期望 {self._next_seq}", degraded=True))
            return
        self._next_seq = seq + 1
        self._pcm.extend(pcm)
        interval = self.settings.asr_degraded_chunk_sec
        if interval > 0:
            seconds = len(self._pcm) / 2 / self.settings.asr_sample_rate
            if seconds - self._uploaded_until >= interval:
                await self._transcribe(final=False)

    async def finish(self) -> None:
        await self._transcribe(final=True)

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
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

    async def _transcribe(self, final: bool) -> None:
        async with self._lock:
            if self._client is None or self._closed:
                return
            wav = pcm16_to_wav(bytes(self._pcm), self.settings.asr_sample_rate)
            strategy = "cumulative_file_reupload" if self.settings.asr_degraded_chunk_sec > 0 and not final else "file_endpoint"
            if self.settings.asr_degraded_chunk_sec > 0:
                strategy = "cumulative_file_reupload"
            headers = {}
            if self.settings.asr_authorization:
                value = self.settings.asr_authorization
                headers["Authorization"] = value if value.lower().startswith("bearer ") else f"Bearer {value}"
            if self.settings.asr_cookie:
                headers["Cookie"] = self.settings.asr_cookie
            try:
                response = await self._client.post(
                    self.settings.asr_file_url,
                    data={"model": self.settings.asr_model},
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.info("降级文件转写失败：%s", type(exc).__name__)
                await self._events.put(
                    AsrEvent(
                        kind="failed",
                        code="file_transcription_failed",
                        message=f"文件转写失败：{scrub(str(exc), [self.settings.asr_authorization, self.settings.asr_cookie])}",
                        degraded=True,
                        strategy=strategy,
                    )
                )
                return
            text = ""
            if isinstance(payload, dict):
                text = str(payload.get("text") or "")
            self._uploaded_until = len(self._pcm) / 2 / self.settings.asr_sample_rate
            kind = "final" if final else "partial"
            await self._events.put(
                AsrEvent(
                    kind=kind,
                    text=text,
                    degraded=True,
                    strategy=strategy,
                    timestamps=False,
                    message="文件转写降级，不是真流式识别",
                )
            )
