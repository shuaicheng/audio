from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import websockets

from app.asr.base import AsrEvent
from app.asr.normalize import merge_hypothesis, visible_transcript
from app.config import Settings

logger = logging.getLogger(__name__)


class RealtimeAsrEngine:
    """vLLM Realtime 转写客户端。

    顺序与 vLLM 文档一致：session.created、session.update、初始
    input_audio_buffer.commit、input_audio_buffer.append、最终 commit(final=true)，
    然后读取 transcription.delta / transcription.done。

    握手被 model_not_found 或 model_not_validated 拒绝时，结果是 asr.unsupported。
    没有拒绝只说明这次连接接受了模型名，不能当成部署服务已经支持 Qwen3-ASR。
    """

    mode = "realtime"
    degraded = False

    def __init__(self, settings: Settings, url: str | None = None) -> None:
        self.settings = settings
        self.url = url or settings.asr_realtime_url
        self._ws: Any = None
        self._reader: asyncio.Task | None = None
        self._dispatcher: asyncio.Task | None = None
        self._events: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self._server_events: asyncio.Queue[dict | None] = asyncio.Queue()
        self._closed = False
        self._started = False
        self._finish_sent = False
        self._hypothesis = ""
        self._next_seq = 1
        self._got_terminal = False
        self.accepted = False

    async def start(self) -> None:
        connect_kwargs: dict[str, Any] = {"open_timeout": self.settings.asr_handshake_timeout_sec}
        headers = self._headers()
        if headers:
            connect_kwargs["additional_headers"] = headers
        try:
            self._ws = await websockets.connect(self.url, **connect_kwargs)
        except Exception as exc:
            await self._emit(
                AsrEvent(
                    kind="unsupported",
                    code="connect_failed",
                    message=f"无法连接实时识别服务：{type(exc).__name__}",
                )
            )
            return
        self._reader = asyncio.create_task(self._read_loop())
        try:
            first = await asyncio.wait_for(self._server_events.get(), self.settings.asr_handshake_timeout_sec)
        except TimeoutError:
            await self._fail_unsupported("handshake_timeout", "实时识别没有返回 session.created")
            return
        if not first or first.get("type") != "session.created":
            await self._fail_unsupported("bad_protocol", "实时识别协议不匹配：缺少 session.created")
            return
        await self._send({"type": "session.update", "model": self.settings.asr_model})
        if await self._consume_probe():
            return
        await self._send({"type": "input_audio_buffer.commit", "final": False})
        if await self._consume_probe():
            return
        self._started = True
        self.accepted = True
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        await self._emit(AsrEvent(kind="ready", validated=True, message="本次连接已通过模型握手"))

    async def push_audio(self, pcm: bytes, seq: int) -> None:
        if not self._started or self._closed:
            return
        if seq != self._next_seq:
            await self._emit(
                AsrEvent(kind="failed", code="audio_sequence", message=f"音频序号不连续，期望 {self._next_seq}")
            )
            return
        self._next_seq = seq + 1
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def finish(self) -> None:
        if not self._started or self._closed or self._finish_sent:
            return
        self._finish_sent = True
        await self._send({"type": "input_audio_buffer.commit", "final": True})

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed and self._ws is None:
            return
        self._closed = True
        for task in (self._reader, self._dispatcher):
            if task and not task.done():
                task.cancel()
        for task in (self._reader, self._dispatcher):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("关闭实时识别连接时出现异常", exc_info=True)
            self._ws = None
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
            item = await self._events.get()
            return item
        try:
            return await asyncio.wait_for(self._events.get(), timeout)
        except TimeoutError:
            return None

    async def _consume_probe(self) -> bool:
        try:
            event = await asyncio.wait_for(self._server_events.get(), self.settings.asr_error_probe_sec)
        except TimeoutError:
            return False
        if event is None:
            await self._fail_unsupported("upstream_closed", "实时识别在握手期间断开")
            return True
        if event.get("type") == "error":
            code, message = _error_parts(event)
            if _is_model_rejection(code, message):
                await self._fail_unsupported(code or "model_rejected", message or "服务端不接受该识别模型")
            else:
                await self._emit(AsrEvent(kind="failed", code=code or "upstream", message=message or "识别服务返回错误"))
                await self.close()
            return True
        await self._apply(event)
        return False

    async def _fail_unsupported(self, code: str, message: str) -> None:
        await self._emit(AsrEvent(kind="unsupported", code=code, message=message, validated=False))
        await self.close()

    async def _dispatch_loop(self) -> None:
        while not self._closed:
            event = await self._server_events.get()
            if event is None:
                if self._finish_sent and not self._got_terminal and not self._closed:
                    await self._emit(AsrEvent(kind="failed", code="upstream_closed", message="识别结束前连接已断开"))
                break
            await self._apply(event)

    async def _apply(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "transcription.delta":
            if isinstance(event.get("text"), str):
                self._hypothesis = str(event["text"])
            else:
                self._hypothesis = merge_hypothesis(self._hypothesis, str(event.get("delta") or ""))
            await self._emit(AsrEvent(kind="partial", text=visible_transcript(self._hypothesis)))
            return
        if kind == "transcription.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                self._hypothesis = text
            visible = visible_transcript(self._hypothesis)
            if self._finish_sent:
                self._got_terminal = True
                await self._emit(AsrEvent(kind="final", text=visible, timestamps=False))
            else:
                await self._emit(AsrEvent(kind="partial", text=visible, timestamps=False))
            return
        if kind == "error":
            code, message = _error_parts(event)
            target = "unsupported" if _is_model_rejection(code, message) else "failed"
            await self._emit(AsrEvent(kind=target, code=code or "upstream", message=message or "识别失败"))

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    await self._emit(AsrEvent(kind="failed", code="bad_upstream_json", message="识别服务返回了无法解析的消息"))
                    continue
                if isinstance(event, dict):
                    await self._server_events.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closed:
                logger.info("实时识别连接结束")
        finally:
            await self._server_events.put(None)

    async def _send(self, payload: dict) -> None:
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def _emit(self, event: AsrEvent) -> None:
        await self._events.put(event)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.settings.asr_authorization:
            value = self.settings.asr_authorization
            headers["Authorization"] = value if value.lower().startswith("bearer ") else f"Bearer {value}"
        if self.settings.asr_cookie:
            headers["Cookie"] = self.settings.asr_cookie
        return headers


def _error_parts(event: dict) -> tuple[str, str]:
    code = str(event.get("code") or "")
    error = event.get("error")
    message = ""
    if isinstance(error, dict):
        message = str(error.get("message") or "")
        code = code or str(error.get("code") or "")
    elif isinstance(error, str):
        message = error
    message = message or str(event.get("message") or "")
    return code, message


def _is_model_rejection(code: str, message: str) -> bool:
    blob = f"{code} {message}".lower()
    tokens = ("model_not_found", "model_not_validated", "does not exist", "not validated", "unknown model")
    return any(token in blob for token in tokens)
