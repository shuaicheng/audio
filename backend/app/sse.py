from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

DONE = object()


class SseParser:
    """按 SSE 事件切分。允许 data 行被任意字节边界拆开。"""

    def __init__(self) -> None:
        self._buf = ""
        self._data: list[str] = []

    def feed(self, text: str) -> list[str]:
        self._buf += text
        events: list[str] = []
        while True:
            newline = self._buf.find("\n")
            if newline < 0:
                break
            line = self._buf[:newline]
            self._buf = self._buf[newline + 1 :]
            if line.endswith("\r"):
                line = line[:-1]
            self._take_line(line, events)
        return events

    def flush(self) -> list[str]:
        events: list[str] = []
        if self._buf:
            line = self._buf[:-1] if self._buf.endswith("\r") else self._buf
            self._buf = ""
            if line:
                self._take_line(line, events)
        if self._data:
            events.append("\n".join(self._data))
            self._data = []
        return events

    def _take_line(self, line: str, events: list[str]) -> None:
        if line == "":
            if self._data:
                events.append("\n".join(self._data))
                self._data = []
            return
        if line.startswith(":"):
            return
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data.append(value)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in {None, "text", "output_text"} and "text" in item:
                    parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def visible_delta_from_payload(payload: dict[str, Any]) -> str:
    """只取用户可见正文。忽略 reasoning_content、工具调用和思维字段。"""

    if "error" in payload and "choices" not in payload:
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "上游语言模型返回错误")
        else:
            message = str(error or "上游语言模型返回错误")
        raise ValueError(message)
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    delta = choice.get("delta")
    if delta is None:
        delta = choice.get("message") or {}
    if not isinstance(delta, dict):
        return ""
    return content_to_text(delta.get("content"))


def classify_sse_data(data: str) -> tuple[str, str]:
    stripped = data.strip()
    if stripped == "[DONE]":
        return "done", ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return "bad", ""
    if not isinstance(payload, dict):
        return "skip", ""
    try:
        text = visible_delta_from_payload(payload)
    except ValueError as exc:
        return "error", str(exc)
    if text:
        return "delta", text
    return "skip", ""


async def iter_visible_deltas(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    import codecs

    parser = SseParser()
    decoder = codecs.getincrementaldecoder("utf-8")()

    def handle(data: str) -> str | None:
        kind, text = classify_sse_data(data)
        if kind == "done":
            return DONE  # type: ignore[return-value]
        if kind == "error":
            raise RuntimeError(text)
        if kind == "delta":
            return text
        return None

    async for chunk in chunks:
        for data in parser.feed(decoder.decode(chunk)):
            item = handle(data)
            if item is DONE:
                return
            if item:
                yield item
    tail = decoder.decode(b"", final=True)
    pending = parser.feed(tail) + parser.flush()
    for data in pending:
        item = handle(data)
        if item is DONE:
            return
        if item:
            yield item
