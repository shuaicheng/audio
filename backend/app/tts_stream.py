"""把 TTS 响应解成 PCM。

stream=true 时，已部署的合成服务返回 text/event-stream：
event:speech.audio.delta，data 里的 audio 是 base64 PCM16，并带 sample_rate。
event:speech.audio.done 只有 usage，不是音频。
stream=false 或明确的 audio/pcm 则是原始 PCM。两种都要边收边解，不能把 SSE 文本当成 PCM。
"""

from __future__ import annotations

import base64
import codecs
import json
from dataclasses import dataclass

from app.protocol import PcmByteAligner
from app.tts import TtsError


@dataclass
class PcmPiece:
    pcm: bytes
    sample_rate: int


class _EventAssembler:
    def __init__(self) -> None:
        self._buf = ""
        self._data: list[str] = []
        self._event = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()

    def feed(self, chunk: bytes) -> list[tuple[str, str]]:
        return self._consume(self._decoder.decode(chunk))

    def finish(self) -> list[tuple[str, str]]:
        events = self._consume(self._decoder.decode(b"", final=True))
        if self._buf:
            self._take_line(self._buf.rstrip("\r"), events)
            self._buf = ""
        if self._data or self._event:
            events.append((self._event, "\n".join(self._data)))
            self._data = []
            self._event = ""
        return events

    def _consume(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        self._buf += text
        events: list[tuple[str, str]] = []
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

    def _take_line(self, line: str, events: list[tuple[str, str]]) -> None:
        if line == "":
            if self._data or self._event:
                events.append((self._event, "\n".join(self._data)))
            self._data = []
            self._event = ""
            return
        if line.startswith(":"):
            return
        if line.startswith("event:"):
            self._event = line[6:].strip()
            return
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data.append(value)


class TtsByteDecoder:
    def __init__(self, *, fallback_rate: int, content_type: str = "") -> None:
        self.fallback_rate = fallback_rate
        self.content_type = content_type or ""
        self._mode: str | None = None
        self._prefix = b""
        self._events = _EventAssembler()
        self._aligner = PcmByteAligner()
        self._done = False
        self.saw_audio = False

    def feed(self, chunk: bytes) -> list[PcmPiece]:
        if self._done or not chunk:
            return []
        if self._mode is None:
            self._prefix += chunk
            decided = self._decide_prefix()
            if decided is None:
                return []
            pending = self._prefix
            self._prefix = b""
            self._mode = decided
            if decided == "pcm":
                return [PcmPiece(pending, self.fallback_rate)]
            return self._feed_sse(pending)
        if self._mode == "pcm":
            return [PcmPiece(chunk, self.fallback_rate)]
        return self._feed_sse(chunk)

    def finish(self) -> list[PcmPiece]:
        if self._mode == "pcm":
            blob = self._prefix
            self._prefix = b""
            return [PcmPiece(blob, self.fallback_rate)] if blob else []
        if self._mode != "sse" and self._prefix:
            if _looks_like_sse(self._prefix):
                self._mode = "sse"
                return self._feed_sse(self._prefix) + self._end_sse()
            blob = self._prefix
            self._prefix = b""
            self._mode = "pcm"
            return [PcmPiece(blob, self.fallback_rate)]
        if self._mode == "sse":
            return self._end_sse()
        return []

    def _decide_prefix(self) -> str | None:
        lowered = self.content_type.lower()
        if "text/event-stream" in lowered:
            return "sse"
        if lowered.startswith("audio/") or "application/octet-stream" in lowered:
            return "pcm"
        if _looks_like_sse(self._prefix):
            return "sse"
        if _undecided_prefix(self._prefix):
            return None
        return "pcm"

    def _feed_sse(self, chunk: bytes) -> list[PcmPiece]:
        pieces: list[PcmPiece] = []
        for name, data in self._events.feed(chunk):
            pieces.extend(self._interpret(name, data))
            if self._done:
                break
        return pieces

    def _end_sse(self) -> list[PcmPiece]:
        pieces: list[PcmPiece] = []
        for name, data in self._events.finish():
            pieces.extend(self._interpret(name, data))
            if self._done:
                break
        self._aligner.flush()
        return pieces

    def _interpret(self, name: str, data: str) -> list[PcmPiece]:
        event_name = name.lower()
        payload = _load_payload(data)
        if _is_error(event_name, payload):
            raise TtsError("tts_stream", _error_message(payload))
        if _is_done(event_name, payload):
            self._done = True
            self._aligner.flush()
            return []
        if payload is None:
            return []
        audio = payload.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            return []
        channels = int(payload.get("channels") or 1)
        if channels != 1:
            raise TtsError("tts_audio", f"暂只播放单声道合成，上游返回了 {channels} 个声道")
        try:
            pcm = base64.b64decode("".join(audio.split()), validate=False)
        except Exception as exc:
            raise TtsError("tts_audio", "合成音频无法从 base64 解码") from exc
        if pcm.startswith((b"event:", b"data:")):
            raise TtsError("tts_audio", "合成结果仍是文本事件，没有解出 PCM")
        rate = _positive_int(payload.get("sample_rate")) or self.fallback_rate
        aligned = self._aligner.feed(pcm)
        if not aligned:
            return []
        self.saw_audio = True
        return [PcmPiece(aligned, rate)]


def _load_payload(data: str) -> dict | None:
    text = data.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        compact = "".join(data.splitlines())
        try:
            payload = json.loads(compact)
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        return payload
    return None


def _is_done(event_name: str, payload: dict | None) -> bool:
    if event_name.endswith("done"):
        return True
    if payload and str(payload.get("type") or "").lower().endswith("done") and "audio" not in payload:
        return True
    return False


def _is_error(event_name: str, payload: dict | None) -> bool:
    if "error" in event_name and (payload is None or "audio" not in (payload or {})):
        return True
    return bool(payload and "error" in payload and "audio" not in payload)


def _error_message(payload: dict | None) -> str:
    if not payload:
        return "语音合成失败"
    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "语音合成失败")
    elif isinstance(error, str) and error:
        message = error
    else:
        message = str(payload.get("message") or "语音合成失败")
    return message[:240]


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _looks_like_sse(buf: bytes) -> bool:
    head = buf.lstrip().lower()
    return head.startswith((b"event:", b"data:", b"id:", b"retry:", b":"))


def _undecided_prefix(buf: bytes) -> bool:
    if len(buf) >= 16:
        return False
    stripped = buf.lstrip()
    if not stripped:
        return True
    for token in (b"event:", b"data:", b"id:", b"retry:"):
        if token.startswith(stripped) or stripped.startswith(token):
            return len(stripped) < len(token)
    return False
