from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.redact import scrub

logger = logging.getLogger(__name__)


class TtsError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class TtsClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    async def stream(self, text: str, cancelled: asyncio.Event) -> AsyncIterator[tuple[bytes, int]]:
        headers: dict[str, str] = {"Accept": "text/event-stream, application/octet-stream, audio/pcm"}
        if self.settings.tts_authorization:
            value = self.settings.tts_authorization
            headers["Authorization"] = value if value.lower().startswith("bearer ") else f"Bearer {value}"
        if self.settings.tts_cookie:
            headers["Cookie"] = self.settings.tts_cookie
        body = {
            "model": self.settings.tts_model,
            "input": text,
            "voice": self.settings.tts_voice,
            "language": self.settings.tts_language,
            "stream": True,
            "response_format": "pcm",
        }
        timeout = httpx.Timeout(connect=10.0, read=self.settings.tts_timeout_sec, write=30.0, pool=10.0)
        client = self._client or httpx.AsyncClient(timeout=timeout)
        owns = self._client is None
        secrets = [self.settings.tts_authorization, self.settings.tts_cookie]
        try:
            async with client.stream("POST", self.settings.tts_url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    raise TtsError("tts_http", f"语音合成失败（HTTP {response.status_code}）")
                content_type = response.headers.get("content-type", "")
                from app.tts_stream import TtsByteDecoder

                decoder = TtsByteDecoder(
                    fallback_rate=_header_rate(response.headers) or self.settings.tts_sample_rate,
                    content_type=content_type,
                )
                logger.info("语音合成响应 content-type=%s", content_type.split(";")[0] or "unknown")

                async def pieces():
                    async for chunk in response.aiter_bytes():
                        if cancelled.is_set():
                            return
                        for piece in decoder.feed(chunk):
                            yield piece
                    for piece in decoder.finish():
                        if cancelled.is_set():
                            return
                        yield piece

                async for piece in pieces():
                    if cancelled.is_set():
                        return
                    if piece.pcm:
                        yield piece.pcm, piece.sample_rate
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError("tts_upstream", "语音合成失败：" + scrub(str(exc), secrets)) from None
        finally:
            if owns:
                await client.aclose()


def _header_rate(headers: httpx.Headers) -> int | None:
    for name in ("x-audio-sample-rate", "x-sample-rate"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            value = int(str(raw).strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return None
