from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.redact import scrub
from app.sse import iter_visible_deltas


class LlmError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class LlmClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    async def stream(self, messages: list[dict], cancelled: asyncio.Event) -> AsyncIterator[str]:
        if not self.settings.audio_llm_api_key:
            raise LlmError("llm_unconfigured", "服务端未配置 AUDIO_LLM_API_KEY")
        headers = {
            "Authorization": f"Bearer {self.settings.audio_llm_api_key}",
            "Accept": "text/event-stream",
        }
        body = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": self.settings.llm_temperature,
        }
        timeout = httpx.Timeout(connect=10.0, read=self.settings.llm_timeout_sec, write=30.0, pool=10.0)
        client = self._client or httpx.AsyncClient(timeout=timeout)
        owns = self._client is None
        try:
            async with client.stream("POST", self.settings.llm_url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    raise LlmError("llm_http", f"语言模型请求失败（HTTP {response.status_code}）")

                async def chunks() -> AsyncIterator[bytes]:
                    async for chunk in response.aiter_bytes():
                        if cancelled.is_set():
                            return
                        yield chunk

                try:
                    async for delta in iter_visible_deltas(chunks()):
                        if cancelled.is_set():
                            return
                        yield delta
                except RuntimeError as exc:
                    raise LlmError("llm_stream", scrub(str(exc), [self.settings.audio_llm_api_key])) from None
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(
                "llm_upstream",
                "语言模型请求失败：" + scrub(str(exc), [self.settings.audio_llm_api_key]),
            ) from None
        finally:
            if owns:
                await client.aclose()
