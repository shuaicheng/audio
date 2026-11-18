from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.asr.factory import build_asr
from app.asr.qwen_streaming import QwenWorkerPool
from app.config import Settings
from app.demo import MockLlm, MockTts
from app.llm import LlmClient
from app.orchestrator import Orchestrator
from app.tts import TtsClient

logger = logging.getLogger("voice")


class RedactFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self.secrets = [secret for secret in secrets if secret]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self.secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _origin_allowed(origin: str | None, settings: Settings) -> bool:
    allowed = settings.cors_origin_list()
    if "*" in allowed:
        return True
    if not origin:
        return True
    return origin in allowed


class Hub:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        use_qwen = settings.app_mode != "mock" and settings.provider_name() == "qwen_streaming"
        self.pool = QwenWorkerPool(settings) if use_qwen else None
        if settings.app_mode == "mock":
            self.llm = MockLlm(settings)
            self.tts = MockTts(settings)
        else:
            self.llm = LlmClient(settings)
            self.tts = TtsClient(settings)

    def open(self) -> Orchestrator:
        return Orchestrator(self.settings, lambda: build_asr(self.settings, self.pool), self.llm, self.tts)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    hub = Hub(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if hub.pool is not None:
            await hub.pool.shutdown()

    app = FastAPI(title="语音对话", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.hub = hub
    origins = settings.cors_origin_list()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logging.getLogger().addFilter(
        RedactFilter(
            [
                settings.audio_llm_api_key,
                settings.asr_authorization,
                settings.asr_cookie,
                settings.tts_authorization,
                settings.tts_cookie,
                settings.app_auth_token,
            ]
        )
    )

    @app.get("/api/health")
    async def health() -> dict:
        provider = "mock" if settings.app_mode == "mock" else settings.provider_name()
        return {
            "status": "ok",
            "mode": settings.app_mode,
            "asr_provider": provider,
            "asr_model": settings.asr_model,
            "llm_model": settings.llm_model,
            "tts_model": settings.tts_model,
            "tts_sample_rate": settings.tts_sample_rate,
            "asr_sample_rate": settings.asr_sample_rate,
            "llm_key": settings.redacted_key_state(),
            "validated_live": False,
            "timestamps": False,
            "duplex": "push_to_talk",
        }

    @app.websocket("/api/ws")
    async def session_socket(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket.headers.get("origin"), settings):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if settings.app_auth_token:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), 5)
                payload = json.loads(raw)
                token = str(payload.get("token") or "") if isinstance(payload, dict) else ""
            except Exception:
                await websocket.close(code=1008)
                return
            if payload.get("type") != "auth" or not hmac.compare_digest(token, settings.app_auth_token):
                await websocket.close(code=1008)
                return
        orchestrator = hub.open()
        sender = asyncio.create_task(_send_loop(websocket, orchestrator))
        await orchestrator.greet()
        try:
            while True:
                incoming = await websocket.receive()
                if incoming["type"] == "websocket.disconnect":
                    break
                data_bytes = incoming.get("bytes")
                data_text = incoming.get("text")
                if data_bytes is not None:
                    await orchestrator.handle_audio(data_bytes)
                elif data_text is not None:
                    await orchestrator.handle_text(data_text)
        except WebSocketDisconnect:
            logger.info("会话断开 %s", orchestrator.session_id)
        finally:
            await orchestrator.close()
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

    return app


async def _send_loop(websocket: WebSocket, orchestrator: Orchestrator) -> None:
    try:
        while True:
            kind, payload = await orchestrator.outgoing.get()
            if kind == "json":
                await websocket.send_json(payload)
            else:
                await websocket.send_bytes(payload)  # type: ignore[arg-type]
    except (WebSocketDisconnect, RuntimeError):
        return


app = create_app()
