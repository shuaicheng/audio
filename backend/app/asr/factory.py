from __future__ import annotations

from app.asr.degraded_file import DegradedFileAsr
from app.asr.mock import MockAsrEngine
from app.asr.qwen_streaming import QwenStreamingAsr, QwenWorkerPool
from app.asr.realtime import RealtimeAsrEngine
from app.config import Settings


def build_asr(settings: Settings, pool: QwenWorkerPool | None = None, realtime_url: str | None = None):
    if settings.app_mode == "mock":
        return MockAsrEngine(settings)
    provider = settings.provider_name()
    if provider == "qwen_streaming":
        return QwenStreamingAsr(settings, pool)
    if provider == "degraded_file":
        return DegradedFileAsr(settings)
    if provider != "realtime":
        raise ValueError(f"未知 ASR_PROVIDER：{settings.asr_provider}")
    return RealtimeAsrEngine(settings, url=realtime_url)
