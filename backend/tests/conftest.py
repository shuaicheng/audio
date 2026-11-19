from __future__ import annotations

import asyncio

import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_mode="live",
        asr_provider="realtime",
        asr_error_probe_sec=0.25,
        asr_handshake_timeout_sec=2,
        asr_timeout_sec=0.8,
        llm_first_token_timeout_sec=0.4,
        llm_timeout_sec=2,
        sentence_max_chars=20,
        sentence_min_chars=6,
        sentence_flush_timeout_sec=0.15,
        playback_max_inflight_bytes=8_000_000,
        playback_stall_timeout_sec=1,
        outgoing_queue_chunks=200,
        max_audio_pending_chunks=2,
        max_utterance_bytes=3_200_000,
        mock_token_delay_sec=0,
        audio_llm_api_key="test-key",
        tts_sample_rate=24000,
        asr_sample_rate=16000,
    )


async def collect(orchestrator, seconds: float = 0.6):
    events = []
    frames = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        timeout = min(0.05, max(0.01, deadline - loop.time()))
        try:
            kind, payload = await asyncio.wait_for(orchestrator.outgoing.get(), timeout)
        except TimeoutError:
            continue
        if kind == "json":
            events.append(payload)
        else:
            frames.append(payload)
    return events, frames
