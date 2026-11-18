from __future__ import annotations

import array

import pytest

from app.asr.base import AsrEvent
from app.endpoint import UtteranceEndpoint
from app.orchestrator import Orchestrator
from app.protocol import encode_upstream_frame
from tests.fakes import RecordingTts, ScriptLlm, ScriptedAsr


def _tone(value: int, samples: int) -> bytes:
    return array.array("h", [value] * samples).tobytes()


def test_silence_after_speech_ends_once_and_leading_silence_does_not():
    endpoint = UtteranceEndpoint(16000, speech_rms=0.05, silence_ms=100, min_speech_ms=20)
    assert endpoint.feed(_tone(0, 4000)) is False
    assert endpoint.feed(_tone(8000, 800)) is False
    assert endpoint.heard_speech is True
    assert endpoint.feed(_tone(0, 2000)) is True
    assert endpoint.feed(_tone(0, 2000)) is False


@pytest.mark.asyncio
async def test_auto_endpoint_finishes_without_a_second_llm(settings):
    settings.endpoint_enabled = True
    settings.endpoint_silence_ms = 100
    settings.endpoint_min_speech_ms = 20
    settings.endpoint_speech_rms = 0.05
    asr = ScriptedAsr()

    async def finish_with_text() -> None:
        asr.finished = True
        await asr.queue.put(AsrEvent(kind="final", text="你好"))

    asr.finish = finish_with_text
    llm = ScriptLlm(["好。"])
    orch = Orchestrator(settings, lambda: asr, llm, RecordingTts())
    await orch.start_speech()
    await orch.handle_audio(encode_upstream_frame(1, 16000, _tone(9000, 800)))
    await orch.handle_audio(encode_upstream_frame(2, 16000, _tone(0, 2000)))
    for _ in range(20):
        if llm.calls:
            break
        await __import__("asyncio").sleep(0.05)
    assert asr.finished is True
    assert llm.calls == 1
    await orch.close()
