from __future__ import annotations

import array
import asyncio

import pytest

from app.asr.base import AsrEvent
from app.endpoint import UtteranceEndpoint
from app.orchestrator import Orchestrator
from app.protocol import encode_upstream_frame
from tests.fakes import RecordingTts, ScriptLlm, ScriptedAsr


def _pcm(value: int, milliseconds: int, rate: int = 16000) -> bytes:
    count = int(rate * milliseconds / 1000)
    return array.array("h", [value] * count).tobytes()


def test_weak_continuous_sound_counts_and_near_silence_does_not():
    clock = {"t": 0.0}
    endpoint = UtteranceEndpoint(16000, onset_rms=0.02, onset_ms=480, clock=lambda: clock["t"])
    for step in range(60):
        clock["t"] = step * 0.1
        assert endpoint.feed(_pcm(120, 100)) is False
    assert endpoint.stage == "waiting"
    clock["t"] = 10
    assert endpoint.feed(_pcm(1200, 100)) is False
    clock["t"] = 10.5
    assert endpoint.feed(_pcm(1200, 100)) is False
    assert endpoint.stage == "speech"


def test_leading_silence_noise_and_pause_use_wall_clock():
    clock = {"t": 0.0}
    endpoint = UtteranceEndpoint(16000, onset_ms=480, silence_ms=1400, clock=lambda: clock["t"])
    for step in range(51):
        clock["t"] = step * 0.1
        assert endpoint.feed(_pcm(0, 100)) is False
    assert endpoint.stage == "waiting"

    clock["t"] = 10
    assert endpoint.feed(_pcm(12000, 200)) is False
    clock["t"] = 10.2
    assert endpoint.feed(_pcm(0, 3000)) is False
    assert endpoint.stage == "waiting"

    clock["t"] = 20
    assert endpoint.feed(_pcm(12000, 100)) is False
    clock["t"] = 20.5
    assert endpoint.feed(_pcm(12000, 100)) is False
    assert endpoint.stage == "speech"
    clock["t"] = 21.0
    assert endpoint.feed(_pcm(0, 100)) is False
    assert endpoint.stage == "pause"
    assert endpoint.fired is False
    clock["t"] = 22.4
    assert endpoint.feed(_pcm(0, 100)) is True
    assert endpoint.feed(_pcm(0, 100)) is False


@pytest.mark.asyncio
async def test_five_seconds_before_speech_keeps_listening(settings):
    settings.max_audio_pending_chunks = 200
    clock = {"t": 0.0}
    asr = ScriptedAsr()
    llm = ScriptLlm(["不该出现"])
    orch = Orchestrator(settings, lambda: asr, llm, RecordingTts())
    await orch.start_speech()
    orch.endpoint.clock = lambda: clock["t"]
    for step in range(51):
        clock["t"] = step * 0.1
        await orch.handle_audio(encode_upstream_frame(step + 1, 16000, _pcm(0, 100)))
    assert orch.phase == "listening"
    assert orch.endpoint.stage == "waiting"
    assert orch.utterance_bytes == 51 * len(_pcm(0, 100))
    assert llm.calls == 0
    await orch.close()


@pytest.mark.asyncio
async def test_manual_silence_finish_does_not_ask_the_model(settings):
    asr = ScriptedAsr()

    async def empty_final() -> None:
        asr.finished = True
        await asr.queue.put(AsrEvent(kind="final", text="  "))

    asr.finish = empty_final
    llm = ScriptLlm(["不该出现"])
    orch = Orchestrator(settings, lambda: asr, llm, RecordingTts())
    await orch.start_speech()
    await orch.handle_audio(encode_upstream_frame(1, 16000, _pcm(0, 5000)))
    await orch.finish_speech()
    await orch.finish_speech()
    await asyncio.sleep(0.1)
    assert asr.finished is True
    assert llm.calls == 0
    await orch.close()


@pytest.mark.asyncio
async def test_speech_then_endpoint_asks_once(settings):
    settings.max_audio_pending_chunks = 200
    clock = {"t": 0.0}
    asr = ScriptedAsr()

    async def final_text() -> None:
        asr.finished = True
        await asr.queue.put(AsrEvent(kind="final", text="你好"))

    asr.finish = final_text
    llm = ScriptLlm(["好。"])
    orch = Orchestrator(settings, lambda: asr, llm, RecordingTts())
    await orch.start_speech()
    orch.endpoint.clock = lambda: clock["t"]
    seq = 1
    for stamp in (0.0, 0.5):
        clock["t"] = stamp
        await orch.handle_audio(encode_upstream_frame(seq, 16000, _pcm(12000, 100)))
        seq += 1
    clock["t"] = 0.9
    await orch.handle_audio(encode_upstream_frame(seq, 16000, _pcm(0, 100)))
    seq += 1
    assert orch.phase == "listening"
    assert llm.calls == 0
    clock["t"] = 2.4
    await orch.handle_audio(encode_upstream_frame(seq, 16000, _pcm(0, 100)))
    for _ in range(20):
        if llm.calls:
            break
        await asyncio.sleep(0.05)
    assert asr.finished is True
    assert llm.calls == 1
    await orch.close()


@pytest.mark.asyncio
async def test_manual_finish_racing_endpoint_submits_the_first_final_once(settings):
    settings.max_audio_pending_chunks = 200
    clock = {"t": 0.0}
    asr = ScriptedAsr()
    commits = 0

    async def final_text() -> None:
        nonlocal commits
        commits += 1
        await asr.queue.put(AsrEvent(kind="final", text="一二三。"))
        await asr.queue.put(AsrEvent(kind="final", text="一二三。一二三。"))

    asr.finish = final_text
    llm = ScriptLlm(["好。"])
    orch = Orchestrator(settings, lambda: asr, llm, RecordingTts())
    await orch.start_speech()
    orch.endpoint.clock = lambda: clock["t"]
    seq = 1
    for stamp in (0.0, 0.5):
        clock["t"] = stamp
        await orch.handle_audio(encode_upstream_frame(seq, 16000, _pcm(12000, 100)))
        seq += 1
    clock["t"] = 0.9
    await orch.handle_audio(encode_upstream_frame(seq, 16000, _pcm(0, 100)))
    seq += 1
    clock["t"] = 2.4
    await orch.handle_audio(encode_upstream_frame(seq, 16000, _pcm(0, 100)))
    await orch.finish_speech()
    await orch.handle_text('{"type":"speech.finish","auto_submit":true}')
    for _ in range(20):
        if llm.calls:
            break
        await asyncio.sleep(0.05)
    assert commits == 1
    assert llm.calls == 1
    assert llm.messages[0][-1]["content"] == "一二三。"
    await orch.close()
