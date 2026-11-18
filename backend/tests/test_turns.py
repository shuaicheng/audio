from __future__ import annotations

import asyncio

import pytest

from app.llm import LlmError
from app.orchestrator import Orchestrator
from app.protocol import decode_tts_frame, encode_upstream_frame
from tests.conftest import collect
from tests.fakes import PausingLlm, RecordingTts, ScriptLlm, ScriptedAsr


def _orch(settings, llm, tts=None, asr=None):
    return Orchestrator(settings, lambda: asr or ScriptedAsr(), llm, tts or RecordingTts())


@pytest.mark.asyncio
async def test_display_is_not_blocked_on_sentence_and_trailing_flush(settings):
    settings.sentence_flush_timeout_sec = 5

    class HoldLlm:
        def __init__(self):
            self.hold = asyncio.Event()

        async def stream(self, messages, cancelled):
            yield "你好"
            await self.hold.wait()

    llm = HoldLlm()
    tts = RecordingTts()
    orch = _orch(settings, llm, tts)
    await orch.submit_text("问")
    early, _ = await collect(orch, 0.2)
    assert any(event["type"] == "llm.delta" and event["text"] == "你好" for event in early)
    assert tts.sentences == []
    llm.hold.set()
    await orch.turn.task
    events, frames = await collect(orch, 0.3)
    assert any(event["type"] == "llm.done" for event in early + events)
    assert tts.sentences == ["你好"]
    assert frames
    decoded = decode_tts_frame(frames[0])
    assert decoded["sample_rate"] == 24000
    assert decoded["turn_id"]
    assert len(decoded["pcm"]) % 2 == 0


@pytest.mark.asyncio
async def test_sentence_timeout_flushes_before_later_tokens(settings):
    settings.sentence_flush_timeout_sec = 0.1

    class IdleLlm:
        def __init__(self):
            self.second_at = None

        async def stream(self, messages, cancelled):
            yield "这是一段没有句号的话"
            await asyncio.sleep(0.35)
            self.second_at = asyncio.get_running_loop().time()
            yield "后续"

    llm = IdleLlm()
    tts = RecordingTts()
    orch = _orch(settings, llm, tts)
    await orch.submit_text("问")
    await orch.turn.task
    assert tts.sentences[0] == "这是一段没有句号的话"
    assert tts.started[0] < llm.second_at
    assert any(sentence == "后续" or sentence.startswith("后续") for sentence in tts.sentences)


@pytest.mark.asyncio
async def test_cancel_stale_turn_and_explicit_interrupted_history(settings):
    llm = PausingLlm()
    tts = RecordingTts()
    orch = _orch(settings, llm, tts)
    await orch.submit_text("第一句")
    for _ in range(20):
        events, _ = await collect(orch, 0.05)
        if any(event["type"] == "llm.delta" for event in events):
            break
    else:
        raise AssertionError("没有等到首个回答增量")
    await orch.stop_answer("user_stop")
    llm.released.set()
    await asyncio.sleep(0.05)
    await orch.submit_text("第二句")
    await orch.turn.task
    events, _ = await collect(orch, 0.3)
    blob = " ".join(event.get("text", "") + event.get("delta", "") for event in events)
    assert "不该出现" not in blob
    assert llm.saw_forbidden is False
    assistant = [item for item in orch.history if item["role"] == "assistant"]
    assert assistant[0]["status"] == "interrupted"
    assert "回答已中断" in assistant[0]["content"]
    assert "部分回答" in assistant[0]["content"]
    assert any(item["content"] == "新的回答。" and item["status"] == "complete" for item in assistant)
    sent = llm.messages[1]
    assert any("回答已中断" in item["content"] for item in sent if item["role"] == "assistant")
    assert not any("不该出现" in item["content"] for item in sent)


@pytest.mark.asyncio
async def test_llm_error_after_partial_and_tts_only_failure(settings):
    llm = ScriptLlm(["已经说了一半"], error=LlmError("llm_stream", "上游中断"))
    orch = _orch(settings, llm, RecordingTts())
    await orch.submit_text("问")
    await orch.turn.task
    events, _ = await collect(orch, 0.2)
    failure = next(event for event in events if event["type"] == "llm.error")
    assert failure["partial_text"] == "已经说了一半"
    assert not any(event["type"] == "llm.done" for event in events)
    assert orch.history[-1]["status"] == "error"
    assert "已经说了一半" in orch.history[-1]["content"]

    llm_ok = ScriptLlm(["完整回答。"])
    tts = RecordingTts(fail=True)
    orch = _orch(settings, llm_ok, tts)
    await orch.submit_text("问")
    await orch.turn.task
    events, _ = await collect(orch, 0.2)
    assert any(event["type"] == "tts.failed" and "文字" not in event["message"] or event["type"] == "tts.failed" for event in events)
    assert any(event["type"] == "llm.done" and event["text"] == "完整回答。" for event in events)
    assert orch.history[-1]["status"] == "complete"
    assert orch.history[-1]["content"] == "完整回答。"


@pytest.mark.asyncio
async def test_llm_timeout_and_playback_hold(settings):
    settings.llm_first_token_timeout_sec = 0.1

    class SlowLlm:
        async def stream(self, messages, cancelled):
            await asyncio.sleep(2)
            yield "太晚"

    orch = _orch(settings, SlowLlm(), RecordingTts())
    await orch.submit_text("问")
    await orch.turn.task
    events, _ = await collect(orch, 0.2)
    assert any(event["type"] == "timeout" and event["scope"] == "llm" for event in events)

    settings.llm_first_token_timeout_sec = 1
    settings.playback_stall_timeout_sec = 1
    llm = ScriptLlm(["你好。"])
    orch = _orch(settings, llm, RecordingTts())
    await orch.gate.set_hold(True)
    await orch.submit_text("问")
    _, frames = await collect(orch, 0.2)
    assert frames == []
    await orch.gate.set_hold(False)
    await orch.turn.task
    _, frames = await collect(orch, 0.3)
    assert frames


@pytest.mark.asyncio
async def test_one_utterance_starts_one_llm_and_partials_replace(settings):
    asr = ScriptedAsr()
    llm = ScriptLlm(["好的。"])
    orch = _orch(settings, llm, RecordingTts(), asr)
    await orch.start_speech()
    await asr.queue.put(__import__("app.asr.base", fromlist=["AsrEvent"]).AsrEvent(kind="partial", text="今天"))
    await asr.queue.put(__import__("app.asr.base", fromlist=["AsrEvent"]).AsrEvent(kind="partial", text="今天天气不错"))
    await asyncio.sleep(0.05)
    events, _ = await collect(orch, 0.2)
    partials = [event["text"] for event in events if event["type"] == "asr.partial"]
    assert partials == ["今天", "今天天气不错"]
    assert llm.calls == 0
    await asr.queue.put(__import__("app.asr.base", fromlist=["AsrEvent"]).AsrEvent(kind="final", text="今天天气不错"))
    await orch.listen_task
    await orch.turn.task
    assert llm.calls == 1
    assert llm.messages[0][-1]["content"] == "今天天气不错"
    await orch.close()


@pytest.mark.asyncio
async def test_queue_limit_and_malformed_audio(settings):
    asr = ScriptedAsr(block_audio=True)
    orch = _orch(settings, ScriptLlm(["好"]), RecordingTts(), asr)
    await orch.start_speech()
    await asyncio.sleep(0.05)
    for seq in range(1, 5):
        await orch.handle_audio(encode_upstream_frame(seq, 16000, b"\x01\x00" * 4))
        await asyncio.sleep(0)
    events, _ = await collect(orch, 0.2)
    assert any(event["type"] == "queue.limit" and event["scope"] == "asr" for event in events)
    asr.block.set()
    bad = encode_upstream_frame(1, 16000, b"\x00\x01")[:-1]
    await orch.handle_audio(bad)
    await orch.handle_audio(encode_upstream_frame(1, 8000, b"\x00\x01"))
    events, _ = await collect(orch, 0.2)
    codes = [event["code"] for event in events if event["type"] == "audio.malformed"]
    assert "malformed_audio" in codes
    assert orch.phase == "listening"
    await orch.close()


@pytest.mark.asyncio
async def test_rejected_asr_does_not_look_like_listening(settings):
    class RejectAsr(ScriptedAsr):
        async def start(self) -> None:
            self.accepted = False
            await self.queue.put(
                __import__("app.asr.base", fromlist=["AsrEvent"]).AsrEvent(
                    kind="unsupported",
                    code="model_not_found",
                    message="服务端不接受该识别模型",
                )
            )

    orch = _orch(settings, ScriptLlm(["不会用到"]), RecordingTts(), RejectAsr())
    await orch.start_speech()
    events, _ = await collect(orch, 0.5)
    assert orch.phase == "ready"
    assert not any(event["type"] == "speech.started" for event in events)
    assert any(event["type"] == "asr.unsupported" for event in events)
    await orch.close()
