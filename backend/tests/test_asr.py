from __future__ import annotations

import base64
import json

import httpx
import pytest
import websockets

from app.asr.degraded_file import DegradedFileAsr
from app.asr.factory import build_asr
from app.asr.normalize import merge_hypothesis, resolve_completed_transcript, visible_transcript
from app.asr.realtime import RealtimeAsrEngine
from app.config import Settings


@pytest.mark.asyncio
async def test_mock_partial_reaches_the_client_before_finish(settings):
    import array

    from app.asr.mock import MockAsrEngine
    from app.orchestrator import Orchestrator
    from app.protocol import encode_upstream_frame
    from tests.conftest import collect

    class _Idle:
        async def stream(self, messages, cancelled):
            if False:
                yield ""

    settings.max_audio_pending_chunks = 20
    loud = array.array("h", [4000] * 8000).tobytes()
    engine = MockAsrEngine(settings)
    orch = Orchestrator(settings, lambda: engine, _Idle(), _Idle())
    await orch.start_speech()
    await orch.handle_audio(encode_upstream_frame(1, 16000, loud))
    events, _ = await collect(orch, 0.3)
    partials = [event for event in events if event["type"] == "asr.partial"]
    assert partials
    assert "不是真实识别" in partials[-1]["text"]
    assert not any(event["type"] == "turn.started" for event in events)
    await orch.close()


@pytest.mark.asyncio
async def test_partial_survives_the_idle_gap_before_the_first_word(settings):
    """The first hypothesis can arrive seconds after capture starts. An idle poll must not close ASR."""
    import array
    import asyncio

    from app.asr.mock import MockAsrEngine
    from app.orchestrator import Orchestrator
    from app.protocol import encode_upstream_frame
    from tests.conftest import collect

    class _Idle:
        async def stream(self, messages, cancelled):
            if False:
                yield ""

    settings.max_audio_pending_chunks = 20
    loud = array.array("h", [4000] * 8000).tobytes()
    engine = MockAsrEngine(settings)
    orch = Orchestrator(settings, lambda: engine, _Idle(), _Idle())
    await orch.start_speech()
    await asyncio.sleep(0.35)
    await orch.handle_audio(encode_upstream_frame(1, 16000, loud))
    events, _ = await collect(orch, 0.4)
    partials = [event for event in events if event["type"] == "asr.partial"]
    assert partials
    assert "不是真实识别" in partials[-1]["text"]
    await orch.finish_speech(auto_submit=True)
    finished, _ = await collect(orch, 0.4)
    assert any(event["type"] == "asr.final" for event in finished)
    assert sum(1 for event in finished if event["type"] == "turn.started") == 1
    await orch.close()


def test_transcript_revision_rules():
    assert merge_hypothesis("你好", "世界") == "你好世界"
    assert merge_hypothesis("你好", "你好世界") == "你好世界"
    once = "一二三四五六七八九。"
    assert merge_hypothesis(once, once) == once
    assert merge_hypothesis("你好", "你好你好") == "你好你好"
    assert merge_hypothesis(merge_hypothesis(merge_hypothesis("你", "好"), "你"), "好") == "你好你好"
    joined = once + once
    assert resolve_completed_transcript(once, [once, once], joined) == once
    assert resolve_completed_transcript("你好你好", ["你", "好", "你", "好"], "你好你好") == "你好你好"
    assert resolve_completed_transcript("你好你好", ["你好", "你好你好"], "你好你好你好") == "你好你好"
    wrapped = "language Chinese<asr_text>" + once
    assert resolve_completed_transcript(wrapped, [wrapped], "language Chinese<asr_text>" + once) == wrapped
    assert visible_transcript(resolve_completed_transcript(wrapped, [wrapped, wrapped], wrapped + wrapped)) == once
    assert visible_transcript("language Chinese<asr_text>今天天气不错") == "今天天气不错"
    assert visible_transcript("language English<asr_text>Hello.language English<asr_text>World") == "Hello.World"
    assert visible_transcript("language Chi") == ""
    assert visible_transcript("你好") == "你好"


async def _serve(handler):
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


@pytest.mark.asyncio
async def test_realtime_handshake_append_and_replace(settings: Settings):
    seen: list[dict] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created", "id": "sess-test"}))
        async for raw in ws:
            event = json.loads(raw)
            seen.append(event)
            if event["type"] == "session.update" and event.get("model") != settings.asr_model:
                await ws.send(json.dumps({"type": "error", "error": "missing model", "code": "model_not_found"}))
            elif event["type"] == "input_audio_buffer.append":
                pcm = base64.b64decode(event["audio"])
                assert pcm == b"\x01\x00" * 16
                await ws.send(json.dumps({"type": "transcription.delta", "delta": "你"}))
                await ws.send(json.dumps({"type": "transcription.delta", "delta": "你好呀"}))
            elif event["type"] == "input_audio_buffer.commit" and event.get("final") is True:
                await ws.send(json.dumps({"type": "transcription.done", "text": "language Chinese<asr_text>你好呀"}))
                return

    server, port = await _serve(handler)
    engine = RealtimeAsrEngine(settings, url=f"ws://127.0.0.1:{port}")
    try:
        await engine.start()
        ready = await engine.next_event(1)
        assert ready is not None and ready.kind == "ready" and ready.validated is True
        await engine.push_audio(b"\x01\x00" * 16, 1)
        partials = []
        for _ in range(4):
            event = await engine.next_event(1)
            if event is not None and event.kind == "partial":
                partials.append(event.text)
            if len(partials) == 2:
                break
        await engine.finish()
        final = await engine.next_event(1)
        assert partials == ["你", "你好呀"]
        assert final is not None and final.kind == "final" and final.text == "你好呀"
        assert final.timestamps is False
        kinds = [item["type"] for item in seen]
        assert kinds[0] == "session.update"
        assert kinds[1] == "input_audio_buffer.commit" and seen[1]["final"] is False
        assert "input_audio_buffer.append" in kinds
        assert kinds[-1] == "input_audio_buffer.commit" and seen[-1]["final"] is True
    finally:
        await engine.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_done_concatenation_does_not_double_one_sentence(settings: Settings):
    """vLLM sets done.text to the raw join of deltas. A repeated full-hypothesis delta must not be submitted twice."""
    once = "一二三四五六七八九。"
    trace: list[str] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created", "id": "sess-once"}))
        async for raw in ws:
            event = json.loads(raw)
            trace.append(event["type"] + (":final" if event.get("final") is True else ""))
            if event["type"] == "input_audio_buffer.append":
                trace.append("upstream.delta")
                await ws.send(json.dumps({"type": "transcription.delta", "delta": once}))
            elif event["type"] == "input_audio_buffer.commit" and event.get("final") is True:
                trace.append("upstream.delta")
                await ws.send(json.dumps({"type": "transcription.delta", "delta": once}))
                trace.append("upstream.done")
                await ws.send(json.dumps({"type": "transcription.done", "text": once + once}))
                await ws.send(json.dumps({"type": "transcription.done", "text": once + once + once}))
                return

    server, port = await _serve(handler)
    settings.asr_error_probe_sec = 0.05
    from app.orchestrator import Orchestrator
    from app.protocol import encode_upstream_frame
    from tests.conftest import collect
    from tests.fakes import RecordingTts, ScriptLlm

    llm = ScriptLlm(["好。"])
    engine = RealtimeAsrEngine(settings, url=f"ws://127.0.0.1:{port}")
    orch = Orchestrator(settings, lambda: engine, llm, RecordingTts())
    try:
        await orch.start_speech()
        await orch.handle_audio(encode_upstream_frame(1, 16000, b"\x01\x00" * 1600))
        live, _ = await collect(orch, 0.6)
        partials = [event["text"] for event in live if event["type"] == "asr.partial"]
        assert partials == [once]
        assert not any(event["type"] == "turn.started" for event in live)
        await orch.finish_speech()
        await orch.finish_speech()
        finished, _ = await collect(orch, 0.6)
        finals = [event["text"] for event in finished if event["type"] == "asr.final"]
        turns = [event["text"] for event in finished if event["type"] == "turn.started"]
        assert finals == [once]
        assert turns == [once]
        assert llm.calls == 1
        assert llm.messages[0][-1]["content"] == once
        assert trace.count("input_audio_buffer.append") == 1
        assert trace.count("input_audio_buffer.commit:final") == 1
        assert "upstream.done" in trace
    finally:
        await orch.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_realtime_partial_before_finish_and_hides_language_prefix(settings: Settings):
    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created", "id": "sess-partial"}))
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "input_audio_buffer.append":
                await ws.send(json.dumps({"type": "transcription.delta", "delta": "language "}))
                await ws.send(json.dumps({"type": "transcription.delta", "delta": "Chinese<asr_text>你好"}))
            elif event["type"] == "input_audio_buffer.commit" and event.get("final") is True:
                await ws.send(json.dumps({"type": "transcription.done", "text": "language Chinese<asr_text>你好"}))
                return

    server, port = await _serve(handler)
    engine = RealtimeAsrEngine(settings, url=f"ws://127.0.0.1:{port}")
    try:
        await engine.start()
        assert (await engine.next_event(1)).kind == "ready"
        await engine.push_audio(b"\x01\x00" * 160, 1)
        partial = await engine.next_event(1)
        assert engine._finish_sent is False
        assert partial is not None and partial.kind == "partial"
        assert partial.text == "你好"
        assert "language" not in partial.text
        await engine.finish()
        final = await engine.next_event(1)
        assert final is not None and final.kind == "final" and final.text == "你好"
    finally:
        await engine.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_realtime_model_rejection_does_not_send_audio(settings: Settings):
    seen: list[str] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created", "id": "sess-no"}))
        async for raw in ws:
            event = json.loads(raw)
            seen.append(event["type"])
            if event["type"] == "session.update":
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "error": "The model `Qwen3-ASR-1.7B` does not exist.",
                            "code": "model_not_found",
                        }
                    )
                )

    server, port = await _serve(handler)
    engine = RealtimeAsrEngine(settings, url=f"ws://127.0.0.1:{port}")
    try:
        await engine.start()
        event = await engine.next_event(1)
        assert event is not None
        assert event.kind == "unsupported"
        assert event.validated is False
        assert "input_audio_buffer.append" not in seen
        assert seen == ["session.update"]
    finally:
        await engine.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_realtime_sessions_are_isolated(settings: Settings):
    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created", "id": "sess"}))
        label = ""
        async for raw in ws:
            event = json.loads(raw)
            if event["type"] == "input_audio_buffer.append":
                pcm = base64.b64decode(event["audio"])
                label = "甲" if pcm[:2] == b"\x11\x00" else "乙"
                await ws.send(json.dumps({"type": "transcription.delta", "delta": label}))
            elif event["type"] == "input_audio_buffer.commit" and event.get("final") is True:
                await ws.send(json.dumps({"type": "transcription.done", "text": label}))
                return

    server, port = await _serve(handler)
    first = RealtimeAsrEngine(settings, url=f"ws://127.0.0.1:{port}")
    second = RealtimeAsrEngine(settings, url=f"ws://127.0.0.1:{port}")
    try:
        await first.start()
        await second.start()
        assert (await first.next_event(1)).kind == "ready"
        assert (await second.next_event(1)).kind == "ready"
        await first.push_audio(b"\x11\x00" * 8, 1)
        await second.push_audio(b"\x22\x00" * 8, 1)
        assert (await first.next_event(1)).text == "甲"
        assert (await second.next_event(1)).text == "乙"
    finally:
        await first.close()
        await second.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_degraded_file_is_opt_in_and_labeled(settings: Settings):
    settings.asr_provider = "realtime"
    assert build_asr(settings, realtime_url="ws://127.0.0.1:9").mode == "realtime"
    settings.asr_provider = "degraded_file"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.content)
        return httpx.Response(200, json={"text": "文件结果"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    engine = DegradedFileAsr(settings, client=client)
    await engine.start()
    ready = await engine.next_event(1)
    assert ready.degraded is True
    await engine.push_audio(b"\x01\x00" * 20, 1)
    assert calls == []
    await engine.finish()
    final = await engine.next_event(1)
    assert final.kind == "final" and final.degraded is True
    assert final.strategy == "file_endpoint"
    assert final.text == "文件结果"
    assert len(calls) == 1
    settings.asr_degraded_chunk_sec = 0.01
    calls.clear()
    engine = DegradedFileAsr(settings, client=client)
    await engine.start()
    await engine.next_event(1)
    await engine.push_audio(b"\x01\x00" * 400, 1)
    partial = await engine.next_event(1)
    assert partial.degraded is True and partial.strategy == "cumulative_file_reupload"
    await engine.push_audio(b"\x01\x00" * 400, 2)
    await engine.finish()
    texts = []
    while True:
        event = await engine.next_event(0.2)
        if event is None:
            break
        texts.append(event)
    assert calls
    assert all(item.strategy == "cumulative_file_reupload" for item in texts if item.kind in {"partial", "final"})
    assert len(calls) >= 2
    assert len(calls[-1]) > len(calls[0])
    await engine.close()
    await client.aclose()
