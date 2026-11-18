from __future__ import annotations

import base64
import json

import httpx
import pytest
import websockets

from app.asr.degraded_file import DegradedFileAsr
from app.asr.factory import build_asr
from app.asr.normalize import merge_hypothesis, visible_transcript
from app.asr.realtime import RealtimeAsrEngine
from app.config import Settings


def test_transcript_revision_rules():
    assert merge_hypothesis("你好", "世界") == "你好世界"
    assert merge_hypothesis("你好", "你好世界") == "你好世界"
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
