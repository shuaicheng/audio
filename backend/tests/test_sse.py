from __future__ import annotations

import json

import httpx
import pytest

from app.llm import LlmClient
from app.sse import SseParser, classify_sse_data, content_to_text, iter_visible_deltas
from app.tts import TtsClient


def test_sse_fragmentation_and_utf8_split():
    payload = {
        "choices": [
            {
                "delta": {
                    "content": "你",
                    "reasoning_content": "不要朗读",
                    "tool_calls": [{"id": "call_1", "function": {"name": "noop"}}],
                }
            }
        ]
    }
    raw = (
        ": keep-alive\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        'data: {"choices":[{"delta":{"content":[{"type":"text","text":"好"},{"type":"tool","text":"隐藏"}]}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    async def chunks():
        for index in range(0, len(raw), 3):
            yield raw[index : index + 3]

    async def run():
        return [item async for item in iter_visible_deltas(chunks())]

    assert asyncio_run(run()) == ["你", "好"]


def test_parser_splits_inside_data_line_and_ignores_reasoning_only():
    parser = SseParser()
    first = parser.feed('data: {"choices":[{"delta":{"reas')
    assert first == []
    second = parser.feed('oning_content":"secret"}}]}\n\n')
    assert second == ['{"choices":[{"delta":{"reasoning_content":"secret"}}]}']
    assert classify_sse_data(second[0]) == ("skip", "")
    done = parser.feed("data: [DONE]\n\n")
    assert classify_sse_data(done[0]) == ("done", "")


def test_content_list_and_upstream_error():
    assert content_to_text([{"type": "text", "text": "甲"}, {"type": "input_audio", "text": "no"}]) == "甲"
    kind, message = classify_sse_data('{"error":{"message":"boom Bearer super-secret-key"}}')
    assert kind == "error"
    assert "boom" in message


def test_flush_without_trailing_blank_line():
    parser = SseParser()
    assert parser.feed('data: {"choices":[{"delta":{"content":"尾"}}]}') == []
    assert parser.flush() == ['{"choices":[{"delta":{"content":"尾"}}]}']


def test_llm_client_request_shape_and_redaction(settings, caplog):
    secret = "super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {secret}"
        body = json.loads(request.content.decode())
        assert body["stream"] is True
        assert body["model"] == settings.llm_model
        assert "reasoning" not in json.dumps(body)
        text = (
            'data: {"choices":[{"delta":{"content":"可见","reasoning_content":"隐藏"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=text.encode())

    settings.audio_llm_api_key = secret
    client = LlmClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def run():
        import asyncio

        cancelled = asyncio.Event()
        return [item async for item in client.stream([{"role": "user", "content": "hi"}], cancelled)]

    with caplog.at_level("INFO"):
        assert asyncio_run(run()) == ["可见"]
    assert secret not in caplog.text


def test_tts_client_posts_pcm_stream_and_reads_rate(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["body"] = body
        assert request.headers["cookie"] == "session=abc"
        return httpx.Response(200, content=b"\x01\x00\x02", headers={"X-Sample-Rate": "22050"})

    settings.tts_cookie = "session=abc"
    client = TtsClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def run():
        import asyncio

        return [item async for item in client.stream("你好", asyncio.Event())]

    chunks = asyncio_run(run())
    assert seen["body"] == {
        "model": settings.tts_model,
        "input": "你好",
        "voice": "vivian",
        "language": "Chinese",
        "stream": True,
        "response_format": "pcm",
    }
    assert chunks == [(b"\x01\x00\x02", 22050)]
    assert "session=abc" not in json.dumps(seen)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_llm_http_error_hides_key(settings):
    settings.audio_llm_api_key = "super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"super-secret-key")

    client = LlmClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    from app.llm import LlmError

    with pytest.raises(LlmError) as caught:
        async for _ in client.stream([], __import__("asyncio").Event()):
            pass
    assert caught.value.code == "llm_http"
    assert "super-secret-key" not in caught.value.message
