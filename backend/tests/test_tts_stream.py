from __future__ import annotations

import base64
import json

import httpx

from app.tts import TtsClient
from app.tts_stream import TtsByteDecoder


def _delta(pcm: bytes, *, rate: int = 24000, channels: int = 1) -> str:
    payload = {
        "audio": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": rate,
        "channels": channels,
        "response_format": "pcm",
    }
    return "event:speech.audio.delta\n" + "data: " + json.dumps(payload) + "\n\n"


def _done() -> str:
    return 'event:speech.audio.done\ndata: {"usage":{"seconds":0.2}}\n\n'


def test_sse_fragments_do_not_become_audio_text():
    pcm = b"\x10\x00\x20\x00\x30\x00"
    raw = (_delta(pcm) + _done()).encode()
    decoder = TtsByteDecoder(fallback_rate=16000, content_type="text/event-stream; charset=utf-8")
    pieces = []
    for index in range(0, len(raw), 3):
        pieces.extend(decoder.feed(raw[index : index + 3]))
    pieces.extend(decoder.finish())
    merged = b"".join(piece.pcm for piece in pieces)
    assert merged == pcm
    assert pieces[0].sample_rate == 24000
    assert not merged.startswith(b"event:")
    assert not merged.startswith(b"data:")
    assert all(len(piece.pcm) % 2 == 0 for piece in pieces)


def test_split_base64_across_chunks_keeps_sample_alignment():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    first, second = pcm[:3], pcm[3:]
    raw = (
        "event:speech.audio.delta\n"
        + 'data: {"audio":"'
        + base64.b64encode(first).decode()
        + '","sample_rate":24000,"channels":1}\n\n'
        + _delta(second)
        + _done()
    ).encode()
    decoder = TtsByteDecoder(fallback_rate=16000)
    pieces = []
    for index in range(len(raw)):
        pieces.extend(decoder.feed(raw[index : index + 1]))
    pieces.extend(decoder.finish())
    assert b"".join(piece.pcm for piece in pieces) == pcm


def test_error_event_is_not_played_and_raw_pcm_still_works():
    decoder = TtsByteDecoder(fallback_rate=24000, content_type="text/event-stream")
    try:
        decoder.feed(b'event:error\ndata: {"error":{"message":"bad voice"}}\n\n')
    except Exception as exc:
        assert "bad voice" in str(exc)
    else:
        raise AssertionError("error event should fail")
    raw = TtsByteDecoder(fallback_rate=22050, content_type="audio/pcm")
    piece = raw.feed(b"\x01\x00\x02")[0]
    assert piece.pcm == b"\x01\x00\x02"
    assert piece.sample_rate == 22050


def test_tts_client_decodes_sse_and_keeps_streaming(settings):
    pcm = b"\x11\x00" * 20
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen["stream"] = body["stream"]
        assert "cookie" not in {key.lower() for key in request.headers}
        payload = (_delta(pcm[:10]) + _delta(pcm[10:]) + _done()).encode()
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream; charset=utf-8"})

    client = TtsClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def run():
        import asyncio

        return [item async for item in client.stream("你好", asyncio.Event())]

    chunks = __import__("asyncio").run(run())
    assert seen["stream"] is True
    assert b"".join(item[0] for item in chunks) == pcm
    assert {item[1] for item in chunks} == {24000}
    assert all(not item[0].startswith(b"event:") for item in chunks)
