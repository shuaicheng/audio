from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.main import RedactFilter, _origin_allowed, create_app
from app.protocol import PcmByteAligner, decode_tts_frame, encode_tts_frame, encode_upstream_frame, decode_upstream_frame
from app.redact import scrub


def test_health_and_logs_do_not_leak_key(settings):
    settings.app_mode = "mock"
    settings.audio_llm_api_key = "super-secret-key"
    settings.app_auth_token = ""
    app = create_app(settings)
    with TestClient(app) as client:
        payload = client.get("/api/health").json()
    assert payload["mode"] == "mock"
    assert payload["validated_live"] is False
    assert payload["llm_key"] == "set"
    assert "super-secret-key" not in str(payload)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "token super-secret-key", (), None)
    assert RedactFilter(["super-secret-key"]).filter(record) is True
    assert "super-secret-key" not in record.getMessage()
    assert "super-secret-key" not in scrub("Authorization: Bearer super-secret-key")
    assert repr(settings) != "" and "super-secret-key" not in repr(settings)


def test_origin_and_auth(settings):
    settings.cors_origins = "http://localhost:5173"
    settings.app_auth_token = "gate-token"
    settings.app_mode = "mock"
    assert _origin_allowed("http://localhost:5173", settings) is True
    assert _origin_allowed("https://evil.example", settings) is False
    app = create_app(settings)
    with TestClient(app) as client:
        with client.websocket_connect("/api/ws") as ws:
            ws.send_json({"type": "auth", "token": "wrong"})
            with __import__("pytest").raises(Exception):
                ws.receive_json()
        with client.websocket_connect("/api/ws") as ws:
            ws.send_json({"type": "auth", "token": "gate-token"})
            ready = ws.receive_json()
            assert ready["type"] == "session.ready"
            assert ready["demo"] is True
            assert "不是线上模型" in ready["demo_label"]


def test_pcm_frames_roundtrip_and_alignment():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    frame = encode_upstream_frame(7, 16000, pcm)
    seq, decoded = decode_upstream_frame(frame, expected_rate=16000)
    assert seq == 7 and decoded == pcm
    down = encode_tts_frame(
        turn_id="turn_abc",
        segment_index=1,
        chunk_index=2,
        sample_rate=22050,
        channels=1,
        pcm=pcm,
        flags=1,
    )
    parsed = decode_tts_frame(down)
    assert parsed["sample_rate"] == 22050
    assert parsed["turn_id"] == "turn_abc"
    assert parsed["pcm"] == pcm
    aligner = PcmByteAligner()
    assert aligner.feed(b"\x01") == b""
    assert aligner.feed(b"\x02\x03") == b"\x01\x02"
    assert aligner.flush() == b""
