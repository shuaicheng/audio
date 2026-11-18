"""按墙钟速度向 vLLM Realtime 送多秒音频，并记录 finish 之前有没有 partial。

这是验收脚本，不是文件分块冒充。默认会先用合成服务生成一句短语音，再重采样到 16 kHz。
连不上或上游只在 commit final 之后才出字时，脚本会明确失败，不会把那种结果说成真流式。
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import sys
import time
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _tts_pcm(url: str, text: str) -> tuple[bytes, int]:
    body = {
        "model": "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": text,
        "voice": "vivian",
        "language": "Chinese",
        "stream": True,
        "response_format": "pcm",
    }
    chunks: list[bytes] = []
    rate = 24000
    with httpx.Client(timeout=20) as client:
        with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                rate = int(response.headers.get("x-audio-sample-rate") or response.headers.get("x-sample-rate") or 24000)
                return response.read(), rate
            buffer = ""
            for piece in response.iter_text():
                buffer += piece
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    data = ""
                    event_name = ""
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                        elif line.startswith("data:"):
                            data += line[5:].strip()
                    if not data or event_name.endswith("done"):
                        continue
                    payload = json.loads(data)
                    if payload.get("sample_rate"):
                        rate = int(payload["sample_rate"])
                    audio = payload.get("audio")
                    if isinstance(audio, str) and audio:
                        chunks.append(base64.b64decode(audio))
    return b"".join(chunks), rate


def _to_16k(pcm: bytes, rate: int) -> bytes:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if rate == 16000:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
    return converted


async def probe(url: str, model: str, pcm16: bytes, chunk_ms: int) -> dict:
    chunk_bytes = max(2, int(16000 * chunk_ms / 1000) * 2)
    events: list[dict] = []
    started = time.monotonic()
    finish_at: float | None = None
    async with websockets.connect(url, open_timeout=8) as ws:
        created = json.loads(await asyncio.wait_for(ws.recv(), 8))
        if created.get("type") != "session.created":
            return {"ok": False, "reason": "missing session.created", "event": created.get("type")}
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

        async def reader() -> None:
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    events.append({"at": round(time.monotonic() - started, 3), "event": event.get("type"), "text": event.get("delta") or event.get("text") or ""})
                    if event.get("type") in {"transcription.done", "error"} and finish_at is not None:
                        return
            except Exception:
                return

        task = asyncio.create_task(reader())
        for offset in range(0, len(pcm16), chunk_bytes):
            piece = pcm16[offset : offset + chunk_bytes]
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(piece).decode("ascii")}))
            await asyncio.sleep(chunk_ms / 1000)
        await asyncio.sleep(0.4)
        before = [item for item in events if item["event"] in {"transcription.delta", "transcription.done"}]
        finish_at = time.monotonic() - started
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        try:
            await asyncio.wait_for(task, 12)
        except TimeoutError:
            task.cancel()
        after = [item for item in events if item["event"] in {"transcription.delta", "transcription.done"}]
    partial_before = [item for item in before if item["event"] == "transcription.delta" and str(item["text"]).strip()]
    return {
        "ok": True,
        "finish_at": round(finish_at, 3),
        "partial_before_finish": len(partial_before),
        "partial_texts_before_finish": [item["text"] for item in partial_before][:6],
        "events": after[-8:],
        "streaming": bool(partial_before),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-url", default="ws://120.48.180.39:10002/v1/realtime")
    parser.add_argument("--tts-url", default="http://120.48.180.39:10001/v1/audio/speech")
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--pcm", default="")
    args = parser.parse_args()
    if args.pcm:
        pcm16 = Path(args.pcm).read_bytes()
        rate = 16000
    else:
        try:
            pcm, rate = _tts_pcm(args.tts_url, args.text)
        except Exception as exc:
            print(json.dumps({"ok": False, "reason": "tts_unreachable", "detail": type(exc).__name__}, ensure_ascii=False))
            raise SystemExit(1)
        pcm16 = _to_16k(pcm, rate)
    if len(pcm16) < 16000:
        print(json.dumps({"ok": False, "reason": "audio_too_short", "bytes": len(pcm16)}, ensure_ascii=False))
        raise SystemExit(1)
    result = asyncio.run(probe(args.asr_url, args.model, pcm16, args.chunk_ms))
    print(json.dumps(result, ensure_ascii=False))
    if not result.get("ok"):
        raise SystemExit(1)
    if not result.get("streaming"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
