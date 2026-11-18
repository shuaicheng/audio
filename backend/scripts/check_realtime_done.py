"""本机检查：同一句只送一遍时，transcription.done 会不会把这句话再拼一次。

不保存音频，也不打印识别正文。需要本机能访问实时识别服务。

    ASR_REALTIME_URL=ws://127.0.0.1:10002/v1/realtime \\
      python backend/scripts/check_realtime_done.py --wav /path/to/once.wav

没有 wav 时，若本机有 espeak-ng，会临时合成一句中文并在进程退出时删掉。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import wave

import websockets

from app.asr.normalize import merge_hypothesis, resolve_completed_transcript, visible_transcript


def _load_pcm16_16k(path: str) -> bytes:
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise SystemExit("请提供 16 kHz、单声道、PCM16 的 wav")
        return handle.readframes(handle.getnframes())


def _synthesize(text: str) -> str:
    target = tempfile.NamedTemporaryFile(prefix="asr-once-", suffix=".wav", delete=False)
    target.close()
    raw = target.name + ".raw.wav"
    subprocess.run(["espeak-ng", "-v", "zh", text, "-w", raw], check=True, capture_output=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw, "-ar", "16000", "-ac", "1", target.name],
        check=True,
        capture_output=True,
    )
    os.unlink(raw)
    return target.name


async def _run(url: str, model: str, pcm: bytes) -> int:
    deltas: list[str] = []
    done_text = ""
    sequence: list[str] = []
    started = time.monotonic()
    async with websockets.connect(url, open_timeout=8) as ws:
        sequence.append(f"{time.monotonic() - started:.2f}s session.wait")
        created = json.loads(await asyncio.wait_for(ws.recv(), 8))
        sequence.append(f"{time.monotonic() - started:.2f}s {created.get('type')}")
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        for offset in range(0, len(pcm), 3200):
            piece = pcm[offset : offset + 3200]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(piece).decode("ascii"),
                    }
                )
            )
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        sequence.append(f"{time.monotonic() - started:.2f}s commit.final bytes={len(pcm)}")
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), 20)
            except TimeoutError:
                sequence.append("timeout waiting for done")
                break
            event = json.loads(raw)
            kind = str(event.get("type") or "")
            if kind == "transcription.delta":
                deltas.append(str(event.get("delta") or ""))
                sequence.append(f"{time.monotonic() - started:.2f}s delta chars={len(deltas[-1])}")
            elif kind == "transcription.done":
                done_text = str(event.get("text") or "")
                sequence.append(f"{time.monotonic() - started:.2f}s done chars={len(done_text)}")
                break
            else:
                sequence.append(f"{time.monotonic() - started:.2f}s {kind}")
    merged = ""
    for piece in deltas:
        merged = merge_hypothesis(merged, piece)
    committed = visible_transcript(resolve_completed_transcript(merged, deltas, done_text))
    joined = "".join(deltas)
    print("sequence:")
    for line in sequence:
        print(" ", line)
    print(f"delta_count={len(deltas)}")
    print(f"joined_chars={len(joined)} merged_chars={len(visible_transcript(merged))} done_chars={len(done_text)}")
    print(f"done_equals_raw_join={done_text == joined}")
    print(f"raw_join_repeats_merged={bool(merged) and done_text == joined and joined != merged}")
    print(f"committed_chars={len(committed)}")
    print(f"committed_equals_merged={committed == visible_transcript(merged)}")
    if done_text == joined and joined != merged:
        print("done 文本是各条 delta 的原文拼接，不等于合并稿。提交时用合并稿，避免说完后再接一遍。")
        return 0
    if not done_text and not merged:
        print("没有收到转写。请确认地址、模型和音频。")
        return 1
    print("done 文本可以当作这一句的完整稿。")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("ASR_REALTIME_URL", "ws://120.48.180.39:10002/v1/realtime"))
    parser.add_argument("--model", default=os.environ.get("ASR_MODEL", "Qwen3-ASR-1.7B"))
    parser.add_argument("--wav")
    parser.add_argument("--text", default="一二三四五六七八九")
    args = parser.parse_args()
    temporary = ""
    path = args.wav
    if not path:
        temporary = _synthesize(args.text)
        path = temporary
    try:
        pcm = _load_pcm16_16k(path)
        raise SystemExit(asyncio.run(_run(args.url, args.model, pcm)))
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
