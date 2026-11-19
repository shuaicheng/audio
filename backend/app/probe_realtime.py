"""手动检查 vLLM Realtime 是否接受配置的模型名。

只发送 session.update，不发送音频，也不发送会启动转写的 commit。
本仓库的测试和启动流程都不会调用它。线上是否支持 Qwen3-ASR 仍以实时握手结果为准。
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets

from app.config import Settings


async def probe(settings: Settings) -> dict:
    headers = {}
    if settings.asr_authorization:
        value = settings.asr_authorization
        headers["Authorization"] = value if value.lower().startswith("bearer ") else f"Bearer {value}"
    async with websockets.connect(
        settings.asr_realtime_url,
        additional_headers=headers or None,
        open_timeout=settings.asr_handshake_timeout_sec,
    ) as ws:
        raw = await asyncio.wait_for(ws.recv(), settings.asr_handshake_timeout_sec)
        first = json.loads(raw)
        if first.get("type") != "session.created":
            return {"ok": False, "reason": "missing session.created", "event": first.get("type")}
        await ws.send(json.dumps({"type": "session.update", "model": settings.asr_model}))
        try:
            raw = await asyncio.wait_for(ws.recv(), settings.asr_error_probe_sec)
        except TimeoutError:
            return {
                "ok": False,
                "validated_live": False,
                "reason": "session.update 没有立即拒绝，也没有正向回执。未发送音频，不能视为识别可用。",
            }
        event = json.loads(raw)
        return {"ok": False, "validated_live": False, "reason": "session.update 收到响应", "event_type": event.get("type")}


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 realtime 模型握手（不上传音频）")
    parser.add_argument("--confirm-live", action="store_true", help="确实要连接 ASR_REALTIME_URL")
    args = parser.parse_args()
    if not args.confirm_live:
        raise SystemExit("拒绝连接。只有人工确认后才能加 --confirm-live。测试不会调用此命令。")
    result = asyncio.run(probe(Settings()))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
