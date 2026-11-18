from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from typing import Any

from app.asr.base import AsrEvent
from app.config import Settings

logger = logging.getLogger(__name__)


class QwenStreamingAsr:
    """把每段音频交给官方 qwen-asr 流式 worker。

    worker 内部按 stream_id 保存 init_streaming_state 的状态。
    partial 事件里的 text 是 state.text 全文，调用方必须替换而不是追加。
    """

    mode = "qwen_streaming"
    degraded = False

    def __init__(self, settings: Settings, pool: "QwenWorkerPool | None" = None) -> None:
        self.settings = settings
        self.pool = pool
        self._stream_id = ""
        self._events: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self._next_seq = 1
        self._last_text = None
        self._closed = False
        self.accepted = False

    async def start(self) -> None:
        if self.pool is None:
            await self._events.put(AsrEvent(kind="failed", code="worker_unconfigured", message="没有可用的 Qwen 流式 worker"))
            return
        try:
            self._stream_id = await self.pool.open_stream(self)
        except Exception as exc:
            await self._events.put(AsrEvent(kind="failed", code="worker_unavailable", message=str(exc)))
            return
        self.accepted = True

    async def push_audio(self, pcm: bytes, seq: int) -> None:
        if not self._stream_id or self._closed:
            return
        if seq != self._next_seq:
            await self._events.put(AsrEvent(kind="failed", code="audio_sequence", message=f"音频序号不连续，期望 {self._next_seq}"))
            return
        self._next_seq = seq + 1
        await self.pool.submit_audio(self._stream_id, seq, pcm)

    async def finish(self) -> None:
        if self._stream_id and self.pool and not self._closed:
            await self.pool.finish_stream(self._stream_id)

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream_id and self.pool:
            await self.pool.close_stream(self._stream_id)
        await self._events.put(None)

    def drain(self) -> list[AsrEvent]:
        items: list[AsrEvent] = []
        while True:
            try:
                item = self._events.get_nowait()
            except asyncio.QueueEmpty:
                return items
            if item is not None:
                items.append(item)

    async def next_event(self, timeout: float | None = None) -> AsrEvent | None:
        if timeout is None:
            return await self._events.get()
        try:
            return await asyncio.wait_for(self._events.get(), timeout)
        except TimeoutError:
            return None

    async def handle_worker_message(self, message: dict) -> None:
        op = message.get("op")
        if op == "opened":
            await self._events.put(AsrEvent(kind="ready", validated=True, timestamps=False, message="Qwen 流式状态已建立"))
            return
        if op == "partial":
            text = str(message.get("text") or "")
            if text != self._last_text:
                self._last_text = text
                await self._events.put(AsrEvent(kind="partial", text=text, language=str(message.get("language") or ""), timestamps=False))
            return
        if op == "final":
            text = str(message.get("text") or "")
            self._last_text = text
            await self._events.put(AsrEvent(kind="final", text=text, language=str(message.get("language") or ""), timestamps=False))
            return
        if op == "error":
            await self._events.put(
                AsrEvent(
                    kind="failed",
                    code=str(message.get("code") or "worker_error"),
                    message=str(message.get("message") or "Qwen 流式 worker 失败"),
                )
            )


class QwenWorkerPool:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.proc: asyncio.subprocess.Process | None = None
        self._write_lock = asyncio.Lock()
        self._streams: dict[str, QwenStreamingAsr] = {}
        self._reader: asyncio.Task | None = None
        self._stderr: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._start_error = ""
        self._counter = 0
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self.proc and self.proc.returncode is None and self._ready.is_set() and not self._start_error:
                return
            python = self.settings.qwen_asr_worker_python or sys.executable
            env = os.environ.copy()
            backend_dir = str(__import__("pathlib").Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")
            env.update(
                {
                    "QWEN_ASR_MODEL": self.settings.qwen_asr_model,
                    "QWEN_ASR_GPU_MEMORY_UTILIZATION": str(self.settings.qwen_asr_gpu_memory_utilization),
                    "QWEN_ASR_MAX_NEW_TOKENS": str(self.settings.qwen_asr_max_new_tokens),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            self._ready = asyncio.Event()
            self._start_error = ""
            self.proc = await asyncio.create_subprocess_exec(
                python,
                "-m",
                "app.asr.qwen_streaming_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=backend_dir,
            )
            self._reader = asyncio.create_task(self._read_stdout())
            self._stderr = asyncio.create_task(self._drain_stderr())
            await self._send({"op": "hello"})
            try:
                await asyncio.wait_for(self._ready.wait(), self.settings.qwen_asr_load_timeout_sec)
            except TimeoutError as exc:
                await self.shutdown()
                raise RuntimeError("Qwen 流式 worker 启动超时") from exc
            if self._start_error:
                message = self._start_error
                await self.shutdown()
                raise RuntimeError(message)

    async def open_stream(self, engine: QwenStreamingAsr) -> str:
        await self.start()
        self._counter += 1
        stream_id = f"qs_{self._counter}"
        self._streams[stream_id] = engine
        await self._send(
            {
                "op": "open",
                "stream_id": stream_id,
                "language": self.settings.qwen_asr_language or None,
                "context": self.settings.qwen_asr_context or None,
                "unfixed_chunk_num": self.settings.qwen_asr_unfixed_chunk_num,
                "unfixed_token_num": self.settings.qwen_asr_unfixed_token_num,
                "chunk_size_sec": self.settings.qwen_asr_chunk_size_sec,
            }
        )
        return stream_id

    async def submit_audio(self, stream_id: str, seq: int, pcm: bytes) -> None:
        await self._send(
            {
                "op": "audio",
                "stream_id": stream_id,
                "seq": seq,
                "pcm16_b64": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def finish_stream(self, stream_id: str) -> None:
        await self._send({"op": "finish", "stream_id": stream_id})

    async def close_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        if self.proc and self.proc.returncode is None:
            await self._send({"op": "close", "stream_id": stream_id})

    async def shutdown(self) -> None:
        if self.proc and self.proc.returncode is None and self.proc.stdin:
            try:
                await self._send({"op": "shutdown"})
                await asyncio.wait_for(self.proc.wait(), 2)
            except Exception:
                self.proc.kill()
                await self.proc.wait()
        for task in (self._reader, self._stderr):
            if task and not task.done():
                task.cancel()
        self.proc = None
        self._ready.clear()

    async def _send(self, payload: dict) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Qwen 流式 worker 没有运行")
        async with self._write_lock:
            data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.proc.stdin.write(data)
            await self.proc.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                logger.info("忽略 worker 非协议输出，长度 %s", len(line))
                continue
            await self._route(message)

    async def _route(self, message: dict) -> None:
        op = message.get("op")
        if op == "ready":
            self._ready.set()
            return
        if op == "error" and not message.get("stream_id"):
            self._start_error = str(message.get("message") or message.get("code") or "worker 启动失败")
            self._ready.set()
            return
        stream_id = str(message.get("stream_id") or "")
        engine = self._streams.get(stream_id)
        if engine is not None:
            await engine.handle_worker_message(message)

    async def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.info("qwen worker: %s", text[:300])
