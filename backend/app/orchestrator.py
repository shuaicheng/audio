from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.config import Settings
from app.endpoint import UtteranceEndpoint
from app.ids import new_id
from app.llm import LlmError
from app.playback_gate import PlaybackGate
from app.protocol import MalformedAudio, PcmByteAligner, decode_upstream_frame, encode_tts_frame
from app.redact import scrub
from app.text.sentences import SentenceChunker, ThinkStripper
from app.tts import TtsError

logger = logging.getLogger(__name__)

Outgoing = tuple[str, dict | bytes]


@dataclass
class Turn:
    turn_id: str
    utterance_id: str
    source: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    visible: str = ""
    llm_done: bool = False
    history_recorded: bool = False
    task: asyncio.Task | None = None
    chunker: SentenceChunker | None = None
    stripper: ThinkStripper = field(default_factory=ThinkStripper)


class Orchestrator:
    def __init__(self, settings: Settings, asr_factory: Callable, llm, tts) -> None:
        self.settings = settings
        self.asr_factory = asr_factory
        self.llm = llm
        self.tts = tts
        self.session_id = new_id("ses")
        self.outgoing: asyncio.Queue[Outgoing] = asyncio.Queue(maxsize=max(8, settings.outgoing_queue_chunks))
        self.history: list[dict] = []
        self.phase = "ready"
        self.asr = None
        self.listen_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None
        self.audio_q: asyncio.Queue | None = None
        self.turn: Turn | None = None
        self.utterance_id = ""
        self.utterance_bytes = 0
        self.auto_submit = True
        self.gate = PlaybackGate(settings.playback_max_inflight_bytes)
        self.closed = False
        self._tasks: set[asyncio.Task] = set()

    async def greet(self) -> None:
        demo = self.settings.app_mode == "mock"
        provider = "mock" if demo else self.settings.provider_name()
        degraded = provider == "degraded_file"
        if demo:
            note = "当前是离线演示，不是线上模型结果。交付时没有对部署地址做过真实调用。"
        elif provider == "realtime":
            note = "说话时就会显示识别。停顿或点「说完了」后才回答，播报时不会自动抢话。"
        elif provider == "qwen_streaming":
            note = "使用官方 qwen-asr 流式接口。它没有词级时间戳。本仓库不把文件重传当作流式。"
        else:
            note = "文件转写是显式降级，不是真流式识别。重复上传整段时会标明 cumulative_file_reupload。"
        await self.emit(
            {
                "type": "session.ready",
                "demo": demo,
                "demo_label": "离线演示模式：识别、回答和声音都是本地模拟，不是线上模型的真实结果。" if demo else "",
                "duplex": "push_to_talk",
                "asr": {
                    "provider": provider,
                    "degraded": degraded,
                    "sample_rate": self.settings.asr_sample_rate,
                    "timestamps": False,
                    "validated_live": False,
                    "note": note,
                },
                "tts": {"sample_rate": self.settings.tts_sample_rate, "encoding": "pcm16le"},
            }
        )
        await self.emit_state()

    async def handle_text(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await self.emit({"type": "error", "code": "malformed_json", "message": "无法解析控制消息"})
            return
        if not isinstance(message, dict):
            await self.emit({"type": "error", "code": "malformed_json", "message": "控制消息格式不正确"})
            return
        kind = message.get("type")
        if kind == "speech.start":
            if "auto_submit" in message:
                self.auto_submit = message.get("auto_submit") is not False
            await self.start_speech()
        elif kind == "speech.finish":
            await self.finish_speech(auto_submit=message.get("auto_submit", True) is not False)
        elif kind == "speech.cancel":
            await self.cancel_speech()
        elif kind == "text.submit":
            await self.submit_text(str(message.get("text") or ""))
        elif kind == "answer.stop":
            await self.stop_answer("user_stop")
        elif kind == "conversation.clear":
            await self.clear()
        elif kind == "playback.hold":
            await self.gate.set_hold(bool(message.get("hold")))
        elif kind == "playback.ack":
            await self.gate.release(int(message.get("bytes") or 0))
        else:
            await self.emit({"type": "error", "code": "unknown_message", "message": "无法识别的控制消息"})

    async def handle_audio(self, frame: bytes) -> None:
        try:
            seq, pcm = decode_upstream_frame(frame, expected_rate=self.settings.asr_sample_rate)
        except MalformedAudio as exc:
            await self.emit({"type": "audio.malformed", "code": "malformed_audio", "message": exc.message})
            return
        if self.phase != "listening" or self.audio_q is None:
            await self.emit({"type": "audio.malformed", "code": "not_listening", "message": "现在没有在收音，请先开始说话"})
            return
        if self.utterance_bytes + len(pcm) > self.settings.max_utterance_bytes:
            await self.emit(
                {
                    "type": "queue.limit",
                    "code": "utterance_too_large",
                    "scope": "asr",
                    "message": "这段话太长，已停止继续接收音频",
                }
            )
            return
        if self.audio_q.full():
            await self.emit(
                {
                    "type": "queue.limit",
                    "code": "audio_queue_full",
                    "scope": "asr",
                    "message": "音频缓冲已满，请稍后再说",
                }
            )
            return
        self.utterance_bytes += len(pcm)
        self.audio_q.put_nowait((seq, pcm))
        if self.settings.endpoint_enabled and self.phase == "listening":
            should_finish = self.endpoint.feed(pcm)
            await self._emit_listen_stage()
            if should_finish and not self._auto_finish_started:
                self._auto_finish_started = True
                self._track(asyncio.create_task(self.finish_speech(auto_submit=self.auto_submit)))

    async def start_speech(self) -> None:
        if self.closed:
            return
        if self.turn and not self.turn.cancelled:
            await self.emit({"type": "error", "code": "busy_answering", "message": "正在回答。请先停止回答，再开始说话"})
            return
        if self.phase == "listening":
            return
        self.asr = self.asr_factory()
        self.utterance_id = new_id("utt")
        self.utterance_bytes = 0
        self._auto_finish_started = False
        self._listen_stage = ""
        self.endpoint = UtteranceEndpoint(
            self.settings.asr_sample_rate,
            onset_rms=self.settings.endpoint_onset_rms,
            release_rms=self.settings.endpoint_release_rms,
            onset_ms=self.settings.endpoint_onset_ms,
            silence_ms=self.settings.endpoint_silence_ms,
        )
        self.audio_q = asyncio.Queue(maxsize=max(1, self.settings.max_audio_pending_chunks))
        await self.asr.start()
        if not getattr(self.asr, "accepted", False):
            self.phase = "ready"
            self.listen_task = self._track(asyncio.create_task(self._pump_asr()))
            return
        self.phase = "listening"
        self.audio_task = self._track(asyncio.create_task(self._pump_audio()))
        self.listen_task = self._track(asyncio.create_task(self._pump_asr()))
        await self.emit(
            {
                "type": "speech.started",
                "utterance_id": self.utterance_id,
                "asr_mode": getattr(self.asr, "mode", ""),
                "degraded": bool(getattr(self.asr, "degraded", False)),
                "timestamps": False,
            }
        )
        await self.emit_state()
        await self._emit_listen_stage()

    async def _emit_listen_stage(self) -> None:
        stage = getattr(self.endpoint, "stage", "")
        if not stage or stage == self._listen_stage:
            return
        self._listen_stage = stage
        await self.emit(
            {
                "type": "listen.activity",
                "stage": stage,
                "utterance_id": self.utterance_id,
            }
        )

    async def finish_speech(self, auto_submit: bool = True) -> None:
        if self.phase != "listening":
            return
        self.auto_submit = auto_submit
        self.phase = "recognizing"
        await self.emit_state()
        if self.audio_q is not None:
            await self.audio_q.put(None)
        if self.audio_task is not None and self.audio_task is not asyncio.current_task():
            with contextlib.suppress(asyncio.CancelledError):
                await self.audio_task
        if self.asr is not None:
            await self.asr.finish()

    async def cancel_speech(self) -> None:
        if self.phase not in {"listening", "recognizing"} and self.asr is None:
            return
        self.phase = "ready"
        if self.audio_q is not None:
            await self.audio_q.put(None)
        if self.audio_task is not None:
            self.audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.audio_task
        if self.asr is not None:
            await self.asr.cancel()
            self.asr = None
        if self.listen_task is not None:
            self.listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.listen_task
        await self.emit_state()

    async def submit_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            await self.emit({"type": "error", "code": "empty_text", "message": "没有可发送的文字"})
            return
        if self.phase in {"listening", "recognizing"}:
            await self.cancel_speech()
        if self.turn and not self.turn.cancelled:
            await self.stop_answer("replaced")
        await self._start_turn(cleaned, source="text", utterance_id=new_id("utt"))

    async def stop_answer(self, reason: str) -> None:
        turn = self.turn
        if turn is None or turn.cancelled:
            return
        turn.cancelled = True
        turn.cancel_event.set()
        self.turn = None
        await self.gate.reset()
        self._record_assistant(turn, "interrupted")
        if turn.task and turn.task is not asyncio.current_task():
            turn.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn.task
        await self.emit({"type": "turn.cancelled", "turn_id": turn.turn_id, "reason": reason})
        await self.emit({"type": "playback.stop", "turn_id": turn.turn_id})
        if self.phase == "answering":
            self.phase = "ready"
        await self.emit_state()

    async def clear(self) -> None:
        if self.phase in {"listening", "recognizing"}:
            await self.cancel_speech()
        if self.turn and not self.turn.cancelled:
            await self.stop_answer("cleared")
        self.history.clear()
        await self.emit({"type": "conversation.cleared"})
        self.phase = "ready"
        await self.emit_state()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.phase in {"listening", "recognizing"}:
            await self.cancel_speech()
        if self.turn and not self.turn.cancelled:
            await self.stop_answer("disconnect")
        if self.asr is not None:
            await self.asr.close()
            self.asr = None
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def _pump_audio(self) -> None:
        assert self.audio_q is not None and self.asr is not None
        while True:
            item = await self.audio_q.get()
            if item is None or self.asr is None:
                break
            seq, pcm = item
            await self.asr.push_audio(pcm, seq)

    async def _pump_asr(self) -> None:
        assert self.asr is not None
        waited = 0.0
        revision = 0
        try:
            while not self.closed:
                try:
                    event = await self.asr.next_event(0.2)
                except TimeoutError:
                    if self.phase == "recognizing":
                        waited += 0.2
                        if waited >= self.settings.asr_timeout_sec:
                            await self.emit(
                                {
                                    "type": "timeout",
                                    "scope": "asr",
                                    "code": "asr_timeout",
                                    "message": "识别超时，没有在限定时间内得到最终文本",
                                    "utterance_id": self.utterance_id,
                                }
                            )
                            self.phase = "ready"
                            await self.emit_state()
                            break
                    continue
                if event is None:
                    break
                waited = 0.0
                degraded = bool(event.degraded)
                if event.kind == "ready":
                    await self.emit(
                        {
                            "type": "asr.ready",
                            "utterance_id": self.utterance_id,
                            "degraded": degraded,
                            "validated": event.validated,
                            "strategy": event.strategy,
                            "message": event.message,
                            "timestamps": False,
                        }
                    )
                elif event.kind == "partial":
                    revision += 1
                    await self.emit(
                        {
                            "type": "asr.partial",
                            "utterance_id": self.utterance_id,
                            "text": event.text,
                            "revision": revision,
                            "degraded": degraded,
                            "strategy": event.strategy,
                            "timestamps": False,
                        }
                    )
                elif event.kind == "final":
                    text = event.text.strip()
                    await self.emit(
                        {
                            "type": "asr.final",
                            "utterance_id": self.utterance_id,
                            "text": event.text,
                            "degraded": degraded,
                            "strategy": event.strategy,
                            "timestamps": False,
                            "awaiting_edit": bool(text) and not self.auto_submit,
                        }
                    )
                    if text and self.auto_submit:
                        await self._start_turn(text, source="speech", utterance_id=self.utterance_id)
                    elif text:
                        self.phase = "ready"
                        await self.emit_state()
                    else:
                        self.phase = "ready"
                        await self.emit({"type": "error", "code": "empty_transcript", "message": "没有识别到内容"})
                        await self.emit_state()
                    break
                elif event.kind == "unsupported":
                    await self.emit(
                        {
                            "type": "asr.unsupported",
                            "code": event.code or "unsupported",
                            "message": event.message or "当前识别服务不接受该模型或协议",
                            "utterance_id": self.utterance_id,
                            "validated": False,
                        }
                    )
                    self.phase = "ready"
                    await self.emit_state()
                    break
                elif event.kind == "failed":
                    await self.emit(
                        {
                            "type": "asr.failed",
                            "code": event.code or "asr_failed",
                            "message": event.message or "识别失败",
                            "utterance_id": self.utterance_id,
                            "degraded": degraded,
                        }
                    )
                    self.phase = "ready"
                    await self.emit_state()
                    break
        finally:
            asr = self.asr
            if asr is not None and self.phase != "listening":
                self.asr = None
                with contextlib.suppress(Exception):
                    await asr.close()

    async def _start_turn(self, text: str, source: str, utterance_id: str) -> None:
        if self.turn and not self.turn.cancelled:
            await self.stop_answer("replaced")
        turn = Turn(turn_id=new_id("turn"), utterance_id=utterance_id, source=source)
        self.turn = turn
        self.history.append(
            {"role": "user", "content": text, "status": "complete", "source": source, "utterance_id": utterance_id}
        )
        self._trim_history()
        self.phase = "answering"
        await self.emit(
            {
                "type": "turn.started",
                "turn_id": turn.turn_id,
                "utterance_id": utterance_id,
                "source": source,
                "text": text,
            }
        )
        await self.emit_state()
        turn.task = self._track(asyncio.create_task(self._run_turn(turn)))

    async def _run_turn(self, turn: Turn) -> None:
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        turn.chunker = SentenceChunker(
            max_chars=self.settings.sentence_max_chars,
            min_chars=self.settings.sentence_min_chars,
            flush_timeout=self.settings.sentence_flush_timeout_sec,
        )
        speaker = asyncio.create_task(self._speak(turn, sentences))
        poller = asyncio.create_task(self._poll_sentences(turn, sentences))
        status = "complete"
        try:
            try:
                stream = self.llm.stream(self._llm_messages(), turn.cancel_event).__aiter__()
                try:
                    first = await asyncio.wait_for(stream.__anext__(), self.settings.llm_first_token_timeout_sec)
                except StopAsyncIteration:
                    first = None
                except TimeoutError:
                    status = "error"
                    await self.emit_for(
                        turn,
                        {
                            "type": "timeout",
                            "scope": "llm",
                            "code": "llm_timeout",
                            "message": "语言模型迟迟没有开始回答",
                            "partial_text": turn.visible,
                        },
                    )
                    return
                if not first:
                    status = "error"
                    await self.emit_for(
                        turn,
                        {
                            "type": "llm.error",
                            "code": "empty_visible",
                            "message": "模型没有返回可朗读的正文",
                            "partial_text": "",
                        },
                    )
                    return
                await self._take_delta(turn, sentences, first)
                async for delta in stream:
                    if turn.cancelled or turn.cancel_event.is_set():
                        status = "cancelled"
                        return
                    await self._take_delta(turn, sentences, delta)
                tail = turn.stripper.flush()
                if tail and not turn.cancelled:
                    await self._take_delta(turn, sentences, tail)
                if turn.chunker and not turn.cancelled:
                    for piece in turn.chunker.flush():
                        sentences.put_nowait(piece)
                if turn.cancelled:
                    status = "cancelled"
                    return
                await self.emit_for(turn, {"type": "llm.done", "text": turn.visible})
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except LlmError as exc:
                status = "error"
                await self.emit_for(
                    turn,
                    {
                        "type": "llm.error",
                        "code": exc.code,
                        "message": exc.message,
                        "partial_text": turn.visible,
                    },
                )
            except Exception as exc:
                status = "error"
                logger.info("语言模型流中断：%s", type(exc).__name__)
                await self.emit_for(
                    turn,
                    {
                        "type": "llm.error",
                        "code": "llm_stream",
                        "message": "语言模型中断：" + scrub(str(exc), [self.settings.audio_llm_api_key]),
                        "partial_text": turn.visible,
                    },
                )
        finally:
            turn.llm_done = True
            with contextlib.suppress(Exception):
                sentences.put_nowait(None)
            poller.cancel()
            if status == "cancelled":
                speaker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await speaker
            with contextlib.suppress(asyncio.CancelledError):
                await poller
            if status == "error":
                self._record_assistant(turn, "error")
            elif status == "complete" and not turn.cancelled:
                self._record_assistant(turn, "complete")
            if self.turn is turn:
                self.turn = None
                self.phase = "ready"
                await self.emit_state()

    async def _take_delta(self, turn: Turn, sentences: asyncio.Queue, delta: str) -> None:
        visible = turn.stripper.feed(delta)
        if not visible or turn.cancelled:
            return
        turn.visible += visible
        await self.emit_for(turn, {"type": "llm.delta", "delta": visible, "text": turn.visible})
        if turn.chunker is None:
            return
        for piece in turn.chunker.push(visible):
            sentences.put_nowait(piece)

    async def _poll_sentences(self, turn: Turn, sentences: asyncio.Queue) -> None:
        try:
            while not turn.cancelled and not turn.llm_done:
                if turn.chunker is not None:
                    for piece in turn.chunker.poll():
                        sentences.put_nowait(piece)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return

    async def _speak(self, turn: Turn, sentences: asyncio.Queue) -> None:
        failed = False
        segment_index = 0
        try:
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                if failed or turn.cancelled:
                    continue
                await self.emit_for(
                    turn,
                    {
                        "type": "speech.segment",
                        "segment_index": segment_index,
                        "text": sentence,
                    },
                )
                aligner = PcmByteAligner()
                chunk_index = 0
                try:
                    async for pcm, sample_rate in self.tts.stream(sentence, turn.cancel_event):
                        if turn.cancelled:
                            return
                        aligned = aligner.feed(pcm)
                        if not aligned:
                            continue
                        try:
                            allowed = await asyncio.wait_for(
                                self.gate.reserve(len(aligned), turn.cancel_event),
                                self.settings.playback_stall_timeout_sec,
                            )
                        except TimeoutError:
                            failed = True
                            await self.emit_for(
                                turn,
                                {
                                    "type": "queue.limit",
                                    "code": "playback_stalled",
                                    "scope": "tts",
                                    "message": "播放端接收过慢，已停止继续合成",
                                },
                            )
                            break
                        if not allowed or turn.cancelled:
                            return
                        frame = encode_tts_frame(
                            turn_id=turn.turn_id,
                            segment_index=segment_index,
                            chunk_index=chunk_index,
                            sample_rate=sample_rate,
                            channels=1,
                            pcm=aligned,
                            flags=0,
                        )
                        await self.emit_audio(frame, turn)
                        chunk_index += 1
                    aligner.flush()
                except TtsError as exc:
                    failed = True
                    await self.emit_for(
                        turn,
                        {
                            "type": "tts.failed",
                            "code": exc.code,
                            "message": exc.message,
                            "segment_index": segment_index,
                        },
                    )
                except Exception as exc:
                    failed = True
                    logger.info("语音合成中断：%s", type(exc).__name__)
                    await self.emit_for(
                        turn,
                        {
                            "type": "tts.failed",
                            "code": "tts_failed",
                            "message": "语音合成失败，文字回答仍会保留",
                            "segment_index": segment_index,
                        },
                    )
                else:
                    if not failed and not turn.cancelled:
                        await self.emit_for(
                            turn,
                            {"type": "tts.segment.done", "segment_index": segment_index, "chunks": chunk_index},
                        )
                segment_index += 1
                if failed:
                    break
            if not failed and not turn.cancelled:
                await self.emit_for(turn, {"type": "tts.done"})
        except asyncio.CancelledError:
            raise

    def _llm_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.settings.llm_system_prompt}]
        for item in self.history[-max(2, self.settings.history_max_turns * 2) :]:
            messages.append({"role": item["role"], "content": item["content"]})
        return messages

    def _record_assistant(self, turn: Turn, status: str) -> None:
        if turn.history_recorded:
            return
        turn.history_recorded = True
        content = turn.visible
        if status == "interrupted":
            content = (content + "\n（回答已中断，未说完）").strip()
        elif status == "error":
            content = (content + "\n（回答因错误中断）").strip()
        self.history.append({"role": "assistant", "content": content, "status": status, "turn_id": turn.turn_id})
        self._trim_history()

    def _trim_history(self) -> None:
        limit = max(2, self.settings.history_max_turns * 2)
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    async def emit(self, event: dict) -> None:
        event.setdefault("session_id", self.session_id)
        await self.outgoing.put(("json", event))

    async def emit_for(self, turn: Turn, event: dict) -> None:
        if turn.cancelled or self.turn is not turn:
            return
        event["turn_id"] = turn.turn_id
        await self.emit(event)

    async def emit_audio(self, frame: bytes, turn: Turn) -> None:
        if turn.cancelled or self.turn is not turn:
            return
        await self.outgoing.put(("bytes", frame))

    async def emit_state(self) -> None:
        labels = {
            "ready": "待命",
            "listening": "正在聆听",
            "recognizing": "正在识别",
            "answering": "正在回答",
        }
        await self.emit({"type": "state", "phase": self.phase, "label": labels.get(self.phase, self.phase)})

    def _track(self, task: asyncio.Task) -> asyncio.Task:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
