"""官方 Qwen3-ASR vLLM 流式 worker。

每个 stream_id 有独立的 streaming state，命令按 stdin 顺序执行。
stdout 只写协议 JSON；模型库的打印被转到 stderr。

真实路径调用：
  Qwen3ASRModel.LLM(...)
  init_streaming_state(...)
  streaming_transcribe(pcm16k, state)  # 读 state.text，全文替换
  finish_streaming_transcribe(state)

没有词级时间戳。依赖未安装时返回 dependency_missing，而不是假装识别成功。
参考：https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


PROTO = None


def _bind_protocol_stdout():
    global PROTO
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1, closefd=True)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    PROTO = proto
    return proto


def emit(payload: dict) -> None:
    if PROTO is None:
        raise RuntimeError("protocol stdout is not bound")
    PROTO.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTO.flush()


class _FakeStreamingModel:
    """仅 QWEN_ASR_WORKER_FAKE=1 时使用，用来测试修订替换，不是产品演示。"""

    def init_streaming_state(self, **kwargs: Any) -> dict:
        return {"text": "", "language": kwargs.get("language") or "Chinese", "samples": 0}

    def streaming_transcribe(self, pcm: Any, state: dict) -> dict:
        state["samples"] += _sample_count(pcm)
        state["text"] = "今天" if state["samples"] < 3200 else "今天天气不错"
        state["language"] = state.get("language") or "Chinese"
        return state

    def finish_streaming_transcribe(self, state: dict) -> dict:
        if state.get("text") and not str(state["text"]).endswith("。"):
            state["text"] = str(state["text"]) + "。"
        return state


def _sample_count(pcm: Any) -> int:
    if isinstance(pcm, (bytes, bytearray)):
        return len(pcm) // 2
    return len(pcm)


def load_model() -> Any:
    if os.environ.get("QWEN_ASR_WORKER_FAKE") == "1":
        return _FakeStreamingModel()
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.LLM(
        model=os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        gpu_memory_utilization=float(os.environ.get("QWEN_ASR_GPU_MEMORY_UTILIZATION", "0.8")),
        max_new_tokens=int(os.environ.get("QWEN_ASR_MAX_NEW_TOKENS", "32")),
    )


def pcm16_to_model_audio(pcm: bytes) -> Any:
    if os.environ.get("QWEN_ASR_WORKER_FAKE") == "1":
        return pcm
    if len(pcm) % 2:
        pcm = pcm[:-1]
    try:
        import numpy as np
    except ImportError:
        if os.environ.get("QWEN_ASR_ALLOW_LIST_AUDIO") != "1":
            raise
        import array

        ints = array.array("h")
        ints.frombytes(pcm)
        return [sample / 32768.0 for sample in ints]
    return np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0


def _state_text(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("text") or "")
    return str(getattr(state, "text", "") or "")


def _state_language(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("language") or "")
    return str(getattr(state, "language", "") or "")


def main() -> int:
    _bind_protocol_stdout()
    model: Any = None
    streams: dict[str, dict] = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            emit({"op": "error", "code": "bad_json", "message": "worker 收到无法解析的命令"})
            continue
        op = command.get("op")
        try:
            if op == "hello":
                try:
                    model = load_model()
                except ImportError as exc:
                    emit(
                        {
                            "op": "error",
                            "code": "dependency_missing",
                            "message": f"未安装 qwen-asr，请在独立环境安装 requirements-qwen-asr.txt：{exc}",
                        }
                    )
                    return 2
                emit(
                    {
                        "op": "ready",
                        "engine": "qwen-asr",
                        "streaming": True,
                        "timestamps": False,
                        "fake": os.environ.get("QWEN_ASR_WORKER_FAKE") == "1",
                    }
                )
                continue
            if op == "shutdown":
                emit({"op": "bye"})
                return 0
            if model is None:
                emit({"op": "error", "code": "not_ready", "message": "worker 尚未完成 hello"})
                continue
            stream_id = str(command.get("stream_id") or "")
            if op == "open":
                kwargs: dict[str, Any] = {
                    "unfixed_chunk_num": int(command.get("unfixed_chunk_num", 2)),
                    "unfixed_token_num": int(command.get("unfixed_token_num", 5)),
                    "chunk_size_sec": float(command.get("chunk_size_sec", 2.0)),
                }
                if command.get("language"):
                    kwargs["language"] = command["language"]
                if command.get("context"):
                    kwargs["context"] = command["context"]
                state = model.init_streaming_state(**kwargs)
                streams[stream_id] = {"state": state, "last_seq": 0, "finished": False}
                emit({"op": "opened", "stream_id": stream_id, "timestamps": False})
                continue
            stream = streams.get(stream_id)
            if stream is None:
                emit({"op": "error", "stream_id": stream_id, "code": "missing_stream", "message": "流式会话不存在"})
                continue
            if op == "audio":
                seq = int(command.get("seq") or 0)
                if seq <= stream["last_seq"]:
                    emit({"op": "error", "stream_id": stream_id, "code": "out_of_order", "message": "音频序号倒退"})
                    continue
                import base64

                pcm = base64.b64decode(command.get("pcm16_b64") or "")
                audio = pcm16_to_model_audio(pcm)
                model.streaming_transcribe(audio, stream["state"])
                stream["last_seq"] = seq
                emit(
                    {
                        "op": "partial",
                        "stream_id": stream_id,
                        "seq": seq,
                        "text": _state_text(stream["state"]),
                        "language": _state_language(stream["state"]),
                        "timestamps": False,
                    }
                )
                continue
            if op == "finish":
                model.finish_streaming_transcribe(stream["state"])
                stream["finished"] = True
                emit(
                    {
                        "op": "final",
                        "stream_id": stream_id,
                        "text": _state_text(stream["state"]),
                        "language": _state_language(stream["state"]),
                        "timestamps": False,
                    }
                )
                continue
            if op == "close":
                streams.pop(stream_id, None)
                emit({"op": "closed", "stream_id": stream_id})
                continue
            emit({"op": "error", "code": "unknown_op", "message": f"未知命令 {op}"})
        except Exception as exc:
            emit({"op": "error", "stream_id": command.get("stream_id"), "code": "worker_error", "message": str(exc)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
    except Exception as exc:
        emit({"op": "error", "code": "dependency_missing" if "qwen_asr" in str(exc) else "worker_crash", "message": str(exc)})
        raise SystemExit(2)
