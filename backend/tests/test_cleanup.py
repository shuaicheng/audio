from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from app.asr.factory import build_asr
from app.asr.qwen_streaming import QwenStreamingAsr, QwenWorkerPool
from app.orchestrator import Orchestrator
from tests.fakes import RecordingTts, ScriptedAsr


@pytest.mark.asyncio
async def test_disconnect_closes_asr_and_cancels_turn(settings):
    asr = ScriptedAsr()

    class HungLlm:
        def __init__(self):
            self.cancelled = False

        async def stream(self, messages, cancelled):
            try:
                yield "开始"
                await asyncio.sleep(30)
                yield "之后"
            finally:
                self.cancelled = cancelled.is_set()

    llm = HungLlm()
    orch = Orchestrator(settings, lambda: asr, llm, RecordingTts())
    await orch.submit_text("问")
    for _ in range(20):
        try:
            kind, payload = await asyncio.wait_for(orch.outgoing.get(), 0.1)
        except TimeoutError:
            continue
        if kind == "json" and payload.get("type") == "llm.delta":
            break
    await orch.close()
    assert orch.closed is True
    assert orch.turn is None
    assert llm.cancelled is True
    pending = [task for task in orch._tasks if not task.done()]
    assert pending == []
    assistant = orch.history[-1]
    assert assistant["status"] == "interrupted"
    assert "回答已中断" in assistant["content"]


@pytest.mark.asyncio
async def test_asr_timeout_and_worker_cleanup(settings, tmp_path: Path):
    asr = ScriptedAsr()
    orch = Orchestrator(settings, lambda: asr, _NoLlm(), RecordingTts())
    settings.asr_timeout_sec = 0.4
    await orch.start_speech()
    await asr.finish()
    await orch.finish_speech()
    found = False
    for _ in range(30):
        try:
            kind, payload = await asyncio.wait_for(orch.outgoing.get(), 0.2)
        except TimeoutError:
            continue
        if kind == "json" and payload.get("type") == "timeout" and payload.get("scope") == "asr":
            found = True
            break
    assert found
    await orch.close()
    assert asr.closed is True

    log = tmp_path / "calls.txt"
    stub = tmp_path / "qwen_asr"
    stub.mkdir()
    (stub / "__init__.py").write_text(
        "import os\n"
        "class Qwen3ASRModel:\n"
        "    @classmethod\n"
        "    def LLM(cls, **kwargs):\n"
        "        open(os.environ['QWEN_ASR_STUB_LOG'], 'a', encoding='utf-8').write('LLM ' + __import__('json').dumps(kwargs, ensure_ascii=False) + '\\n')\n"
        "        print('banner-from-model')\n"
        "        return cls()\n"
        "    def init_streaming_state(self, **kwargs):\n"
        "        open(os.environ['QWEN_ASR_STUB_LOG'], 'a', encoding='utf-8').write('init\\n')\n"
        "        return {'text': '', 'language': kwargs.get('language') or '', 'n': 0}\n"
        "    def streaming_transcribe(self, pcm, state):\n"
        "        state['n'] += len(pcm)\n"
        "        state['text'] = '今天' if state['n'] < 20 else '今天天气不错'\n"
        "        open(os.environ['QWEN_ASR_STUB_LOG'], 'a', encoding='utf-8').write('step ' + state['text'] + '\\n')\n"
        "    def finish_streaming_transcribe(self, state):\n"
        "        if state['text'] and not state['text'].endswith('。'):\n"
        "            state['text'] += '。'\n"
        "        open(os.environ['QWEN_ASR_STUB_LOG'], 'a', encoding='utf-8').write('finish ' + state['text'] + '\\n')\n",
        encoding="utf-8",
    )
    settings.qwen_asr_load_timeout_sec = 5
    settings.qwen_asr_model = "Qwen/Qwen3-ASR-1.7B"
    settings.qwen_asr_max_new_tokens = 32
    pool = QwenWorkerPool(settings)
    pool_env_pythonpath = str(tmp_path)
    # The pool sets PYTHONPATH to the backend dir; prepend the stub by wrapping python.
    wrapper = tmp_path / "python-stub.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"export PYTHONPATH='{tmp_path}:{Path(__file__).resolve().parents[1]}'\n"
        f"export QWEN_ASR_STUB_LOG='{log}'\n"
        f"export QWEN_ASR_ALLOW_LIST_AUDIO=1\n"
        f"exec {sys.executable} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    settings.qwen_asr_worker_python = str(wrapper)
    first = QwenStreamingAsr(settings, pool)
    second = QwenStreamingAsr(settings, pool)
    await first.start()
    await second.start()
    assert (await first.next_event(2)).kind == "ready"
    assert (await second.next_event(2)).kind == "ready"
    await first.push_audio(b"\x01\x00" * 10, 1)
    left = await first.next_event(2)
    await second.push_audio(b"\x01\x00" * 2000, 1)
    right = await second.next_event(2)
    await first.push_audio(b"\x01\x00" * 2000, 2)
    revised = await first.next_event(2)
    assert left.text == "今天"
    assert revised.text == "今天天气不错"
    assert right.text == "今天天气不错"
    assert revised.text != "今天今天天气不错"
    await first.finish()
    final = await first.next_event(2)
    assert final.kind == "final" and final.text == "今天天气不错。"
    assert final.timestamps is False
    await first.close()
    await second.close()
    proc = pool.proc
    await pool.shutdown()
    assert proc is not None
    assert proc.returncode is not None
    text = log.read_text(encoding="utf-8")
    assert "init" in text and "finish" in text
    assert "Qwen/Qwen3-ASR-1.7B" in text
    assert "banner-from-model" not in text
    assert pool_env_pythonpath
    settings.asr_provider = "qwen_streaming"
    settings.app_mode = "live"
    assert build_asr(settings, pool).mode == "qwen_streaming"


class _NoLlm:
    async def stream(self, messages, cancelled):
        if False:
            yield ""
