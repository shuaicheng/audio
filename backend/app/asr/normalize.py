from __future__ import annotations

import re

_ASR_TEXT = re.compile(r"<asr_text>\s*(.*?)(?=(?:language\s+\S+<asr_text>)|$)", re.IGNORECASE | re.DOTALL)
_LANG_PREFIX = re.compile(r"^\s*language\s+[A-Za-z_-]+\s*", re.IGNORECASE)


def visible_transcript(raw: str) -> str:
    """去掉 Qwen 实时输出里的 language/<asr_text> 包装。没有包装时原样返回。"""

    if "<asr_text>" not in raw.lower():
        return _LANG_PREFIX.sub("", raw).strip()
    parts = [match.group(1).strip() for match in _ASR_TEXT.finditer(raw)]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    return "".join(parts).strip()


def merge_hypothesis(previous: str, delta: str) -> str:
    """vLLM 的 transcription.delta 是增量。若服务把全文又放进 delta，则改为替换。"""

    if not delta:
        return previous
    if not previous:
        return delta
    if delta.startswith(previous):
        return delta
    return previous + delta
