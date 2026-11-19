from __future__ import annotations

import re

_ASR_TEXT = re.compile(r"<asr_text>\s*(.*?)(?=(?:language\s+\S+<asr_text>)|$)", re.IGNORECASE | re.DOTALL)
_LANG_PREFIX = re.compile(r"^\s*language\s+[A-Za-z_-]+\s*", re.IGNORECASE)


def visible_transcript(raw: str) -> str:
    """去掉 Qwen 实时输出里的 language/<asr_text> 包装。包装还没写完时不把前缀显示出来。"""

    lowered = raw.lower()
    if "<asr_text>" not in lowered:
        if re.match(r"^\s*language\b", raw, re.IGNORECASE):
            return ""
        return _LANG_PREFIX.sub("", raw).strip()
    parts = [match.group(1).strip() for match in _ASR_TEXT.finditer(raw)]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    return "".join(parts).strip()


def merge_hypothesis(previous: str, delta: str) -> str:
    """把一条 transcription.delta 并进当前假设。

    增量片段直接接上。若这条 delta 本身就是从开头重写的全文，则替换，不能再接一次。
    """

    if not delta:
        return previous
    if not previous:
        return delta
    if delta.startswith(previous):
        return delta
    return previous + delta


def resolve_completed_transcript(merged: str, raw_deltas: list[str], done_text: str) -> str:
    """决定 transcription.done 之后该提交哪一稿。

    vLLM 的 done.text 就是把每条 delta 原文顺序拼起来。delta 若是增量，拼起来等于已经合并的假设，
    done 就是这一句的完整稿。delta 若是一次次重写的全文，拼起来会把同一句再接一遍；
    那不是新的权威稿，提交时仍用合并后的假设。done 文本和原文拼接不同时，才用它替换。
    """

    joined = "".join(raw_deltas)
    if not done_text:
        return merged
    if done_text == joined and joined != merged:
        return merged
    return done_text
