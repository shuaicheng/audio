from __future__ import annotations

import inspect
import time

from app.text.sentences import SentenceChunker, ThinkStripper


def chunker(**kwargs) -> SentenceChunker:
    clock = {"now": 0.0}
    return SentenceChunker(
        max_chars=kwargs.get("max_chars", 20),
        min_chars=kwargs.get("min_chars", 6),
        flush_timeout=kwargs.get("flush_timeout", 0.5),
        clock=lambda: clock["now"],
    ), clock


def test_punctuation_length_timeout_and_flush_do_not_block():
    segmenter, clock = chunker(max_chars=10, min_chars=6, flush_timeout=0.5)
    assert inspect.iscoroutinefunction(segmenter.push) is False
    started = time.perf_counter()
    assert segmenter.push("一二三四五六七八九，十")[0].endswith("，")
    assert time.perf_counter() - started < 0.05
    hard = SentenceChunker(max_chars=10, min_chars=6, flush_timeout=5, clock=lambda: 0)
    pieces = hard.push("一二三四五六七八九十一二三四五")
    assert "".join(pieces) == "一二三四五六七八九十一二三四五"[: sum(map(len, pieces))]
    assert all(len(piece) <= 10 for piece in pieces)
    assert hard.flush() == [] or len("".join(pieces + hard.flush())) >= 10

    idle, clock = chunker()
    assert idle.push("今天天气真好") == []
    clock["now"] = 0.49
    assert idle.poll(0.49) == []
    clock["now"] = 0.5
    assert idle.poll(0.5) == ["今天天气真好"]
    fresh, clock = chunker()
    assert fresh.push("今天") == []
    clock["now"] = 0.4
    assert fresh.push("天气") == []
    assert fresh.poll(0.5) == []
    assert fresh.flush() == ["今天天气"]


def test_markdown_code_reasoning_are_not_spoken():
    segmenter = SentenceChunker(max_chars=80, min_chars=2, flush_timeout=10, clock=lambda: 0)
    spoken = []
    spoken += segmenter.push("可见<thi")
    spoken += segmenter.push("nk>秘密</think>文本。")
    spoken += segmenter.push("开头。```python\nprint(1)\n```结尾。")
    spoken += segmenter.push("这是**很重要**的话。")
    spoken += segmenter.push("请看[链接文字](http://example.test)。")
    spoken += segmenter.push("![图](http://example.test/a.png)正文。")
    spoken += segmenter.push("# 标题\n正文。")
    spoken += segmenter.flush()
    merged = "".join(spoken)
    assert "秘密" not in merged
    assert "print" not in merged
    assert "**" not in merged
    assert "http" not in merged
    assert "可见文本。" in merged
    assert "开头。" in merged
    assert "结尾。" in merged
    assert "这是很重要的话。" in merged
    assert "链接文字。" in merged
    assert "正文。" in merged
    assert "标题" in merged


def test_think_stripper_holds_split_tag():
    stripper = ThinkStripper()
    assert stripper.feed("答案<thi") == "答案"
    assert stripper.feed("nk>隐藏</think>继续") == "继续"
    assert "隐藏" not in stripper.feed("。")
