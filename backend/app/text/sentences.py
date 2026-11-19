from __future__ import annotations

import time
from collections.abc import Callable

STRONG = "。！？!?；;\n"
WEAK = "，,、：:"


def _suffix_keep(text: str, token: str) -> int:
    low = text.lower()
    tok = token.lower()
    upper = min(len(text), len(tok) - 1)
    for size in range(upper, 0, -1):
        if low.endswith(tok[:size]):
            return size
    return 0


class SentenceChunker:
    """把可见回答切成短句。push 立即返回，超时由 poll 处理，不阻塞调用方。"""

    def __init__(
        self,
        max_chars: int = 42,
        min_chars: int = 6,
        flush_timeout: float = 0.7,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.flush_timeout = flush_timeout
        self.clock = clock or time.monotonic
        self.mode = "text"
        self.tail = ""
        self.pending = ""
        self.pending_since: float | None = None
        self.at_line_start = True
        self.skip_hashes = False
        self.label = ""
        self.drop_label = False

    def push(self, text: str) -> list[str]:
        if text:
            self.tail += text
            self._consume()
        return self._drain(final=False)

    def poll(self, now: float | None = None) -> list[str]:
        if not self.pending.strip() or self.pending_since is None:
            return []
        current = self.clock() if now is None else now
        if current - self.pending_since < self.flush_timeout:
            return []
        piece = self.pending.strip()
        self.pending = ""
        self.pending_since = None
        return [piece] if piece else []

    def flush(self) -> list[str]:
        if self.mode == "text" and self.tail:
            keep = max(_suffix_keep(self.tail, "```"), _suffix_keep(self.tail, "<think>"))
            self._append(self.tail[:-keep] if keep else self.tail)
        if self.mode == "link_after" and not self.drop_label:
            self._append(self.label)
        self.mode = "text"
        self.tail = ""
        self.label = ""
        self.drop_label = False
        return self._drain(final=True)

    def _consume(self) -> None:
        while self.tail:
            if self.mode == "fence":
                found = self.tail.find("```")
                if found < 0:
                    self.tail = self.tail[-2:]
                    return
                self.tail = self.tail[found + 3 :]
                self.mode = "text"
                continue
            if self.mode == "inline":
                found = self.tail.find("`")
                if found < 0:
                    self.tail = ""
                    return
                self.tail = self.tail[found + 1 :]
                self.mode = "text"
                continue
            if self.mode == "think":
                lower = self.tail.lower()
                found = lower.find("</think>")
                if found < 0:
                    keep = _suffix_keep(self.tail, "</think>")
                    self.tail = self.tail[-keep:] if keep else ""
                    return
                self.tail = self.tail[found + 8 :]
                self.mode = "text"
                continue
            if self.mode in {"link_label", "image_label"}:
                found = self.tail.find("]")
                if found < 0:
                    if "\n" in self.tail:
                        if not self.drop_label:
                            self._append(self.label + self.tail)
                        self.label = ""
                        self.tail = ""
                        self.mode = "text"
                        self.drop_label = False
                        continue
                    self.label += self.tail
                    self.tail = ""
                    return
                self.label += self.tail[:found]
                self.tail = self.tail[found + 1 :]
                if self.tail.startswith("("):
                    self.tail = self.tail[1:]
                    self.mode = "link_url"
                    continue
                if self.tail == "":
                    self.mode = "link_after"
                    return
                if not self.drop_label:
                    self._append(self.label)
                self.label = ""
                self.drop_label = False
                self.mode = "text"
                continue
            if self.mode == "link_after":
                if self.tail.startswith("("):
                    self.tail = self.tail[1:]
                    self.mode = "link_url"
                    continue
                if not self.drop_label:
                    self._append(self.label)
                self.label = ""
                self.drop_label = False
                self.mode = "text"
                continue
            if self.mode == "link_url":
                found = self.tail.find(")")
                if found < 0:
                    self.tail = ""
                    return
                self.tail = self.tail[found + 1 :]
                if not self.drop_label:
                    self._append(self.label)
                self.label = ""
                self.drop_label = False
                self.mode = "text"
                continue

            fence_at = self.tail.find("```")
            think_at = self.tail.lower().find("<think>")
            tick_at = self.tail.find("`")
            image_at = self.tail.find("![")
            link_at = self.tail.find("[")
            candidates: list[tuple[int, str]] = []
            if fence_at >= 0:
                candidates.append((fence_at, "fence"))
            if think_at >= 0:
                candidates.append((think_at, "think"))
            if image_at >= 0:
                candidates.append((image_at, "image"))
            if link_at >= 0:
                candidates.append((link_at, "link"))
            if tick_at >= 0 and (fence_at < 0 or tick_at < fence_at):
                if self.tail[tick_at : tick_at + 3] == "```":
                    pass
                elif self.tail[tick_at:] in {"`", "``"} and tick_at + 3 > len(self.tail):
                    self._append(self.tail[:tick_at])
                    self.tail = self.tail[tick_at:]
                    return
                else:
                    candidates.append((tick_at, "inline"))
            if not candidates:
                keep = max(
                    _suffix_keep(self.tail, "```"),
                    _suffix_keep(self.tail, "<think>"),
                    _suffix_keep(self.tail, "</think>"),
                    1 if self.tail.endswith("[") else 0,
                )
                emit = self.tail[:-keep] if keep else self.tail
                self._append(emit)
                self.tail = self.tail[-keep:] if keep else ""
                return
            at, kind = min(candidates, key=lambda item: item[0])
            self._append(self.tail[:at])
            self.tail = self.tail[at:]
            if kind == "fence":
                self.tail = self.tail[3:]
                self.mode = "fence"
            elif kind == "think":
                self.tail = self.tail[7:]
                self.mode = "think"
            elif kind == "inline":
                self.tail = self.tail[1:]
                self.mode = "inline"
            elif kind == "image":
                self.tail = self.tail[2:]
                self.mode = "image_label"
                self.label = ""
                self.drop_label = True
            elif kind == "link":
                self.tail = self.tail[1:]
                self.mode = "link_label"
                self.label = ""
                self.drop_label = False

    def _append(self, text: str) -> None:
        if not text:
            return
        for char in text:
            if char in "*_":
                continue
            if self.at_line_start and char == "#":
                self.skip_hashes = True
                continue
            if self.skip_hashes:
                if char == "#":
                    continue
                self.skip_hashes = False
                if char == " ":
                    self.at_line_start = False
                    continue
            if char == "\n":
                self.at_line_start = True
            elif not char.isspace():
                self.at_line_start = False
            self.pending += char
        self._mark()

    def _mark(self) -> None:
        if self.pending.strip():
            self.pending_since = self.clock()
        else:
            self.pending_since = None

    def _drain(self, final: bool) -> list[str]:
        pieces: list[str] = []
        while self.pending:
            strong = _first_index(self.pending, STRONG)
            if strong is not None:
                piece = self.pending[: strong + 1].strip()
                self.pending = self.pending[strong + 1 :].lstrip()
                if piece:
                    pieces.append(piece)
                continue
            if len(self.pending) >= self.max_chars:
                window = self.pending[: self.max_chars]
                weak = _last_index(window, WEAK + " ")
                cut = weak + 1 if weak is not None and weak >= max(1, self.min_chars // 2) else self.max_chars
                piece = self.pending[:cut].strip()
                self.pending = self.pending[cut:].lstrip()
                if piece:
                    pieces.append(piece)
                continue
            weak = _first_index(self.pending, WEAK)
            if weak is not None and weak + 1 >= self.min_chars:
                piece = self.pending[: weak + 1].strip()
                self.pending = self.pending[weak + 1 :].lstrip()
                if piece:
                    pieces.append(piece)
                continue
            break
        if final:
            piece = self.pending.strip()
            self.pending = ""
            if piece:
                pieces.append(piece)
        self._mark()
        return pieces


def _first_index(text: str, alphabet: str) -> int | None:
    indexes = [text.find(char) for char in alphabet if char in text]
    if not indexes:
        return None
    return min(indexes)


def _last_index(text: str, alphabet: str) -> int | None:
    indexes = [text.rfind(char) for char in alphabet if char in text]
    if not indexes:
        return None
    return max(indexes)


class ThinkStripper:
    """从展示文本中去掉思维链标签，不改动其余正文。"""

    def __init__(self) -> None:
        self.mode = "text"
        self.buf = ""

    def feed(self, data: str) -> str:
        self.buf += data
        output: list[str] = []
        while self.buf:
            if self.mode == "text":
                lower = self.buf.lower()
                found = lower.find("<think>")
                if found >= 0:
                    output.append(self.buf[:found])
                    self.buf = self.buf[found + 7 :]
                    self.mode = "think"
                    continue
                keep = _suffix_keep(self.buf, "<think>")
                output.append(self.buf[:-keep] if keep else self.buf)
                self.buf = self.buf[-keep:] if keep else ""
                break
            lower = self.buf.lower()
            found = lower.find("</think>")
            if found >= 0:
                self.buf = self.buf[found + 8 :]
                self.mode = "text"
                continue
            keep = _suffix_keep(self.buf, "</think>")
            self.buf = self.buf[-keep:] if keep else ""
            break
        return "".join(output)

    def flush(self) -> str:
        if self.mode != "text":
            self.buf = ""
            self.mode = "text"
            return ""
        leftover = self.buf
        self.buf = ""
        return leftover
