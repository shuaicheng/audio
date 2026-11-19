from __future__ import annotations

from app.endpoint import pcm_rms


class AsrForwardGate:
    """开口前的静音照常送出。确认出声之后，只再送很短的尾音。

    更长的静音仍交给本地端点判断要不要收句，但不再送给识别服务。
    识别模型若在静音上顺着前文往下写，就没有这段音频可以写。
    """

    def __init__(self, sample_rate: int, release_rms: float, tail_ms: int) -> None:
        self.sample_rate = sample_rate
        self.release_rms = release_rms
        self.tail_samples = max(0, int(sample_rate * tail_ms / 1000))
        self.heard_voice = False
        self.silence_samples = 0
        self.forwarded_samples = 0
        self.held_samples = 0
        self.voice_since_text = False
        self.grew_without_voice = False
        self.last_text = ""

    def wants(self, pcm: bytes) -> bool:
        samples = len(pcm) // 2
        if samples <= 0:
            return False
        if pcm_rms(pcm) >= self.release_rms:
            return True
        if not self.heard_voice or self.silence_samples < self.tail_samples:
            return True
        return False

    def commit(self, pcm: bytes, forwarded: bool) -> None:
        samples = len(pcm) // 2
        if samples <= 0:
            return
        if not forwarded:
            self.held_samples += samples
            return
        if pcm_rms(pcm) >= self.release_rms:
            self.heard_voice = True
            self.silence_samples = 0
            self.voice_since_text = True
        else:
            self.silence_samples += samples
        self.forwarded_samples += samples

    def note_text(self, text: str) -> None:
        voiced = self.voice_since_text
        self.voice_since_text = False
        previous = self.last_text
        self.last_text = text
        if previous and text != previous and not voiced:
            self.grew_without_voice = True
