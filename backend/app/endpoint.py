from __future__ import annotations

import array
import math
import time


class UtteranceEndpoint:
    """开口前一直等待。只有连续、够久的人声才算说过话，之后才计算句末停顿。

    时间用墙钟，不用一口气塞进来的采样长度。启动瞬态或单次噪声如果没有持续到
    onset，会把候选清掉，后面的静音也不会结束录音。
    """

    def __init__(
        self,
        sample_rate: int,
        onset_rms: float = 0.05,
        release_rms: float = 0.02,
        onset_ms: int = 480,
        silence_ms: int = 1400,
        clock=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.onset_rms = onset_rms
        self.release_rms = release_rms
        self.onset_seconds = onset_ms / 1000
        self.silence_seconds = silence_ms / 1000
        self.clock = clock or time.monotonic
        self.stage = "waiting"
        self.fired = False
        self._onset_at: float | None = None
        self._silence_at: float | None = None

    def feed(self, pcm: bytes, now: float | None = None) -> bool:
        if self.fired or len(pcm) < 2:
            return False
        moment = self.clock() if now is None else now
        rms = _rms(pcm)
        if self.stage == "waiting":
            if rms >= self.onset_rms:
                if self._onset_at is None:
                    self._onset_at = moment
                elif moment - self._onset_at + 0.001 >= self.onset_seconds:
                    self.stage = "speech"
                    self._silence_at = None
            else:
                self._onset_at = None
            return False
        if rms >= self.release_rms:
            self._silence_at = None
            if self.stage != "speech":
                self.stage = "speech"
            return False
        if self._silence_at is None:
            self._silence_at = moment
            self.stage = "pause"
            return False
        if moment - self._silence_at + 0.001 >= self.silence_seconds:
            self.fired = True
            return True
        self.stage = "pause"
        return False


def _rms(pcm: bytes) -> float:
    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    samples = array.array("h")
    samples.frombytes(usable)
    if not samples:
        return 0.0
    energy = 0
    for value in samples:
        energy += value * value
    return math.sqrt(energy / len(samples)) / 32768.0
