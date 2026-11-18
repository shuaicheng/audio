from __future__ import annotations

import array
import math


class UtteranceEndpoint:
    """听到人声之后，连续静音才结束这一句。开头的静音不会触发。"""

    def __init__(
        self,
        sample_rate: int,
        speech_rms: float = 0.018,
        silence_ms: int = 900,
        min_speech_ms: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.speech_rms = speech_rms
        self.silence_samples = max(1, int(sample_rate * silence_ms / 1000))
        self.min_speech_samples = max(1, int(sample_rate * min_speech_ms / 1000))
        self.speech_samples = 0
        self.silence_run = 0
        self.heard_speech = False
        self.fired = False

    def feed(self, pcm: bytes) -> bool:
        if self.fired or len(pcm) < 2:
            return False
        usable = pcm[: len(pcm) - (len(pcm) % 2)]
        samples = array.array("h")
        samples.frombytes(usable)
        if not samples:
            return False
        energy = 0
        for value in samples:
            energy += value * value
        rms = math.sqrt(energy / len(samples)) / 32768.0
        if rms >= self.speech_rms:
            self.speech_samples += len(samples)
            self.silence_run = 0
            if self.speech_samples >= self.min_speech_samples:
                self.heard_speech = True
        elif self.heard_speech:
            self.silence_run += len(samples)
        if self.heard_speech and self.silence_run >= self.silence_samples:
            self.fired = True
            return True
        return False
