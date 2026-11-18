from __future__ import annotations

import struct


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    if len(pcm) % (2 * channels) != 0:
        pcm = pcm[: len(pcm) - (len(pcm) % (2 * channels))]
    data_size = len(pcm)
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header + pcm
