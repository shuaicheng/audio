from __future__ import annotations

import struct


class MalformedAudio(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def decode_upstream_frame(frame: bytes, *, expected_rate: int) -> tuple[int, bytes]:
    if len(frame) < 16 or frame[:4] != b"AUPC":
        raise MalformedAudio("无法识别的音频帧")
    version, _flags, seq, rate, channels = struct.unpack_from("<BBIIH", frame, 4)
    if version != 1:
        raise MalformedAudio("不支持的音频帧版本")
    if channels != 1:
        raise MalformedAudio("需要单声道 16 位音频")
    if rate != expected_rate:
        raise MalformedAudio(f"采样率应为 {expected_rate} Hz")
    pcm = frame[16:]
    if not pcm:
        raise MalformedAudio("空音频帧")
    if len(pcm) % 2 != 0:
        raise MalformedAudio("PCM 没有按 16 位采样对齐")
    return seq, pcm


def encode_upstream_frame(seq: int, sample_rate: int, pcm: bytes, channels: int = 1) -> bytes:
    if len(pcm) % (2 * channels) != 0:
        raise MalformedAudio("PCM 没有按采样帧对齐")
    header = struct.pack("<4sBBIIH", b"AUPC", 1, 0, seq, sample_rate, channels)
    return header + pcm


def encode_tts_frame(
    *,
    turn_id: str,
    segment_index: int,
    chunk_index: int,
    sample_rate: int,
    channels: int,
    pcm: bytes,
    flags: int = 0,
) -> bytes:
    token = turn_id.encode("utf-8")
    if len(token) > 255:
        raise ValueError("turn id 过长")
    frame_bytes = 2 * channels
    if frame_bytes <= 0 or len(pcm) % frame_bytes != 0:
        raise ValueError("下行 PCM 没有按采样帧对齐")
    header = b"".join(
        [
            b"ADNA",
            struct.pack("<BBB", 1, flags, len(token)),
            token,
            struct.pack("<HIIH", segment_index, chunk_index, sample_rate, channels),
        ]
    )
    return header + pcm


def decode_tts_frame(frame: bytes) -> dict:
    if len(frame) < 19 or frame[:4] != b"ADNA":
        raise MalformedAudio("无法识别的下行音频帧")
    version, flags, id_len = struct.unpack_from("<BBB", frame, 4)
    if version != 1:
        raise MalformedAudio("不支持的下行音频版本")
    start = 7
    end = start + id_len
    if len(frame) < end + 12:
        raise MalformedAudio("下行音频帧过短")
    turn_id = frame[start:end].decode("utf-8")
    segment_index, chunk_index, sample_rate, channels = struct.unpack_from("<HIIH", frame, end)
    pcm = frame[end + 12 :]
    if channels <= 0 or len(pcm) % (2 * channels) != 0:
        raise MalformedAudio("下行 PCM 没有按采样帧对齐")
    return {
        "turn_id": turn_id,
        "flags": flags,
        "segment_index": segment_index,
        "chunk_index": chunk_index,
        "sample_rate": sample_rate,
        "channels": channels,
        "pcm": pcm,
    }


class PcmByteAligner:
    """把任意切分的 PCM16 字节流收成完整采样帧。"""

    def __init__(self, channels: int = 1) -> None:
        self.frame_bytes = 2 * channels
        self._hold = b""

    def feed(self, data: bytes) -> bytes:
        blob = self._hold + data
        usable = len(blob) - (len(blob) % self.frame_bytes)
        self._hold = blob[usable:]
        return blob[:usable]

    def flush(self) -> bytes:
        self._hold = b""
        return b""
