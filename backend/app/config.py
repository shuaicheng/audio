from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """服务端配置。密钥只从环境变量读取，repr 不包含密钥正文。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_mode: str = "live"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    app_auth_token: str = ""

    asr_provider: str = "realtime"
    asr_realtime_url: str = "ws://120.48.180.39:10002/v1/realtime"
    asr_file_url: str = "http://120.48.180.39:10002/v1/audio/transcriptions"
    asr_model: str = "Qwen3-ASR-1.7B"
    asr_sample_rate: int = 16000
    asr_timeout_sec: float = 20
    asr_handshake_timeout_sec: float = 5
    asr_error_probe_sec: float = 0.4
    asr_authorization: str = ""
    asr_cookie: str = ""
    asr_degraded_chunk_sec: float = 0

    qwen_asr_worker_python: str = ""
    qwen_asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_asr_gpu_memory_utilization: float = 0.8
    qwen_asr_max_new_tokens: int = 32
    qwen_asr_unfixed_chunk_num: int = 2
    qwen_asr_unfixed_token_num: int = 5
    qwen_asr_chunk_size_sec: float = 2.0
    qwen_asr_language: str = ""
    qwen_asr_context: str = ""
    qwen_asr_load_timeout_sec: float = 600

    llm_url: str = "http://120.48.180.39:20002/v1/chat/completions"
    llm_model: str = "Qwen3.8-Flash"
    audio_llm_api_key: str = ""
    llm_timeout_sec: float = 90
    llm_first_token_timeout_sec: float = 25
    llm_temperature: float = 0.7
    llm_system_prompt: str = (
        "你是语音对话助手。用口语短句回答，适合直接朗读。"
        "不要使用 Markdown、列表、代码块或思维链。"
    )

    tts_url: str = "http://120.48.180.39:10001/v1/audio/speech"
    tts_model: str = "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_voice: str = "vivian"
    tts_language: str = "Chinese"
    tts_sample_rate: int = 24000
    tts_timeout_sec: float = 60
    tts_authorization: str = ""
    tts_cookie: str = ""

    sentence_max_chars: int = 42
    sentence_min_chars: int = 6
    sentence_flush_timeout_sec: float = 0.7

    max_audio_pending_chunks: int = 128
    max_utterance_bytes: int = 3_200_000
    playback_max_inflight_bytes: int = 96_000
    playback_stall_timeout_sec: float = 15
    outgoing_queue_chunks: int = 32
    history_max_turns: int = 16
    mock_token_delay_sec: float = 0.02
    endpoint_enabled: bool = True
    endpoint_silence_ms: int = 900
    endpoint_min_speech_ms: int = 300
    endpoint_speech_rms: float = 0.018

    secrets_fingerprint: str = Field(default="", exclude=True)

    def cors_origin_list(self) -> list[str]:
        items = [part.strip() for part in self.cors_origins.split(",") if part.strip()]
        return items or ["http://localhost:5173"]

    def provider_name(self) -> str:
        if self.asr_provider == "auto":
            return "realtime"
        return self.asr_provider

    def redacted_key_state(self) -> str:
        return "set" if self.audio_llm_api_key else "empty"

    def __repr__(self) -> str:
        return (
            f"Settings(app_mode={self.app_mode!r}, asr_provider={self.asr_provider!r}, "
            f"llm_model={self.llm_model!r}, audio_llm_api_key={self.redacted_key_state()})"
        )
