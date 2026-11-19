from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AsrEvent:
    kind: str
    text: str = ""
    code: str = ""
    message: str = ""
    degraded: bool = False
    strategy: str = ""
    language: str = ""
    validated: bool = False
    timestamps: bool = False
    extra: dict = field(default_factory=dict)


class AsrEngine(Protocol):
    mode: str
    degraded: bool

    async def start(self) -> None: ...

    async def push_audio(self, pcm: bytes, seq: int) -> None: ...

    async def finish(self) -> None: ...

    async def cancel(self) -> None: ...

    async def close(self) -> None: ...

    def drain(self) -> list[AsrEvent]: ...
