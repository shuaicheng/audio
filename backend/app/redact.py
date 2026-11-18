from __future__ import annotations

import re


_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-+/=]{6,}")
_KEY_PARAM = re.compile(r"(?i)(api[_-]?key|token|cookie|authorization)(=|:)\s*\S+")


def scrub(text: str, secrets: list[str] | None = None, limit: int = 240) -> str:
    cleaned = text or ""
    for secret in secrets or []:
        if secret:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    cleaned = _BEARER.sub(r"\1[REDACTED]", cleaned)
    cleaned = _KEY_PARAM.sub(r"\1\2[REDACTED]", cleaned)
    cleaned = cleaned.replace("\n", " ")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + "…"
    return cleaned
