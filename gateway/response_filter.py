"""Shared gateway response filtering helpers."""

from __future__ import annotations

import re
from typing import Any


_GATEWAY_SILENT_RESPONSE_RE = re.compile(
    r"^\s*(?:`{1,3}\s*)?[\[\(]?\s*(?:NO_RESPONSE|SILENT)\s*[\]\)]?\s*(?:`{1,3})?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def is_gateway_silent_response(text: Any) -> bool:
    """Return True when model output is only an internal no-send sentinel.

    Group-chat agents can use a sentinel like ``[NO_RESPONSE]`` to mean
    "intentionally stay silent". The marker is control flow, not user-facing
    content, so platform delivery paths must suppress it if it reaches them.
    """
    if text is None:
        return False
    body = str(text).strip()
    if not body:
        return False
    return bool(_GATEWAY_SILENT_RESPONSE_RE.fullmatch(body))
