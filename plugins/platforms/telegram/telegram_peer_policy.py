"""Telegram peer admission and per-turn addressing policy.

Adapted from the local telegram-peer-silence patch.
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def mentions_only_peer_suppressed(adapter: "TelegramAdapter", message: "Message") -> bool:
    """Drop unaddressed mentions-only peers and known automated warning loops."""
    if not bool(getattr(getattr(message, "from_user", None), "is_bot", False)):
        return False
    if adapter._is_own_message(message):
        return False
    # These are gateway diagnostics, not fresh handoff requests. Restrict to
    # bot-origin warning prefixes; humans can still quote/report the errors.
    for text in (getattr(message, "text", None), getattr(message, "caption", None)):
        text = (adapter._clean_bot_trigger_text(text) or "").strip()
        if text.startswith((
            "⚠️ The model returned a silence marker",
            "⚠️ The model returned no response after processing tool results.",
            "⚠️ Processing completed but no response was generated.",
        )):
            return True
    mode = os.getenv("TELEGRAM_ALLOW_BOTS") or adapter.config.extra.get("allow_bots", "none")
    return str(mode).lower().strip() == "mentions" and not adapter._message_mentions_bot(message)


def reply_expected(adapter: "TelegramAdapter", message: "Message") -> bool | None:
    """Admission alone is not a request; peer replies need a current mention."""
    if not adapter._is_group_chat(message):
        return True
    if not getattr(adapter, "_bot", None):
        return None
    if adapter._message_mentions_bot(message):
        return True
    if bool(getattr(getattr(message, "from_user", None), "is_bot", False)):
        return False
    return adapter._is_reply_to_bot(message) or adapter._message_matches_mention_patterns(message)
