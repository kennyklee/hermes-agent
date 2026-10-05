"""Telegram group addressing helpers, kept out of the adapter facade.

The identity line lives in ``channel_prompt`` and therefore in the cached-agent signature: it
must be stable for the life of a session (username only — never a per-message fact).
"""

from typing import TYPE_CHECKING, Optional
from gateway.platforms._shared import extra_or_secret

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def mentions_only_peer_suppressed(adapter: "TelegramAdapter", message: "Message") -> bool:
    """Mentions-only peers cannot wake via reply chains or leak status noise into context."""
    mode = extra_or_secret(adapter.config.extra, "allow_bots", "TELEGRAM_ALLOW_BOTS", "none")
    return (
        str(mode).lower().strip() == "mentions"
        and adapter._sender_is_other_bot(message)
        and not adapter._message_mentions_bot(message)
    )


def mentions_other_participants(adapter: "TelegramAdapter", message: "Message") -> bool:
    """True when a ``mention``/``text_mention`` entity names someone other than this bot."""
    own = adapter._current_bot_username()
    bot_id = getattr(adapter._bot, "id", None) if adapter._bot else None
    for source_text, entities in adapter._entity_sources(message):
        for entity in entities:
            entity_type = adapter._entity_type(entity)
            if entity_type == "mention":
                handle = (adapter._entity_span(source_text, entity) or "").strip().lstrip("@").lower()
                if handle and handle != own:
                    return True
            elif entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and getattr(user, "id", None) != bot_id:
                    return True
    return False


def reply_expected(adapter: "TelegramAdapter", message: "Message") -> Optional[bool]:
    """Admission is not an obligation to answer ambient group chatter.

    Peers must address us in the current message; human replies and configured
    wake words remain directed requests. DMs retain the visible failure fallback.
    """
    if not adapter._is_group_chat(message):
        return True
    if not getattr(adapter, "_bot", None):
        return None  # No identity: retain the gateway's unknown-address fallback.
    if adapter._message_mentions_bot(message):
        return True
    if adapter._sender_is_other_bot(message):
        return False
    return adapter._is_reply_to_bot(message) or adapter._message_matches_mention_patterns(message)


def group_trigger_text(adapter: "TelegramAdapter", message: "Message", text: Optional[str]) -> Optional[str]:
    """Strip our own handle only when we are the sole addressee. With other participants named,
    ``@research_bot , @ops_bot are you both listening?`` must not reach us as ``, @ops_bot …``."""
    if adapter._is_group_chat(message) and mentions_other_participants(adapter, message):
        return text
    return adapter._clean_bot_trigger_text(text)


def group_identity_prompt(
    adapter: "TelegramAdapter", message: "Message", channel_prompt: Optional[str],
) -> Optional[str]:
    """Session-stable identity line so the model can read retained @mentions as itself or not."""
    if not adapter._is_group_chat(message) or not getattr(adapter, "_bot", None):
        return channel_prompt
    username = adapter._current_bot_username()
    if not username:
        return channel_prompt
    identity = (
        f"Your Telegram bot username in this group: @{username}. "
        "Mentions of other bots are not requests for you to relay the message."
    )
    return f"{channel_prompt}\n\n{identity}" if channel_prompt else identity
