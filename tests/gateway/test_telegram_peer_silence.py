"""Peer replies must not turn intentional silence into a bot warning loop."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms.event import MessageType
from tests.gateway.test_telegram_group_gating import _make_adapter, _group_message
from tests.gateway.test_gateway_silence_tokens import _runner


@pytest.mark.parametrize("scope", [{}, {"free_response_chats": ["-100"]}, {"free_response_topics": ["-100:7"]}])
@pytest.mark.parametrize("require_mention", [False, True])
@pytest.mark.parametrize("via_env", [False, True])
def test_mentions_mode_requires_current_peer_address(monkeypatch, scope, require_mention, via_env):
    adapter = _make_adapter(require_mention=require_mention, mention_patterns=["Hermes"],
                            observe_unmentioned_group_messages=True, allowed_chats=["-100"], **scope)
    if via_env:
        monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "mentions")
        adapter.config.extra["allow_bots"] = "all"
    else:
        monkeypatch.delenv("TELEGRAM_ALLOW_BOTS", raising=False)
        adapter.config.extra["allow_bots"] = "mentions"
    for text in ("NO_REPLY", "Hermes, model returned empty content", "plain chatter"):
        msg = _group_message(text, reply_to_bot=True, thread_id=7)
        msg.from_user.is_bot = True
        assert not adapter._should_process_message(msg)
        assert not adapter._should_observe_unmentioned_group_message(msg)
        msg.from_user.is_bot = False
        assert adapter._should_process_message(msg)  # human reply still works
    msg = _group_message("@hermes_bot please check", thread_id=7)
    msg.from_user.is_bot = True
    assert adapter._should_process_message(msg)


def test_unmentioned_peer_is_not_observed_but_human_is():
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"],
                            group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter.config.extra["allow_bots"] = "mentions"
    msg = _group_message("plain chatter")
    msg.from_user.is_bot = True
    assert not adapter._should_observe_unmentioned_group_message(msg)
    msg.from_user.is_bot = False
    assert adapter._should_observe_unmentioned_group_message(msg)


@pytest.mark.asyncio
async def test_unmentioned_peer_stops_before_dispatch_and_reactions():
    adapter = _make_adapter(require_mention=False, allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter.config.extra["allow_bots"] = "mentions"
    adapter._enqueue_text_event = Mock()
    adapter._observe_unmentioned_group_message = Mock()
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter.on_processing_start = AsyncMock()
    msg = _group_message("NO_REPLY", reply_to_bot=True)
    msg.from_user.is_bot = True
    await adapter._handle_text_message(SimpleNamespace(message=msg, update_id=1), None)
    adapter._enqueue_text_event.assert_not_called()
    adapter._observe_unmentioned_group_message.assert_not_called()
    adapter.on_processing_start.assert_not_awaited()
    adapter._ensure_forum_commands.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot,address,expected", [
    (True, "reply", False), (True, "plain", False), (True, "mention", False),
    (False, "reply", True), (False, "plain", False), (False, "mention", True),
    (False, "dm", True), (False, "wake", True), (False, "unknown", None),
])
@pytest.mark.parametrize("final", ["[SILENT]", "(empty)"])
async def test_telegram_addressing_reaches_gateway_silence_policy(monkeypatch, tmp_path, is_bot, address, expected, final):
    adapter = _make_adapter(require_mention=False, mention_patterns=["^Hermes"])
    adapter.config.extra["allow_bots"] = "all"
    text = {"mention": "@hermes_bot check", "wake": "Hermes check"}.get(address, "side chatter")
    msg = _group_message(text, reply_to_bot=address == "reply")
    msg.from_user.is_bot = is_bot
    if address == "dm":
        msg.chat.type = "private"
    if address == "unknown":
        adapter._bot = None
    event = adapter._build_message_event(msg, MessageType.TEXT)
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": final, "messages": [{"role": "user", "content": text},
        {"role": "assistant", "content": final}], "tools": [], "history_offset": 0,
        "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    })
    response = await runner._handle_message_with_agent(
        event, event.source, "agent:main:telegram:group:-1001:12345", 1)
    # Only intentional silence on an unaddressed turn is suppressed. A bot peer's turn
    # never earns the visible failure fallback either way, to avoid a warning loop.
    # Actual empty model failures and directed human requests retain the fallback.
    if is_bot or (final == "[SILENT]" and expected is False):
        assert response == ""
    else:
        assert response and response.strip() not in {"[SILENT]", "(empty)"}
    assert event.reply_expected is expected


@pytest.mark.parametrize("mode", ["mentions", "all"])
@pytest.mark.parametrize("warning", [
    "⚠️ The model returned a silence marker instead of a response.",
    "⚠️ The model returned no response after processing tool results. Try again.",
    "⚠️ Processing completed but no response was generated. Try again.",
])
def test_tagged_automated_warning_cannot_restart_peer_loop(mode, warning):
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"],
                            group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter.config.extra["allow_bots"] = mode
    msg = _group_message("@hermes_bot " + warning, reply_to_bot=True)
    msg.from_user.is_bot = True
    assert not adapter._should_process_message(msg)
    assert not adapter._should_observe_unmentioned_group_message(msg)
    msg.from_user.is_bot = False
    assert adapter._should_process_message(msg)  # humans can report the warning


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["[SILENT]", "NO_REPLY", "(empty)", "  (empty)\n"])
async def test_tagged_bot_handoff_can_intentionally_stay_silent(monkeypatch, tmp_path, final):
    adapter = _make_adapter(require_mention=True, bot_username="HenryHermesAi_bot")
    adapter.config.extra["allow_bots"] = "mentions"
    msg = _group_message("@HenryHermesAi_bot HANDOFF-CHECK Mac")
    msg.from_user.is_bot = True
    assert adapter._should_process_message(msg)
    event = adapter._build_message_event(msg, MessageType.TEXT)
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": final, "messages": [], "tools": [],
        "api_calls": 1, "failed": False,
    })
    response = await runner._handle_message_with_agent(
        event, event.source, "agent:main:telegram:group:-1001:12345", 1)
    assert response == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot", [False, True])
@pytest.mark.parametrize("final,failed", [
    ("", True), ("(empty)", True), ("NO_REPLY", True),
    ("The directory is (empty), so no files were changed.", False),
])
async def test_failure_and_substantive_empty_prose_are_preserved(monkeypatch, tmp_path, is_bot, final, failed):
    adapter = _make_adapter()
    msg = _group_message("@hermes_bot check")
    msg.from_user.is_bot = is_bot
    event = adapter._build_message_event(msg, MessageType.TEXT)
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": final, "messages": [], "tools": [],
        "api_calls": 1, "failed": failed, "error": "upstream unavailable",
    })
    response = await runner._handle_message_with_agent(
        event, event.source, "agent:main:telegram:group:-1001:12345", 1)
    if failed:
        assert response  # upstream wording differs; a failed turn is never silent
    else:
        assert response == final
