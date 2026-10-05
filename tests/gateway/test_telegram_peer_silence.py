"""Peer replies must not turn intentional silence into a bot warning loop."""
from unittest.mock import AsyncMock

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


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot,address,expected", [
    (True, "reply", False), (True, "plain", False), (True, "mention", True),
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
    # Only intentional silence on an unaddressed turn is suppressed. Actual empty
    # model failures and directed human requests retain the visible fallback.
    if final == "[SILENT]" and expected is False:
        assert response == ""
    else:
        assert response and response != "[SILENT]"
    assert event.reply_expected is expected
