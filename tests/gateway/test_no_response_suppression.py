"""Regression tests for internal no-response sentinel suppression."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.SLACK)
        self.sent: list[dict] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="sent-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def _keep_typing(self, chat_id, interval=2.0, metadata=None, stop_event=None) -> None:
        if stop_event is not None:
            await stop_event.wait()


def _slack_event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="channel",
            user_id="U123",
            user_name="Human",
            thread_id="1700000000.000000",
            message_id="1700000000.000001",
        ),
        message_id="1700000000.000001",
    )


@pytest.mark.asyncio
async def test_base_adapter_background_suppresses_standalone_no_response():
    """The final adapter boundary must never send a standalone [NO_RESPONSE]."""
    adapter = _CaptureAdapter()

    async def handler(event):
        return "[NO_RESPONSE]"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(_slack_event(), "slack:C123:thread")

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_base_adapter_background_keeps_real_content_with_no_response_literal():
    """Only standalone sentinels are suppressed; explanatory text still sends."""
    adapter = _CaptureAdapter()

    async def handler(event):
        return "Expected behavior: do not show [NO_RESPONSE]."

    adapter.set_message_handler(handler)
    await adapter._process_message_background(_slack_event(), "slack:C123:thread")

    assert [item["content"] for item in adapter.sent] == [
        "Expected behavior: do not show [NO_RESPONSE]."
    ]


@pytest.mark.asyncio
async def test_stream_consumer_suppresses_no_response_before_first_send():
    """Streaming previews must not leak [NO_RESPONSE] before the runner sees final text."""
    platform_adapter = MagicMock()
    platform_adapter.REQUIRES_EDIT_FINALIZE = False
    platform_adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    platform_adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    platform_adapter.MAX_MESSAGE_LENGTH = 4096

    cfg = StreamConsumerConfig(cursor="▉")
    consumer = GatewayStreamConsumer(platform_adapter, "C123", cfg)

    ok = await consumer._send_or_edit(f"[NO_RESPONSE]{cfg.cursor}")

    assert ok is True
    platform_adapter.send.assert_not_called()
    platform_adapter.edit_message.assert_not_called()
    assert consumer.final_response_sent is True
