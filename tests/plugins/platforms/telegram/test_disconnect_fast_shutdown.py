"""FIX 2 — Telegram disconnect must not wait out the in-flight getUpdates long poll.

``updater.stop()`` only returns once the polling loop's current getUpdates completes, blocking up to
the ~10s server-side long-poll timeout (~3.5s observed on a production restart). ``_stop_updater_for_shutdown``
aborts the getUpdates HTTP request after a short grace so the long poll errors out and stop() returns
promptly — without dropping updates (an aborted poll acknowledges nothing to Telegram).
"""

import asyncio

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))


class _FakePollingRequest:
    """Stands in for PTB's getUpdates HTTPXRequest (``bot._request[0]``)."""

    def __init__(self, on_shutdown):
        self._on_shutdown = on_shutdown
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1
        self._on_shutdown()  # aborting the request unblocks the in-flight long poll

    async def initialize(self):
        return None


class _FakeUpdater:
    def __init__(self, released: asyncio.Event):
        self.running = True
        self._released = released
        self.stop_started = asyncio.Event()

    async def stop(self):
        self.stop_started.set()
        # Mimics PTB waiting for the in-flight getUpdates long poll to finish.
        await self._released.wait()


class _FakeBot:
    def __init__(self, polling_req, general_req):
        self._request = (polling_req, general_req)


class _FakeApp:
    def __init__(self, updater, bot):
        self.updater = updater
        self.bot = bot


@pytest.mark.asyncio
async def test_stop_updater_aborts_inflight_long_poll(monkeypatch):
    monkeypatch.setattr(tg_adapter, "_SHUTDOWN_LONG_POLL_GRACE", 0.05)
    adapter = _make_adapter()

    released = asyncio.Event()
    updater = _FakeUpdater(released)
    polling_req = _FakePollingRequest(on_shutdown=released.set)
    adapter._app = _FakeApp(updater, _FakeBot(polling_req, object()))

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(adapter._stop_updater_for_shutdown(), timeout=5)
    elapsed = loop.time() - started

    assert updater.stop_started.is_set()
    assert polling_req.shutdown_calls == 1  # the in-flight poll was aborted
    assert released.is_set()
    # Returned on the order of the short grace, nowhere near the 15s updater-stop timeout.
    assert elapsed < 3.0


@pytest.mark.asyncio
async def test_stop_updater_no_abort_when_poll_returns_promptly(monkeypatch):
    """When stop() completes inside the grace (poll already returned), no abort is issued."""
    monkeypatch.setattr(tg_adapter, "_SHUTDOWN_LONG_POLL_GRACE", 1.0)
    adapter = _make_adapter()

    released = asyncio.Event()
    released.set()  # poll already finished → stop() returns immediately
    updater = _FakeUpdater(released)
    polling_req = _FakePollingRequest(on_shutdown=released.set)
    adapter._app = _FakeApp(updater, _FakeBot(polling_req, object()))

    await asyncio.wait_for(adapter._stop_updater_for_shutdown(), timeout=5)

    assert updater.stop_started.is_set()
    assert polling_req.shutdown_calls == 0  # nothing to abort


@pytest.mark.asyncio
async def test_abort_inflight_getupdates_is_safe_without_app():
    """No app/bot (never connected, or torn down) → abort is a quiet no-op."""
    adapter = _make_adapter()
    adapter._app = None
    await adapter._abort_inflight_getupdates()  # must not raise
