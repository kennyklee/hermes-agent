"""Cold-start fast-connect: confirm the transport via a getMe bootstrap probe instead of blocking the
connect banner on the first getUpdates long-poll (which returns only at the ~10s long-poll timeout when the
queue is empty). Update delivery and wedge detection must be unchanged."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _ControlledRequest:
    """Minimal PTB request double (mirrors test_telegram_polling_progress)."""

    instances = []

    @staticmethod
    def parse_json_payload(payload):
        return json.loads(payload.decode("utf-8", "replace"))

    def __init__(self, *args, result=None, error=None, **kwargs):
        self.result = result
        self.error = error
        self.args = args
        self.kwargs = kwargs
        type(self).instances.append(self)

    async def do_request(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


class _LifecycleBuilder:
    def __init__(self, app):
        self.app = app
        self.polling_request = None

    def token(self, _token):
        return self

    def application_class(self, _application_class, _kwargs=None):
        return self

    def request(self, _request):
        return self

    def get_updates_request(self, request):
        self.polling_request = request
        return self

    def concurrent_updates(self, _processor):
        return self

    def build(self):
        return self.app


def _lifecycle_app(*, get_me):
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = True
    # An idle cold boot: the first getUpdates long-poll has NOT returned, so no progress is recorded.
    app.updater.start_polling = AsyncMock()
    app.updater.start_webhook = AsyncMock()
    app.updater.stop = AsyncMock()
    app.bot = MagicMock()
    app.bot.delete_webhook = AsyncMock()
    app.bot.get_me = get_me
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.running = True
    return app


def _configure(monkeypatch, adapter, app):
    builder = _LifecycleBuilder(app)

    class _Application:
        @staticmethod
        def builder():
            return builder

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _ControlledRequest)
    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", MagicMock())

    async def _noop_heartbeat():
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", _noop_heartbeat)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    return builder


def _make_adapter(**extra) -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra))


@pytest.mark.asyncio
async def test_cold_connect_confirms_via_getme_without_awaiting_first_long_poll(monkeypatch):
    """Idle cold boot: getUpdates has not round-tripped, but a getMe probe confirms the transport, so
    connect returns fast. The getUpdates offset is untouched and the background verifier is armed."""
    get_me = AsyncMock(return_value=MagicMock())
    # drop_pending_on_cold_boot=false → queued updates must be preserved; the fix must not change this.
    adapter = _make_adapter(drop_pending_on_cold_boot=False)
    app = _lifecycle_app(get_me=get_me)
    builder = _configure(monkeypatch, adapter, app)

    try:
        connected = await asyncio.wait_for(adapter.connect(), timeout=2)
        assert connected is True
        # The getMe bootstrap probe confirmed health — not a getUpdates round-trip.
        get_me.assert_awaited()
        assert not adapter._polling_progress_event.is_set()
        # getMe proved the send path: outbound sends (restart notice, replies) must not be refused as
        # send_path_degraded while the first idle long poll is still pending.
        assert adapter.send_path_degraded is False
        # getUpdates offset preserved exactly: cold-boot drop_pending honored, probe issued no get_updates.
        app.updater.start_polling.assert_awaited_once()
        assert app.updater.start_polling.await_args.kwargs["drop_pending_updates"] is False
        assert builder.polling_request.result is None  # probe never drove the getUpdates request
        # Wedge detection intact: the background progress verifier is armed to require real getUpdates progress.
        assert adapter._polling_progress_verifier_task is not None
        assert not adapter._polling_progress_verifier_task.done()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_real_getupdates_progress_still_satisfies_cold_connect(monkeypatch):
    """When updates are queued the first getUpdates round-trip fires immediately; that strongest proof still
    opens the gate (and getMe need never be probed)."""
    get_me = AsyncMock(return_value=MagicMock())
    adapter = _make_adapter()
    app = _lifecycle_app(get_me=get_me)

    async def start_polling_records_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    app.updater.start_polling = AsyncMock(side_effect=start_polling_records_progress)
    _configure(monkeypatch, adapter, app)

    try:
        connected = await asyncio.wait_for(adapter.connect(), timeout=2)
        assert connected is True
        assert adapter._polling_progress_event.is_set()
        # progress won the race before the probe was ever needed.
        get_me.assert_not_awaited()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_cold_connect_fails_closed_when_getme_and_getupdates_both_dead(monkeypatch):
    """A network-dead transport: getMe errors and getUpdates never progresses, so the gate falls through to
    the overall deadline and fails closed (retryable fatal) — unchanged fail-closed semantics."""
    get_me = AsyncMock(side_effect=ConnectionError("offline"))
    adapter = _make_adapter()
    app = _lifecycle_app(get_me=get_me)
    _configure(monkeypatch, adapter, app)
    # Keep the overall readiness deadline short so the test is fast.
    monkeypatch.setattr(tg_adapter, "_INITIAL_POLLING_PROGRESS_TIMEOUT", 0.3)

    try:
        connected = await asyncio.wait_for(adapter.connect(), timeout=3)
        assert connected is False
        assert adapter.has_fatal_error is True
    finally:
        await adapter.disconnect()
