"""Tests for ``/reload-platform <name>`` — hot-reload one platform adapter's connection
settings from config.yaml without a full gateway restart.

Covers both the lifecycle primitive (``_reload_platform_adapter`` — disconnect the live
adapter, reconnect with fresh config via the reconnect-watcher machinery, preserving the
server-side update queue) and the slash-command handler (argument parsing, fresh-config read,
and the refuse-rather-than-half-reload guards).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class StubAdapter(BasePlatformAdapter):
    """Adapter whose connect() result is controllable; records is_reconnect per connect()."""

    def __init__(self, *, platform=Platform.TELEGRAM, succeed=True, token="test"):
        super().__init__(PlatformConfig(enabled=True, token=token), platform)
        self._succeed = succeed
        self.connect_calls: list[bool] = []
        self.disconnected = False

    async def connect(self, *, is_reconnect: bool = False):
        self.connect_calls.append(is_reconnect)
        return self._succeed

    async def disconnect(self):
        self.disconnected = True

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_runner():
    """Minimal GatewayRunner with just the attributes the reload paths touch."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="old")}
    )
    runner._running = True
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    runner.session_store = MagicMock()
    # Install-path collaborators driven by _reconnect_failed_platform → _install_reconnected_adapter.
    runner._sync_voice_mode_state_to_adapter = MagicMock()
    runner._bind_voice_input_callback = MagicMock()
    runner._wire_adapter_handlers = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    runner._ensure_reconnect_watcher_running = MagicMock()
    runner._schedule_planned_restart_replay = MagicMock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    runner._schedule_resume_pending_sessions = MagicMock(return_value=0)
    runner._bounded_adapter_teardown = AsyncMock()
    return runner


def _make_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm"
    )
    return MessageEvent(text=text, source=source, message_id="m1")


# --- Lifecycle primitive ---------------------------------------------------------------------

class TestReloadPlatformAdapter:
    @pytest.mark.asyncio
    async def test_reload_disconnects_old_and_reconnects_with_is_reconnect(self):
        """A live adapter is torn down and a fresh one connects with is_reconnect=True so the
        server-side update queue (offline-period messages) is preserved (#46621)."""
        runner = _make_runner()
        old = StubAdapter(token="old")
        runner.adapters[Platform.TELEGRAM] = old
        new = StubAdapter(token="new")

        new_config = PlatformConfig(enabled=True, token="new")
        with patch.object(runner, "_create_adapter", return_value=new):
            with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
                connected = await runner._reload_platform_adapter(Platform.TELEGRAM, new_config)

        assert connected is True
        assert runner.adapters[Platform.TELEGRAM] is new
        assert new.connect_calls == [True], "reload must preserve the update queue (is_reconnect=True)"
        runner._bounded_adapter_teardown.assert_awaited_once()
        # Running config now points at the fresh section for every downstream path.
        assert runner.config.platforms[Platform.TELEGRAM] is new_config
        assert Platform.TELEGRAM not in runner._failed_platforms

    @pytest.mark.asyncio
    async def test_failed_reconnect_stays_queued_for_background_retry(self):
        """If the fresh adapter fails to connect, the platform is left in the reconnect queue so the
        watcher keeps retrying with the new config (the command reports retrying-in-background)."""
        runner = _make_runner()
        old = StubAdapter(token="old")
        runner.adapters[Platform.TELEGRAM] = old
        failing = StubAdapter(token="new", succeed=False)

        new_config = PlatformConfig(enabled=True, token="new")
        with patch.object(runner, "_create_adapter", return_value=failing):
            with patch("gateway.run._dispose_unused_adapter", new=AsyncMock()):
                connected = await runner._reload_platform_adapter(Platform.TELEGRAM, new_config)

        assert connected is False
        assert Platform.TELEGRAM not in runner.adapters
        assert Platform.TELEGRAM in runner._failed_platforms
        assert runner._failed_platforms[Platform.TELEGRAM]["config"] is new_config


# --- Slash-command handler -------------------------------------------------------------------

class TestReloadPlatformCommand:
    @pytest.mark.asyncio
    async def test_reload_reads_fresh_config_and_reconnects(self):
        """The handler loads config.yaml fresh, pulls the named platform's section, and swaps it in."""
        runner = _make_runner()
        runner.adapters[Platform.TELEGRAM] = StubAdapter(token="old")
        fresh_config = PlatformConfig(enabled=True, token="new")
        fresh = GatewayConfig(platforms={Platform.TELEGRAM: fresh_config})
        runner._reload_platform_adapter = AsyncMock(return_value=True)

        with patch("gateway.config.load_gateway_config", return_value=fresh):
            out = await runner._handle_reload_platform_command(_make_event("/reload-platform telegram"))

        runner._reload_platform_adapter.assert_awaited_once_with(Platform.TELEGRAM, fresh_config)
        assert "telegram" in out and "reconnected" in out

    @pytest.mark.asyncio
    async def test_retrying_message_when_not_immediately_connected(self):
        runner = _make_runner()
        runner.adapters[Platform.TELEGRAM] = StubAdapter(token="old")
        fresh = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="new")})
        runner._reload_platform_adapter = AsyncMock(return_value=False)

        with patch("gateway.config.load_gateway_config", return_value=fresh):
            out = await runner._handle_reload_platform_command(_make_event("/reload-platform telegram"))

        assert "retrying" in out.lower() or "background" in out.lower()

    @pytest.mark.asyncio
    async def test_usage_when_no_name(self):
        runner = _make_runner()
        out = await runner._handle_reload_platform_command(_make_event("/reload-platform"))
        assert "Usage" in out

    @pytest.mark.asyncio
    async def test_unknown_platform(self):
        runner = _make_runner()
        out = await runner._handle_reload_platform_command(_make_event("/reload-platform nope"))
        assert "Unknown platform" in out

    @pytest.mark.asyncio
    async def test_refuses_when_not_connected(self):
        """A platform with no live adapter (not queued) is refused rather than half-reloaded."""
        runner = _make_runner()
        runner._reload_platform_adapter = AsyncMock()
        out = await runner._handle_reload_platform_command(_make_event("/reload-platform discord"))
        assert "discord" in out
        runner._reload_platform_adapter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refuses_when_queued(self):
        """A platform already in the reconnect queue is owned by the watcher — refuse."""
        runner = _make_runner()
        runner._failed_platforms[Platform.DISCORD] = {
            "config": PlatformConfig(enabled=True, token="t"), "attempts": 1, "next_retry": 0,
        }
        runner._reload_platform_adapter = AsyncMock()
        out = await runner._handle_reload_platform_command(_make_event("/reload-platform discord"))
        assert "queue" in out.lower()
        runner._reload_platform_adapter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refuses_when_disabled_in_fresh_config(self):
        """If the fresh config disables the platform, there is nothing to reconnect — refuse."""
        runner = _make_runner()
        runner.adapters[Platform.TELEGRAM] = StubAdapter(token="old")
        fresh = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=False, token="x")})
        runner._reload_platform_adapter = AsyncMock()

        with patch("gateway.config.load_gateway_config", return_value=fresh):
            out = await runner._handle_reload_platform_command(_make_event("/reload-platform telegram"))

        assert "disabled" in out.lower()
        runner._reload_platform_adapter.assert_not_awaited()


# --- SIGHUP hot reload -----------------------------------------------------------------------

class TestReloadChangedPlatforms:
    @pytest.mark.asyncio
    async def test_reloads_only_changed_connected_platforms(self):
        """SIGHUP reconnects only platforms whose section changed; unchanged/disconnected skipped."""
        runner = _make_runner()
        runner.adapters = {
            Platform.TELEGRAM: StubAdapter(platform=Platform.TELEGRAM, token="old"),
            Platform.DISCORD: StubAdapter(platform=Platform.DISCORD, token="same"),
        }
        runner.config = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="old"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="same"),
        })
        fresh = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="new"),   # changed
            Platform.DISCORD: PlatformConfig(enabled=True, token="same"),   # unchanged
            Platform.SLACK: PlatformConfig(enabled=True, token="x"),        # not connected
        })
        runner._reload_platform_adapter = AsyncMock(return_value=True)

        with patch("gateway.config.load_gateway_config", return_value=fresh):
            reloaded = await runner._reload_changed_platforms_from_disk()

        assert reloaded == [Platform.TELEGRAM]
        runner._reload_platform_adapter.assert_awaited_once_with(
            Platform.TELEGRAM, fresh.platforms[Platform.TELEGRAM]
        )

    @pytest.mark.asyncio
    async def test_noop_when_nothing_changed(self):
        runner = _make_runner()
        runner.adapters = {Platform.TELEGRAM: StubAdapter(token="old")}
        fresh = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="old")})
        runner._reload_platform_adapter = AsyncMock()

        with patch("gateway.config.load_gateway_config", return_value=fresh):
            reloaded = await runner._reload_changed_platforms_from_disk()

        assert reloaded == []
        runner._reload_platform_adapter.assert_not_awaited()

    def test_sighup_handler_schedules_reload(self):
        """The SIGHUP signal handler builds and retains a background reload task."""
        from gateway.run import _start_gateway_make_reload_platforms_signal_handler

        runner = MagicMock()

        async def _coro():
            return []

        runner._reload_changed_platforms_from_disk = MagicMock(return_value=_coro())
        handler = _start_gateway_make_reload_platforms_signal_handler(runner)

        async def _drive():
            handler()
            await asyncio.sleep(0)

        asyncio.run(_drive())
        runner._reload_changed_platforms_from_disk.assert_called_once()
        runner._retain_background_task.assert_called_once()


# --- Registration consistency ----------------------------------------------------------------

class TestReloadPlatformRegistration:
    def test_command_registered(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = {c.name for c in COMMAND_REGISTRY}
        assert "reload-platform" in names

    def test_dispatched_on_idle_path(self):
        from gateway.run_busy import GatewayBusySessionMixin

        assert "reload-platform" in GatewayBusySessionMixin._IDLE_COMMANDS

    def test_handler_discoverable_by_name(self):
        """``reload-platform`` → ``_handle_reload_platform_command`` (the ``-``→``_`` convention)."""
        from gateway.run import GatewayRunner

        assert hasattr(GatewayRunner, "_handle_reload_platform_command")
