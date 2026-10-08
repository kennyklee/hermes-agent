"""Item 5 — Telegram fallback-IP DoH discovery runs OFF the cold-connect critical path.

Cold connect starts with SEED_FALLBACK_IPS immediately and refreshes the list via DNS-over-HTTPS in a
tracked background task, so the ~0.4–0.8s (up to multi-second) inline discovery wait no longer delays
"✓ telegram connected". These tests pin the background scheduler's contract directly; the connect-path
integration (stuck discovery must not block connect; seeds are used) is covered by
test_telegram_polling_progress::test_fallback_discovery_timeout_uses_seed_ipv4.
"""

import asyncio

import pytest

from gateway.config import Platform
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_network import SEED_FALLBACK_IPS


def _bare_adapter():
    """A TelegramAdapter shell with only what _schedule_fallback_ip_discovery touches."""
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM  # backs the read-only .name property
    adapter._background_tasks = set()
    return adapter


async def _drain(adapter):
    task = getattr(adapter, "_fallback_discovery_task", None)
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_background_discovery_caches_a_non_seed_result(monkeypatch):
    discovered = ["149.154.167.99", "149.154.175.50"]

    async def _doh():
        return discovered

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _doh)
    adapter = _bare_adapter()

    adapter._schedule_fallback_ip_discovery()
    await _drain(adapter)

    assert adapter._discovered_fallback_ips == discovered


@pytest.mark.asyncio
async def test_background_discovery_does_not_cache_a_seed_equal_result(monkeypatch):
    """A seed-equal result is left uncached so a later reconnect retries DoH."""
    async def _doh():
        return list(SEED_FALLBACK_IPS)

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _doh)
    adapter = _bare_adapter()

    adapter._schedule_fallback_ip_discovery()
    await _drain(adapter)

    assert getattr(adapter, "_discovered_fallback_ips", None) is None


@pytest.mark.asyncio
async def test_background_discovery_is_idempotent_while_in_flight(monkeypatch):
    """A second schedule while one is in flight must not spawn a duplicate DoH task."""
    gate = asyncio.Event()
    calls = 0

    async def _doh():
        nonlocal calls
        calls += 1
        await gate.wait()
        return ["149.154.167.99"]

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _doh)
    adapter = _bare_adapter()

    adapter._schedule_fallback_ip_discovery()
    first = adapter._fallback_discovery_task
    adapter._schedule_fallback_ip_discovery()
    assert adapter._fallback_discovery_task is first  # no second task
    gate.set()
    await _drain(adapter)
    assert calls == 1


@pytest.mark.asyncio
async def test_background_discovery_skips_when_already_cached(monkeypatch):
    async def _boom():
        raise AssertionError("discovery must not run when a result is already cached")

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _boom)
    adapter = _bare_adapter()
    adapter._discovered_fallback_ips = ["149.154.167.99"]

    adapter._schedule_fallback_ip_discovery()
    assert getattr(adapter, "_fallback_discovery_task", None) is None


@pytest.mark.asyncio
async def test_background_discovery_survives_a_failing_doh(monkeypatch):
    """A DoH failure must not raise out of the fire-and-forget task, and leaves nothing cached."""
    async def _fail():
        raise OSError("doh unreachable")

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _fail)
    adapter = _bare_adapter()

    adapter._schedule_fallback_ip_discovery()
    await _drain(adapter)

    assert getattr(adapter, "_discovered_fallback_ips", None) is None
    assert adapter._fallback_discovery_task.done()
