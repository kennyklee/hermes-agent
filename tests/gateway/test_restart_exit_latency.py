"""Item 3 — bound the post-"Gateway stopped" Python-exit latency on the restart path.

The 2.2s gap between "Gateway stopped" and os._exit was the sum of bounded pre-exit waits that the
wedge-proof design already discards (os._exit kills the daemon MCP-shutdown thread, the leftover
tasks, and anything still in the service cgroup, with the force-reap + death supervisor + systemd as
backstops). These tests lock the tightened bounds so a future bump is caught.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gateway.run as gateway_run
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE


def test_exit_latency_constants_are_tight():
    """Regression lock: the restart-path timing bounds must stay small (see docstring)."""
    assert gateway_run._GATEWAY_LOOP_FINALIZE_TIMEOUT <= 0.25
    assert gateway_run._MCP_CHILD_TREE_REAP_GRACE <= 0.5
    assert gateway_run._MCP_SHUTDOWN_DRAIN_TIMEOUT <= 2.0


def test_finalize_gateway_loop_default_timeout_is_bounded_on_cancel_swallow():
    """With the MODULE DEFAULT bound (no explicit timeout), a task that swallows the first cancel
    cannot stall exit — proving _GATEWAY_LOOP_FINALIZE_TIMEOUT is small, not the old 1.0s."""
    loop = asyncio.new_event_loop()

    async def _swallow_once():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            await asyncio.sleep(100)  # ignore the first cancel

    async def _spawn():
        loop.create_task(_swallow_once())
        await asyncio.sleep(0)

    try:
        loop.run_until_complete(_spawn())
        started = time.monotonic()
        gateway_run._finalize_gateway_loop(loop)  # uses the module default
        elapsed = time.monotonic() - started
    finally:
        loop.close()

    # Default bound is 0.25s; allow scheduling slack but prove it is nowhere near the old 1.0s.
    assert elapsed < 0.6, f"finalize default bound too loose: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_shutdown_tail_passes_tight_mcp_drain_timeout(monkeypatch):
    """The shutdown/restart tail must cap the graceful MCP close at _MCP_SHUTDOWN_DRAIN_TIMEOUT,
    not the function's looser 5.0s default — otherwise a wedged close re-stretches the restart."""
    recorded = {}

    async def _fake_mcp_shutdown(timeout=5.0, config=None):
        recorded["timeout"] = timeout
        return True

    monkeypatch.setattr(gateway_run, "_shutdown_mcp_servers_nonblocking", _fake_mcp_shutdown)

    runner = SimpleNamespace(
        should_exit_with_failure=False, exit_reason=None, exit_code=None,
        _restart_requested=True, _restart_via_service=True, config=None)
    watcher_thread = Mock()

    with pytest.raises(SystemExit) as excinfo:
        await gateway_run._start_gateway_shutdown_tail(
            runner, None, threading.Event(), None, None, None,
            threading.Event(), watcher_thread, [False])

    assert excinfo.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert recorded["timeout"] == gateway_run._MCP_SHUTDOWN_DRAIN_TIMEOUT
