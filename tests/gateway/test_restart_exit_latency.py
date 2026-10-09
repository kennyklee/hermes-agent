"""Item 3 — bound the post-"Gateway stopped" Python-exit latency on the restart path.

The 2.2s gap between "Gateway stopped" and os._exit was the sum of bounded pre-exit waits that the
wedge-proof design already discards (os._exit kills the daemon MCP-shutdown thread, the leftover
tasks, and anything still in the service cgroup, with the force-reap + death supervisor + systemd as
backstops). These tests lock the tightened bounds so a future bump is caught.
"""

import asyncio
import json
import logging
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
    # Lowered 2.0s -> 1.0s: the exit-tail instrumentation showed the graceful MCP close dominating the
    # post-"Gateway stopped" gap, and this budget carries no data-safety obligation (durable state is
    # flushed in _stop_impl before "Gateway stopped"; force-reap still guarantees teardown).
    assert gateway_run._MCP_SHUTDOWN_DRAIN_TIMEOUT <= 1.0


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


@pytest.mark.asyncio
async def test_shutdown_tail_records_per_phase_timing(tmp_path, monkeypatch, caplog):
    """The post-'Gateway stopped' tail must log an INFO breakdown and write a durable
    gateway.exit_tail_timing record (per-phase + dominant) even though the verdict raises SystemExit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async def _fake_mcp_shutdown(timeout=5.0, config=None):
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(gateway_run, "_shutdown_mcp_servers_nonblocking", _fake_mcp_shutdown)

    runner = SimpleNamespace(
        should_exit_with_failure=False, exit_reason=None, exit_code=None,
        _restart_requested=True, _restart_via_service=True, config=None)

    with caplog.at_level(logging.INFO, logger=gateway_run.logger.name):
        with pytest.raises(SystemExit) as excinfo:
            await gateway_run._start_gateway_shutdown_tail(
                runner, None, threading.Event(), None, None, None,
                threading.Event(), Mock(), [False])

    assert excinfo.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE

    diag = tmp_path / "logs" / "gateway-exit-diag.log"
    assert diag.exists(), "exit-tail timing was not written to the exit-diag JSONL"
    records = [json.loads(line) for line in diag.read_text(encoding="utf-8").splitlines() if line.strip()]
    tail = [r for r in records if r.get("tag") == "gateway.exit_tail_timing"]
    assert tail, f"no gateway.exit_tail_timing record in {records}"
    rec = tail[-1]
    assert {"cron_ticker_drain", "housekeeping_drain", "mcp_shutdown"} <= set(rec["phases_s"])
    assert isinstance(rec["total_s"], (int, float))
    assert rec["dominant"] in rec["phases_s"]
    assert any("Shutdown tail complete" in r.getMessage() for r in caplog.records), "no INFO breakdown"


def test_finalize_gateway_loop_records_timing_when_tasks_remain(tmp_path, monkeypatch):
    """A leftover-task cancellation sweep writes a gateway.loop_finalize_timing record (the sweep runs
    after the tail, so its cost belongs in the same 'Gateway stopped' → loop-exit budget)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    loop = asyncio.new_event_loop()

    async def _sleeper():
        await asyncio.sleep(100)

    async def _spawn():
        loop.create_task(_sleeper())
        await asyncio.sleep(0)

    try:
        loop.run_until_complete(_spawn())
        gateway_run._finalize_gateway_loop(loop, timeout=0.2)
    finally:
        loop.close()

    diag = tmp_path / "logs" / "gateway-exit-diag.log"
    records = [json.loads(line) for line in diag.read_text(encoding="utf-8").splitlines() if line.strip()]
    fin = [r for r in records if r.get("tag") == "gateway.loop_finalize_timing"]
    assert fin and fin[-1]["task_count"] >= 1, f"no loop_finalize_timing record in {records}"


def test_finalize_gateway_loop_records_nothing_without_leftover_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    loop = asyncio.new_event_loop()
    try:
        gateway_run._finalize_gateway_loop(loop, timeout=0.2)
    finally:
        loop.close()
    diag = tmp_path / "logs" / "gateway-exit-diag.log"
    assert not diag.exists() or "loop_finalize_timing" not in diag.read_text(encoding="utf-8")
