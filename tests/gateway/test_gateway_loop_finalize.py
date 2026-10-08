"""FIX 3 — the gateway entrypoint runs its event loop without asyncio.run's slow teardown.

``asyncio.run``'s ``loop.shutdown_default_executor()`` blocked ~3s on the restart path waiting on
default-executor threads / cancellation-swallowing tasks AFTER graceful teardown had completed.
``_run_gateway_event_loop`` / ``_finalize_gateway_loop`` replace that with a short, bounded task
sweep, since ``_exit_after_graceful_shutdown`` hard-exits via os._exit right after anyway.
"""

import asyncio
import time

import pytest

import gateway.run as gateway_run
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE


def test_run_gateway_event_loop_returns_value_and_propagates_systemexit():
    """The runner returns the coroutine's value and re-raises SystemExit (so main() routes both
    through the os._exit backstop)."""
    async def _ok():
        return True

    assert gateway_run._run_gateway_event_loop(_ok()) is True

    async def _restart():
        raise SystemExit(GATEWAY_SERVICE_RESTART_EXIT_CODE)

    with pytest.raises(SystemExit) as excinfo:
        gateway_run._run_gateway_event_loop(_restart())
    assert excinfo.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE


def test_finalize_gateway_loop_is_bounded_on_cancel_swallowing_task():
    """Finalization cancels leftover tasks under a short bound and returns promptly even if a task
    swallows the first cancel — it never blocks on shutdown_default_executor."""
    loop = asyncio.new_event_loop()

    async def _swallow_once():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            # Simulate a wedged shielded await that ignores the first cancel.
            await asyncio.sleep(100)

    async def _spawn():
        loop.create_task(_swallow_once())
        await asyncio.sleep(0)  # let it reach the first sleep

    try:
        loop.run_until_complete(_spawn())
        started = time.monotonic()
        gateway_run._finalize_gateway_loop(loop, timeout=0.3)
        elapsed = time.monotonic() - started
    finally:
        loop.close()

    assert elapsed < 3.0  # bounded; did not hang the ~3s asyncio.run teardown ever paid


def test_finalize_gateway_loop_noop_without_pending_tasks():
    """No pending tasks → finalization is an immediate no-op."""
    loop = asyncio.new_event_loop()
    try:
        started = time.monotonic()
        gateway_run._finalize_gateway_loop(loop, timeout=5.0)
        assert time.monotonic() - started < 0.5
    finally:
        loop.close()
