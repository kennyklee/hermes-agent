"""Restart/stop must not block on a hygiene-compression worker.

A detached session-hygiene summary is tracked as a deferred agent worker (gateway/run_turn.py), and the
after-turn restart wait used to hold for it like any other active work — so a /restart could sit ~70s behind
a convenience re-summarization (the production incident behind this change). The policy is now:

* a summary NOT yet inside its commit is abandoned (commit admission revoked so it can never mutate the
  session afterwards, agent hard-interrupted so its thread unwinds), leaving the full transcript persisted
  for the next boot to re-compress;
* a summary already inside its watermark-fenced commit is waited for, but only briefly (bounded), because
  that commit is one atomic SQLite transaction;
* a NON-compression deferred worker still defers the restart exactly as before.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


class FakeFence:
    """Minimal ``CompressionCommitFence`` stand-in: exposes the two members the restart path reads."""

    def __init__(self, commit_in_flight: bool = False):
        self._commit_in_flight = commit_in_flight
        self.revoked = False

    @property
    def commit_in_flight(self) -> bool:
        return self._commit_in_flight

    def revoke_commit_admission(self) -> None:
        self.revoked = True


def _restart_runner():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner._restart_after_turn_timeout = 300.0
    runner._scale_to_zero_status = MagicMock()
    return runner, adapter


@pytest.mark.asyncio
async def test_pre_commit_compression_is_excluded_from_awaitable_counts():
    runner, _ = _restart_runner()
    worker = asyncio.get_running_loop().create_future()
    fence = FakeFence(commit_in_flight=False)
    runner._track_deferred_agent_worker(worker, MagicMock(), commit_fence=fence)

    # Still a live worker for raw accounting / the shutdown snapshot …
    assert runner._active_deferred_agent_worker_count() == 1
    # … but NOT something the after-turn wait or the drain hold for.
    assert runner._awaitable_deferred_agent_worker_count() == 0
    assert runner._awaitable_work_count() == 0
    assert runner._drain_deferred_worker_count() == 0
    worker.cancel()


@pytest.mark.asyncio
async def test_restart_wait_abandons_pre_commit_compression_and_proceeds_immediately():
    runner, _ = _restart_runner()
    worker = asyncio.get_running_loop().create_future()
    agent = MagicMock()
    fence = FakeFence(commit_in_flight=False)
    runner._track_deferred_agent_worker(worker, agent, commit_fence=fence)

    with patch("gateway.run.request_hard_interrupt") as interrupt:
        proceeded = await asyncio.wait_for(
            runner._await_active_work_before_restart(), timeout=1.0
        )

    # Nothing awaitable remains → proceed straight to stop(), no 70s hold.
    assert proceeded is False
    # Revoked (so a late worker can never mutate the session) and interrupted (so its thread unwinds).
    assert fence.revoked is True
    interrupt.assert_called_once_with(agent, "Gateway restarting", tool_reason="gateway shutdown")
    worker.cancel()


@pytest.mark.asyncio
async def test_restart_wait_waits_only_for_the_commit_section_then_returns_drained():
    runner, _ = _restart_runner()
    worker = asyncio.get_running_loop().create_future()
    agent = MagicMock()
    fence = FakeFence(commit_in_flight=True)
    runner._track_deferred_agent_worker(worker, agent, commit_fence=fence)

    # A committing worker is NOT abandoned; it is held for by the bounded commit wait.
    assert runner._committing_compression_count() == 1
    assert runner._awaitable_deferred_agent_worker_count() == 0

    async def _finish_commit():
        await asyncio.sleep(0.05)
        fence._commit_in_flight = False
        worker.set_result(([], None))

    finisher = asyncio.create_task(_finish_commit())
    with patch("gateway.run.request_hard_interrupt") as interrupt:
        proceeded = await asyncio.wait_for(
            runner._await_active_work_before_restart(), timeout=2.0
        )
    await finisher

    # Commit finished within the window → fully drained; the worker was never abandoned/interrupted.
    assert proceeded is True
    assert fence.revoked is False
    interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_commit_section_wait_is_bounded_when_commit_never_finishes():
    runner, _ = _restart_runner()
    runner._RESTART_COMPRESSION_COMMIT_WAIT_S = 0.05  # keep the bound tiny for the test
    worker = asyncio.get_running_loop().create_future()
    fence = FakeFence(commit_in_flight=True)  # never leaves the commit section
    runner._track_deferred_agent_worker(worker, MagicMock(), commit_fence=fence)

    with patch("gateway.run.request_hard_interrupt"):
        proceeded = await asyncio.wait_for(
            runner._await_active_work_before_restart(), timeout=2.0
        )

    # The bounded wait elapsed and we proceeded to stop() rather than hanging on the commit.
    assert proceeded is False
    worker.cancel()


@pytest.mark.asyncio
async def test_non_compression_deferred_worker_still_defers_restart():
    runner, _ = _restart_runner()
    worker = asyncio.get_running_loop().create_future()
    runner._deferred_agent_workers = {worker: MagicMock()}

    assert runner._awaitable_deferred_agent_worker_count() == 1
    assert runner._awaitable_work_count() == 1

    # The wait holds (and times out) for a non-compression deferred worker, exactly as before.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(runner._await_active_work_before_restart(), timeout=0.3)
    worker.cancel()


@pytest.mark.asyncio
async def test_abandon_is_idempotent_and_leaves_transcript_untouched():
    """Abandonment's ONLY session-facing action is revoking commit admission, which the fence contract
    turns into ``begin_commit`` refusing — so compress_context returns the original transcript unchanged
    and ``archive_and_compact`` is never called (full history stays persisted; next boot re-runs hygiene)."""
    runner, _ = _restart_runner()
    worker = asyncio.get_running_loop().create_future()
    agent = MagicMock()
    fence = FakeFence(commit_in_flight=False)
    runner._track_deferred_agent_worker(worker, agent, commit_fence=fence)

    with patch("gateway.run.request_hard_interrupt") as interrupt:
        first = runner._abandon_restart_blocking_compression("Gateway restarting")
        second = runner._abandon_restart_blocking_compression("Gateway restarting")

    assert (first, second) == (1, 0)  # idempotent: second call abandons nothing new
    assert fence.revoked is True
    interrupt.assert_called_once()
    worker.cancel()


@pytest.mark.asyncio
async def test_drain_ignores_pre_commit_compression_but_waits_for_committing():
    runner, _ = _restart_runner()
    pre_commit = asyncio.get_running_loop().create_future()
    runner._track_deferred_agent_worker(pre_commit, MagicMock(), commit_fence=FakeFence(False))

    # The stop() drain does not hold for a pre-commit summary.
    assert runner._drain_deferred_worker_count() == 0
    _snap, timed_out = await runner._drain_active_agents(2.0)
    assert timed_out is False

    # A committing summary IS counted by the drain (one atomic transaction to let finish).
    committing = asyncio.get_running_loop().create_future()
    runner._track_deferred_agent_worker(committing, MagicMock(), commit_fence=FakeFence(True))
    assert runner._drain_deferred_worker_count() == 1

    pre_commit.cancel()
    committing.cancel()
