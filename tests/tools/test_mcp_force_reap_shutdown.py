"""FIX 1 — terminal reap of MCP child process TREES on gateway shutdown.

The graceful ``shutdown_mcp_servers`` path can be abandoned before its own SIGKILL pass when its
budget is spent inside the SDK transport close / loop drain, leaving npx→node trees alive in the
gateway cgroup for the death supervisor / systemd to reap ~3-4s after exit. ``_force_reap_mcp_child_trees``
is the guaranteed, prompt terminal reap that leaves the cgroup empty at exit.
"""

import signal

from unittest.mock import patch


def _reset_state():
    from tools.mcp_tool import _lock
    from tools.mcp_tool_lifecycle import (
        _orphan_stdio_pid_servers, _orphan_stdio_pids, _stdio_pgids, _stdio_pids, _stdio_starttimes)
    with _lock:
        _stdio_pids.clear()
        _orphan_stdio_pids.clear()
        _orphan_stdio_pid_servers.clear()
        _stdio_pgids.clear()
        _stdio_starttimes.clear()


def test_force_reap_noop_when_no_children():
    """No tracked children → returns 0 and never sleeps (every MCP-free shutdown pays nothing)."""
    from tools.mcp_tool_lifecycle import _force_reap_mcp_child_trees

    _reset_state()
    with patch("tools.mcp_tool.time.sleep") as mock_sleep, \
         patch("tools.mcp_tool.os.killpg") as mock_killpg:
        assert _force_reap_mcp_child_trees(grace=1.0) == 0
    mock_sleep.assert_not_called()
    mock_killpg.assert_not_called()


def test_force_reap_kills_active_child_tree_via_killpg():
    """An ACTIVE stdio child (in _stdio_pids) is SIGTERM'd then SIGKILL'd through its process group
    (reaching reparented npx→node grandchildren), with the short shutdown grace, and counted."""
    from tools.mcp_tool import _lock
    from tools.mcp_tool_lifecycle import _force_reap_mcp_child_trees, _stdio_pgids, _stdio_pids, _stdio_starttimes

    _reset_state()
    fake_pid = 515151
    with _lock:
        _stdio_pids[fake_pid] = "server-a"
        _stdio_pgids[fake_pid] = fake_pid  # start_new_session spawn: pgid == leader pid
        _stdio_starttimes[fake_pid] = 999999

    with patch("tools.mcp_tool_lifecycle._leader_start_time", return_value=999999), \
         patch("tools.mcp_tool.os.killpg") as mock_killpg, \
         patch("gateway.status._pid_exists", return_value=True), \
         patch("tools.mcp_tool.time.sleep") as mock_sleep:
        reaped = _force_reap_mcp_child_trees(grace=0.25)

    assert reaped == 1
    # Short shutdown grace is honoured (not the 2s default).
    mock_sleep.assert_called_once_with(0.25)
    mock_killpg.assert_any_call(fake_pid, signal.SIGTERM)
    mock_killpg.assert_any_call(fake_pid, signal.SIGKILL)  # survived SIGTERM → forced


def test_force_reap_drains_ledgers_so_second_pass_is_noop():
    """Reaping pops the PIDs out of the ledgers, so a racing/second reap finds nothing."""
    from tools.mcp_tool import _lock
    from tools.mcp_tool_lifecycle import _force_reap_mcp_child_trees, _stdio_pgids, _stdio_pids

    _reset_state()
    fake_pid = 525252
    with _lock:
        _stdio_pids[fake_pid] = "server-b"
        _stdio_pgids[fake_pid] = fake_pid

    with patch("tools.mcp_tool.os.killpg"), \
         patch("gateway.status._pid_exists", return_value=False), \
         patch("tools.mcp_tool.time.sleep"):
        assert _force_reap_mcp_child_trees(grace=0.0) == 1
        assert _force_reap_mcp_child_trees(grace=0.0) == 0
