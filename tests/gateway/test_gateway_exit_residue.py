"""FIX 2 — the hard-exit backstop logs what is still alive (non-daemon threads + direct child PIDs) so a
slow cgroup-empty on the restart path is diagnosable from logs. Purely observational: it signals nothing
and kills nothing (background terminal processes are intentionally persisted across a restart)."""

import logging
import subprocess
import sys
import threading
import time

import gateway.run as gateway_run


def test_live_nondaemon_threads_reports_a_blocked_worker():
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, name="residue-probe-thread", daemon=False)
    worker.start()
    try:
        names = {t.name for t in gateway_run._live_nondaemon_threads()}
        assert "residue-probe-thread" in names
        # Daemon threads and the current thread are excluded.
        assert threading.current_thread() not in gateway_run._live_nondaemon_threads()
    finally:
        stop.set()
        worker.join(timeout=2)


def test_describe_child_pids_lists_a_spawned_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Give /proc a moment to reflect the new child.
        deadline = time.monotonic() + 2
        pids = {}
        while time.monotonic() < deadline:
            pids = {pid: name for pid, name in gateway_run._describe_child_pids()}
            if child.pid in pids:
                break
            time.sleep(0.02)
        assert child.pid in pids, f"spawned child {child.pid} not in {pids}"
    finally:
        child.kill()
        child.wait(timeout=5)


def test_log_exit_residue_emits_debug_line_and_never_raises(caplog):
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, name="residue-log-thread", daemon=False)
    worker.start()
    try:
        with caplog.at_level(logging.DEBUG, logger=gateway_run.logger.name):
            gateway_run._log_exit_residue(75)
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "hard-exit (code 75) residue" in joined
        assert "residue-log-thread" in joined
    finally:
        stop.set()
        worker.join(timeout=2)


def test_log_exit_residue_is_silent_when_nothing_lingers(monkeypatch, caplog):
    monkeypatch.setattr(gateway_run, "_live_nondaemon_threads", lambda: [])
    monkeypatch.setattr(gateway_run, "_describe_child_pids", lambda: [])
    with caplog.at_level(logging.DEBUG, logger=gateway_run.logger.name):
        gateway_run._log_exit_residue(0)
    assert "residue" not in "\n".join(r.getMessage() for r in caplog.records)
