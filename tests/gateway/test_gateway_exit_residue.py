"""FIX 2 — the hard-exit backstop logs what is still alive (non-daemon threads + direct child PIDs) so a
slow cgroup-empty on the restart path is diagnosable from logs. Purely observational: it signals nothing
and kills nothing (background terminal processes are intentionally persisted across a restart)."""

import json
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


def test_log_exit_residue_writes_to_stderr(monkeypatch, capsys):
    """The logger.* call is swallowed post-drain on the os._exit path, so the diagnostic MUST also
    land on stderr (systemd journal) — that is the fix for why it never appeared in production."""
    monkeypatch.setattr(gateway_run, "_live_nondaemon_threads", lambda: [])
    monkeypatch.setattr(gateway_run, "_describe_child_pids",
                        lambda: [(4242, "node .../mcp-remote-env-header.mjs")])
    gateway_run._log_exit_residue(75)
    err = capsys.readouterr().err
    assert "[gateway-exit-residue]" in err
    assert "hard-exit (code 75) residue" in err
    assert "4242:node" in err


def test_log_exit_residue_appends_to_exit_diag_jsonl(tmp_path, monkeypatch):
    """Structured record lands in gateway-exit-diag.log — the same durable sink as the CLI's
    _exit_diag records — so a slow cgroup-empty is forensically reconstructable after os._exit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_live_nondaemon_threads", lambda: [])
    monkeypatch.setattr(gateway_run, "_describe_child_pids",
                        lambda: [(4242, "node .../mcp-remote-env-header.mjs")])
    gateway_run._log_exit_residue(75)
    diag = tmp_path / "logs" / "gateway-exit-diag.log"
    assert diag.exists(), "exit-diag JSONL was not written"
    records = [json.loads(line) for line in diag.read_text(encoding="utf-8").splitlines() if line.strip()]
    residue = [r for r in records if r.get("tag") == "gateway.exit_residue"]
    assert residue, f"no gateway.exit_residue record in {records}"
    rec = residue[-1]
    assert rec["exit_code"] == 75
    assert rec["child_pids"] == [{"pid": 4242, "cmd": "node .../mcp-remote-env-header.mjs"}]
    assert rec["nondaemon_threads"] == []


def test_log_exit_residue_jsonl_silent_when_nothing_lingers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_live_nondaemon_threads", lambda: [])
    monkeypatch.setattr(gateway_run, "_describe_child_pids", lambda: [])
    gateway_run._log_exit_residue(0)
    diag = tmp_path / "logs" / "gateway-exit-diag.log"
    assert not diag.exists() or "gateway.exit_residue" not in diag.read_text(encoding="utf-8")
