"""The first turn of a session with history after a gateway (re)start gets a one-shot restart note."""
import time
from types import SimpleNamespace

from gateway.run_turn import GatewayTurnMixin  # noqa: F401  (mixin hosting _hmwa_* helpers)


def _runner(started=None):
    cls = GatewayTurnMixin
    r = SimpleNamespace(_startup_time=started if started is not None else time.time())
    r.__dict__.setdefault("_restart_note_seen", set())
    r._hmwa_restart_sidecar_note = cls._hmwa_restart_sidecar_note.__get__(r)
    return r


def test_note_added_once_for_session_with_history():
    r = _runner(time.time() - 60)
    notes = []
    r._hmwa_restart_sidecar_note("k1", [{"role": "user", "content": "hi"}], notes)
    assert len(notes) == 1
    assert "gateway restarted at" in notes[0] and "/reset" in notes[0]
    r._hmwa_restart_sidecar_note("k1", [{"role": "user", "content": "hi"}], notes)
    assert len(notes) == 1  # one-shot per session per process


def test_no_note_for_fresh_session_or_missing_key():
    r = _runner()
    notes = []
    r._hmwa_restart_sidecar_note("k2", [], notes)
    r._hmwa_restart_sidecar_note("", [{"role": "user", "content": "x"}], notes)
    assert notes == []


def test_independent_sessions_each_get_one_note():
    r = _runner()
    notes = []
    h = [{"role": "user", "content": "x"}]
    r._hmwa_restart_sidecar_note("a", h, notes)
    r._hmwa_restart_sidecar_note("b", h, notes)
    assert len(notes) == 2
