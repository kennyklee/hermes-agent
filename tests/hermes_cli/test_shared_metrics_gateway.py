"""Invariants for gateway / cron shared metrics: bounded platform names with proven catalog provenance,
opt-in gating, and contract-valid rows."""

import json
import textwrap
import threading
import time

import pytest

from hermes_cli.observability import relay_shared_metrics as rsm
from hermes_cli.observability import shared_metrics_catalog as catalog
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_gateway as smg


@pytest.fixture
def marks(monkeypatch):
    got = []
    monkeypatch.setattr(rsm, "enabled", lambda: True)
    monkeypatch.setattr(rsm, "record_process_mark", lambda mark, data: got.append((mark, dict(data))))

    def read():
        smg.drain()
        return [(contract._DECISION_MARK_METRICS[mark], data) for mark, data in got]

    return read


def _install_platform_plugin(home, monkeypatch, *, dir_name, platform, catalog_name=None):
    """A user plugin under $HERMES_HOME/plugins registering ``platform``; ``catalog_name`` writes the
    installer-owned record a plugin-catalog install leaves (a URL install writes none)."""
    from gateway.platform_registry import PlatformEntry, platform_registry

    plugin_dir = home / "plugins" / dir_name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "adapter.py").write_text(textwrap.dedent("""
        def make_adapter(config):
            return None
    """), encoding="utf-8")
    # Anything inside the tree is the repo's to write: it must never prove a catalog install.
    (plugin_dir / ".hermes-catalog.json").write_text(json.dumps({"catalog_name": "sendblue"}), encoding="utf-8")
    namespace = {}
    exec(compile((plugin_dir / "adapter.py").read_text(), str(plugin_dir / "adapter.py"), "exec"), namespace)
    if catalog_name:
        (home / "plugins" / ".install-metadata.json").write_text(json.dumps({
            dir_name: {"source": "https://example.invalid/repo", "catalog": {"name": catalog_name, "tier": "official"}},
        }), encoding="utf-8")
    platform_registry.register(PlatformEntry(
        name=platform, label=platform, adapter_factory=namespace["make_adapter"], check_fn=lambda: True,
        source="plugin", plugin_name="sendblue"))
    monkeypatch.setattr(catalog, "catalog_platform_names", lambda: frozenset({"sendblue", "vk-platform"}))
    catalog._catalog_platform_owner.cache_clear()
    return lambda: platform_registry.unregister(platform)


def test_platform_names_are_published_or_proven_by_the_installer_record(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    cleanup = [
        _install_platform_plugin(home, monkeypatch, dir_name="vk", platform="vk", catalog_name="vk-platform"),
        # A URL install claiming a catalog name everywhere it can (tree sidecar, manifest name) stays anonymous.
        _install_platform_plugin(tmp_path / "other", monkeypatch, dir_name="x", platform="sneaky"),
    ]
    try:
        assert contract.adapter_platform("telegram") == "telegram"
        assert contract.adapter_platform("irc") == "irc"  # bundled plugins/platforms/irc
        assert contract.adapter_platform("vk") == "vk-platform"
        assert contract.adapter_platform("sneaky") == "plugin"
        assert contract.adapter_platform("my-private-bridge") == "plugin"
        assert contract.gateway_platform({"platform": "vk", "surface": "gateway"}, "gateway") == "vk-platform"
        assert "vk-platform" in contract.GATEWAY_PLATFORMS
    finally:
        for undo in cleanup:
            undo()
        catalog._catalog_platform_owner.cache_clear()


def test_every_gateway_row_is_contract_valid_and_disabled_config_records_nothing(marks, monkeypatch):
    class Adapter:
        platform = "telegram"
        fatal_error_code = "telegram_auth_error"

    smg.record_platform_connect(Adapter(), "telegram", is_reconnect=False, ok=False, exc=TimeoutError("x"))
    smg.record_platform_disconnect(Adapter())
    smg.record_cron_missed({"deliver": "telegram:-100123"})
    rows = marks()
    assert rows == [
        ("hermes.platform.health", {"error_class": "network", "event": "connect_failed", "platform": "telegram"}),
        ("hermes.platform.health", {"error_class": "auth", "event": "disconnect", "platform": "telegram"}),
        ("hermes.cron.run", {"delivery_kind": "platform", "duration_bucket": "lt_1s", "outcome": "missed"}),
    ]
    assert all(contract.counter_dimensions_are_valid(name, dims) for name, dims in rows)

    monkeypatch.setattr(rsm, "enabled", lambda: False)
    smg.record_platform_disconnect(Adapter())
    assert len(marks()) == 3


def test_every_gateway_adapter_platform_is_named_and_accepted_by_contract_and_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from pathlib import Path

    from gateway.config import Platform

    schema = json.loads((Path(contract.__file__).parent / "schemas/hermes.shared_metrics.v3.schema.json").read_text())
    rows = {
        "platform_health_counter": ("hermes.platform.health", {"error_class": "network", "event": "disconnect"}),
        "platform_delivery_counter": ("hermes.platform.delivery", {"failure_class": "none", "outcome": "sent"}),
        "reply_latency_counter": ("hermes.gateway.reply_latency", {"first_response_bucket": "lt_2s"}),
    }
    for member in Platform:  # relay and msgraph_webhook are gateway adapters with no setup-wizard entry
        name = contract.adapter_platform(member)
        assert name == member.value
        for definition, (metric, dims) in rows.items():
            dims = {**dims, "platform": name}
            assert contract.counter_dimensions_are_valid(metric, dims)
            jsonschema.validate({"name": metric, "type": "counter", "dimensions": dims, "value": 1},
                                {"$defs": schema["$defs"], "$ref": f"#/$defs/{definition}"})


# ---- worker teardown at gateway stop (exit residue / interpreter-finalize hygiene) ----

def _metrics_threads():
    return [t for t in threading.enumerate() if t.name.startswith("hermes-gateway-metrics")]


def _wait_no_metrics_thread(timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _metrics_threads():
        time.sleep(0.02)
    return not _metrics_threads()


@pytest.fixture
def fresh_metrics_worker():
    """Isolate the module-global metrics executor around a test and tear it down afterward."""
    def _reset():
        with smg._executor_lock:
            executor, smg._executor, smg._shutdown = smg._executor, None, False
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    _reset()
    yield smg
    _reset()


def test_shutdown_flushes_queued_work_then_stops_the_worker(fresh_metrics_worker):
    """A recording already queued ahead of the FIFO flush runs; the non-daemon worker then exits."""
    recorded = []
    fresh_metrics_worker._submit(recorded.append, "row")
    assert _metrics_threads(), "a submit should have started the worker thread"

    fresh_metrics_worker.shutdown()

    assert recorded == ["row"], "shutdown dropped an already-queued recording"
    assert fresh_metrics_worker._executor is None
    assert _wait_no_metrics_thread(), "the non-daemon metrics worker survived shutdown"


def test_submit_after_shutdown_is_dropped_not_resurrected(fresh_metrics_worker):
    """Once stopped, a late metric is dropped rather than respawning the worker we just reaped."""
    fresh_metrics_worker.shutdown()  # no worker started yet: just flips the flag

    assert fresh_metrics_worker._submit(lambda: None) is None
    assert fresh_metrics_worker._executor is None
    assert not _metrics_threads(), "a post-shutdown submit resurrected the worker"


def test_shutdown_is_idempotent_and_safe_with_no_worker(fresh_metrics_worker):
    fresh_metrics_worker.shutdown()  # _executor is None — must not raise
    fresh_metrics_worker.shutdown()  # idempotent
    assert fresh_metrics_worker._executor is None


def test_shutdown_is_bounded_when_a_recording_is_slow(fresh_metrics_worker):
    """A wedged recording occupying the single worker must not block teardown past the flush bound."""
    release = threading.Event()
    try:
        fresh_metrics_worker._submit(release.wait)  # occupies the one worker
        started = time.monotonic()
        fresh_metrics_worker.shutdown(flush_timeout=0.2)
        elapsed = time.monotonic() - started
        assert elapsed < 1.5, f"shutdown blocked on a slow recording: {elapsed:.2f}s"
        assert fresh_metrics_worker._executor is None
    finally:
        release.set()
