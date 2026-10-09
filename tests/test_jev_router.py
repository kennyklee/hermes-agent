"""Unit tests for the opt-in Jev model router (jev_router.py). HTTP is always mocked."""
import io
import json

import pytest

import jev_router


BASE_CFG = {
    "routing": {
        "jev": {
            "enabled": True,
            "tiers": {
                "haiku": "claude-haiku-4-5-20251001",
                "sonnet": "claude-sonnet-5-5",
                "opus": "claude-opus-5-5",
            },
            "min_confidence": 0.70,
            "chat_mode": "upgrade_only",
            "chat_floor": "sonnet",
            "chat_reset_idle_minutes": 30,
            "scope": ["gateway", "cron", "delegate"],
        }
    }
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep every test off the network, off the filesystem, and with a fresh sticky cache."""
    jev_router._session_picks.clear()
    monkeypatch.setattr(jev_router, "_load_api_key", lambda: "test-key")
    monkeypatch.setattr(jev_router, "_append_jsonl", lambda record: None)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(jev_router.time, "monotonic", c)
    return c


def _answer(choice, confidence):
    """Install a fake Jev answer and return (tier, confidence) from _ask_jev."""
    return lambda key, task, timeout: (choice, confidence)


# --- tier mapping -------------------------------------------------------------

def test_map_tier_to_model():
    tiers = BASE_CFG["routing"]["jev"]["tiers"]
    assert jev_router._map_tier_to_model("haiku", tiers, "d") == "claude-haiku-4-5-20251001"
    assert jev_router._map_tier_to_model("sonnet", tiers, "d") == "claude-sonnet-5-5"
    assert jev_router._map_tier_to_model("opus", tiers, "d") == "claude-opus-5-5"
    # Unknown/missing tier -> default model.
    assert jev_router._map_tier_to_model("mystery", tiers, "default-x") == "default-x"
    assert jev_router._map_tier_to_model(None, tiers, "default-x") == "default-x"


def test_tier_rank_orders_haiku_sonnet_opus():
    assert jev_router._tier_rank("haiku") < jev_router._tier_rank("sonnet") < jev_router._tier_rank("opus")
    assert jev_router._tier_rank("nonsense") == -1


# --- round-up -------------------------------------------------------------------

def test_apply_round_up():
    # Below threshold rounds up one tier.
    assert jev_router._apply_round_up("haiku", 0.5, 0.70) == ("sonnet", True)
    assert jev_router._apply_round_up("sonnet", 0.69, 0.70) == ("opus", True)
    # At/above threshold keeps the pick.
    assert jev_router._apply_round_up("haiku", 0.70, 0.70) == ("haiku", False)
    assert jev_router._apply_round_up("sonnet", 0.99, 0.70) == ("sonnet", False)
    # opus is the ceiling -- never rounds past it.
    assert jev_router._apply_round_up("opus", 0.10, 0.70) == ("opus", False)


def test_decide_rounds_up_low_confidence(monkeypatch):
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("haiku", 0.4))
    jev = jev_router._read_config(BASE_CFG)
    d = jev_router._decide(
        task="task", jev=jev, scope="cron", id_for_log="j1",
        default_model="default-model", prev_tier=None, allow_downgrade=True,
    )
    assert d.tier == "sonnet"
    assert d.rounded_up is True
    assert d.model == "claude-sonnet-5-5"


def test_decide_maps_confident_choice(monkeypatch):
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("sonnet", 0.95))
    jev = jev_router._read_config(BASE_CFG)
    d = jev_router._decide(
        task="summarise these notes", jev=jev, scope="cron", id_for_log="j1",
        default_model="default-model", prev_tier=None, allow_downgrade=True,
    )
    assert d.routed is True
    assert d.tier == "sonnet"
    assert d.model == "claude-sonnet-5-5"
    assert d.rounded_up is False


# --- fail-safe --------------------------------------------------------------

def test_decide_fails_safe_on_http_error(monkeypatch):
    def boom(key, task, timeout):
        raise TimeoutError("jev timeout")

    monkeypatch.setattr(jev_router, "_ask_jev", boom)
    jev = jev_router._read_config(BASE_CFG)
    d = jev_router._decide(
        task="task", jev=jev, scope="cron", id_for_log="j1",
        default_model="default-model", prev_tier=None, allow_downgrade=True,
    )
    assert d.routed is False
    assert d.model == "default-model"


def test_decide_fails_safe_without_key(monkeypatch):
    monkeypatch.setattr(jev_router, "_load_api_key", lambda: None)
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("opus", 0.99))  # must not be called
    jev = jev_router._read_config(BASE_CFG)
    d = jev_router._decide(
        task="task", jev=jev, scope="cron", id_for_log="j1",
        default_model="default-model", prev_tier=None, allow_downgrade=True,
    )
    assert d.routed is False
    assert d.model == "default-model"


def test_gateway_fails_safe_to_current_session_tier(monkeypatch, clock):
    """A Jev timeout mid-session must keep the session's *current* tier, not some unrelated
    global default -- the spec's "chat: current tier" fail-safe."""
    answers = iter([("opus", 0.95)])

    def ask(key, task, timeout):
        return next(answers)

    monkeypatch.setattr(jev_router, "_ask_jev", ask)
    args = dict(session_key="s1", history=None, cfg=BASE_CFG,
                default_model="config-default-model", provider="anthropic", has_model_override=False)
    first = jev_router.route_gateway_turn(message="hard debugging task", **args)
    assert first == "claude-opus-5-5"

    def boom(key, task, timeout):
        raise TimeoutError("jev timeout")

    monkeypatch.setattr(jev_router, "_ask_jev", boom)
    second = jev_router.route_gateway_turn(message="another turn", **args)
    assert second == "claude-opus-5-5"  # kept the session's current (upgraded) tier, not the floor


def test_gateway_disabled_returns_default(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("opus", 0.9))
    cfg = {"routing": {"jev": {"enabled": False}}}
    out = jev_router.route_gateway_turn(
        session_key="s1", message="hi", history=None, cfg=cfg,
        default_model="default-model", provider="anthropic", has_model_override=False,
    )
    assert out == "default-model"
    assert called == []


def test_gateway_chat_mode_off_returns_default(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("opus", 0.9))
    cfg = {"routing": {"jev": {"enabled": True, "chat_mode": "off", "scope": ["gateway"]}}}
    out = jev_router.route_gateway_turn(
        session_key="s1", message="hi", history=None, cfg=cfg,
        default_model="default-model", provider="anthropic", has_model_override=False,
    )
    assert out == "default-model"
    assert called == []


# --- gateway: sticky, upgrade-only, asked every turn -------------------------

def test_gateway_session_starts_at_chat_floor(monkeypatch, clock):
    """Even a confident low-tier pick on turn one doesn't go below chat_floor."""
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("haiku", 0.99))
    out = jev_router.route_gateway_turn(
        session_key="s1", message="quick status check", history=None, cfg=BASE_CFG,
        default_model="default-model", provider="anthropic", has_model_override=False,
    )
    assert out == "claude-sonnet-5-5"  # chat_floor, never downgraded below it


def test_gateway_upgrades_and_stays_sticky_never_downgrades(monkeypatch, clock):
    """Jev is asked every turn; a later low-tier pick must NOT pull the session back down."""
    answers = iter([("opus", 0.95), ("haiku", 0.99), ("haiku", 0.99)])
    calls = []

    def ask(key, task, timeout):
        calls.append(task)
        return next(answers)

    monkeypatch.setattr(jev_router, "_ask_jev", ask)
    args = dict(session_key="s1", history=None, cfg=BASE_CFG,
                default_model="default-model", provider="anthropic", has_model_override=False)

    first = jev_router.route_gateway_turn(message="hard debugging task", **args)
    assert first == "claude-opus-5-5"

    clock.t += 60  # well within the idle window
    second = jev_router.route_gateway_turn(message="ok now something trivial", **args)
    assert second == "claude-opus-5-5"  # Jev said haiku this turn, but never downgrade

    clock.t += 60
    third = jev_router.route_gateway_turn(message="still trivial", **args)
    assert third == "claude-opus-5-5"

    # Jev was actually asked all three turns (every-turn contract), not just once.
    assert len(calls) == 3


def test_gateway_upgrade_mid_session_then_sticky(monkeypatch, clock):
    answers = iter([("sonnet", 0.95), ("opus", 0.95), ("sonnet", 0.95)])
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: next(answers))
    args = dict(session_key="s1", history=None, cfg=BASE_CFG,
                default_model="default-model", provider="anthropic", has_model_override=False)

    assert jev_router.route_gateway_turn(message="normal task", **args) == "claude-sonnet-5-5"
    assert jev_router.route_gateway_turn(message="now debug this hard", **args) == "claude-opus-5-5"
    # Third turn's pick (sonnet) is below the session's current tier (opus) -- stays at opus.
    assert jev_router.route_gateway_turn(message="back to normal", **args) == "claude-opus-5-5"


# --- idle reset ---------------------------------------------------------------

def test_idle_over_30min_resets_to_floor(monkeypatch, clock):
    answers = iter([("opus", 0.95), ("haiku", 0.95)])
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: next(answers))
    args = dict(session_key="s1", history=None, cfg=BASE_CFG,
                default_model="default-model", provider="anthropic", has_model_override=False)

    first = jev_router.route_gateway_turn(message="hard debugging task", **args)
    assert first == "claude-opus-5-5"
    # 31 minutes idle -> resets to chat_floor (sonnet), and a confident haiku pick still can't
    # go below the floor.
    clock.t += 31 * 60
    second = jev_router.route_gateway_turn(message="quick status check", **args)
    assert second == "claude-sonnet-5-5"


def test_idle_under_30min_stays_sticky(monkeypatch, clock):
    calls = []

    def ask(key, task, timeout):
        calls.append(task)
        return ("sonnet", 0.95)

    monkeypatch.setattr(jev_router, "_ask_jev", ask)
    args = dict(session_key="s1", history=None, cfg=BASE_CFG,
                default_model="default-model", provider="anthropic", has_model_override=False)

    assert jev_router.route_gateway_turn(message="hard debugging task", **args) == "claude-sonnet-5-5"
    clock.t += 29 * 60
    assert jev_router.route_gateway_turn(message="follow up", **args) == "claude-sonnet-5-5"
    assert len(calls) == 2  # still asked every turn, just didn't reset


def test_reset_session_drops_sticky_tier(monkeypatch, clock):
    answers = iter([("opus", 0.95), ("haiku", 0.95)])
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: next(answers))
    args = dict(session_key="s1", history=None, cfg=BASE_CFG,
                default_model="default-model", provider="anthropic", has_model_override=False)

    assert jev_router.route_gateway_turn(message="hard debugging task", **args) == "claude-opus-5-5"
    jev_router.reset_session("s1")  # simulates /new or /reset
    assert jev_router.route_gateway_turn(message="quick status check", **args) == "claude-sonnet-5-5"


def test_reset_session_noop_for_unknown_or_none():
    jev_router.reset_session(None)       # must not raise
    jev_router.reset_session("missing")  # must not raise


# --- override / provider gating ----------------------------------------------

def test_model_override_respected(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("opus", 0.9))
    out = jev_router.route_gateway_turn(
        session_key="s1", message="hi", history=None, cfg=BASE_CFG,
        default_model="user-picked-model", provider="anthropic", has_model_override=True,
    )
    assert out == "user-picked-model"
    assert called == []


def test_non_anthropic_provider_not_routed(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("opus", 0.9))
    out = jev_router.route_gateway_turn(
        session_key="s1", message="hi", history=None, cfg=BASE_CFG,
        default_model="gpt-model", provider="openai", has_model_override=False,
    )
    assert out == "gpt-model"
    assert called == []


# --- cron eligibility ---------------------------------------------------------

def test_cron_eligible():
    assert jev_router.cron_eligible({"id": "j1"}) is True                       # provider unset
    assert jev_router.cron_eligible({"id": "j1", "provider": "anthropic"}) is True
    assert jev_router.cron_eligible({"id": "j1", "provider": "claude"}) is True  # alias
    assert jev_router.cron_eligible({"id": "j1", "provider": "openai"}) is False
    assert jev_router.cron_eligible({"id": "j1", "model": "claude-opus-5-5"}) is False  # pinned
    assert jev_router.cron_eligible({"id": "j1", "model": "   "}) is True        # blank pin ignored


def test_cron_eligible_falls_back_to_fleet_default_provider():
    # No per-job provider pin -> the cron fleet default decides eligibility.
    assert jev_router.cron_eligible({"id": "j1"}, effective_provider="openai") is False
    assert jev_router.cron_eligible({"id": "j1"}, effective_provider="anthropic") is True
    # A job-level provider pin always wins over the fleet default.
    assert jev_router.cron_eligible({"id": "j1", "provider": "anthropic"}, effective_provider="openai") is True


def test_route_cron_job_skips_ineligible(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("haiku", 0.9))
    # Pinned model -> untouched, no Jev call.
    out = jev_router.route_cron_job(
        job={"id": "j1", "model": "claude-opus-5-5"}, cfg=BASE_CFG,
        prompt="do the thing", default_model="default-model",
    )
    assert out == "default-model"
    assert called == []


def test_route_cron_job_routes_eligible(monkeypatch):
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("haiku", 0.95))
    out = jev_router.route_cron_job(
        job={"id": "j1"}, cfg=BASE_CFG, prompt="is the service up?", default_model="default-model",
    )
    assert out == "claude-haiku-4-5-20251001"


def test_route_cron_job_disabled_scope(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("haiku", 0.9))
    cfg = {"routing": {"jev": {"enabled": True, "scope": ["gateway"]}}}  # cron not in scope
    out = jev_router.route_cron_job(
        job={"id": "j1"}, cfg=cfg, prompt="is the service up?", default_model="default-model",
    )
    assert out == "default-model"
    assert called == []


# --- delegate: full-instructions routing, downgrade allowed ------------------

def test_delegate_eligible():
    assert jev_router.delegate_eligible(None, None) is True
    assert jev_router.delegate_eligible(None, "anthropic") is True
    assert jev_router.delegate_eligible("claude-opus-5-5", None) is False  # pinned
    assert jev_router.delegate_eligible(None, "openai") is False


def test_route_delegate_task_allows_downgrade(monkeypatch):
    """Delegated tasks have no prompt cache to protect -- a confident low-tier pick must be
    honored even when the parent/default model is a bigger tier."""
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("haiku", 0.95))
    out = jev_router.route_delegate_task(
        task_id="t1", instructions="What time is my next meeting?", cfg=BASE_CFG,
        default_model="claude-opus-5-5", provider="anthropic", pinned_model=None,
    )
    assert out == "claude-haiku-4-5-20251001"


def test_route_delegate_task_allows_upgrade(monkeypatch):
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("opus", 0.95))
    out = jev_router.route_delegate_task(
        task_id="t1", instructions="Debug this gnarly multi-file regression with tests",
        cfg=BASE_CFG, default_model="claude-haiku-4-5-20251001", provider="anthropic", pinned_model=None,
    )
    assert out == "claude-opus-5-5"


def test_route_delegate_task_skips_pinned_model(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("haiku", 0.9))
    out = jev_router.route_delegate_task(
        task_id="t1", instructions="anything", cfg=BASE_CFG,
        default_model="claude-opus-5-5", provider="anthropic", pinned_model="claude-opus-5-5",
    )
    assert out == "claude-opus-5-5"
    assert called == []


def test_route_delegate_task_skips_non_anthropic_provider(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("haiku", 0.9))
    out = jev_router.route_delegate_task(
        task_id="t1", instructions="anything", cfg=BASE_CFG,
        default_model="gpt-5", provider="openai", pinned_model=None,
    )
    assert out == "gpt-5"
    assert called == []


def test_route_delegate_task_disabled_scope(monkeypatch):
    called = []
    monkeypatch.setattr(jev_router, "_ask_jev", lambda *a: called.append(1) or ("haiku", 0.9))
    cfg = {"routing": {"jev": {"enabled": True, "scope": ["gateway", "cron"]}}}  # delegate not in scope
    out = jev_router.route_delegate_task(
        task_id="t1", instructions="anything", cfg=cfg,
        default_model="claude-opus-5-5", provider="anthropic", pinned_model=None,
    )
    assert out == "claude-opus-5-5"
    assert called == []


# --- HTTP layer (urllib mocked) -----------------------------------------------

def test_ask_jev_parses_response(monkeypatch):
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode())
        return _Resp(json.dumps({"answers": {"t0": {"choice": "sonnet", "confidence": 0.82}}}).encode())

    monkeypatch.setattr(jev_router.urllib.request, "urlopen", fake_urlopen)
    tier, conf = jev_router._ask_jev("secret-key", "summarise notes", 1.5)
    assert (tier, conf) == ("sonnet", 0.82)
    assert captured["url"] == jev_router._API_URL
    assert captured["timeout"] == 1.5
    assert captured["auth"] == "Bearer secret-key"
    # Request shape mirrors the proven prototype.
    assert captured["body"]["model"] == "jev-latest"
    assert captured["body"]["state"]["tasks"] == ["summarise notes"]
    assert captured["body"]["questions"]["t0"]["type"] == "choice"


def test_read_config_defaults_when_missing():
    jev = jev_router._read_config({})
    assert jev.enabled is False
    assert jev.min_confidence == 0.70
    assert jev.chat_mode == "upgrade_only"
    assert jev.chat_floor == "sonnet"
    assert jev.chat_reset_idle_minutes == 30
    assert jev.scope == ["gateway", "cron", "delegate"]
    assert jev.tiers["haiku"] == "claude-haiku-4-5-20251001"


def test_read_config_rejects_invalid_chat_mode_and_floor():
    cfg = {"routing": {"jev": {"chat_mode": "bogus", "chat_floor": "bogus"}}}
    jev = jev_router._read_config(cfg)
    assert jev.chat_mode == "upgrade_only"
    assert jev.chat_floor == "sonnet"


# --- no secrets / message text in logs or JSONL -------------------------------

def test_jsonl_record_excludes_message_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(jev_router, "_append_jsonl", lambda record: captured.update(record))
    monkeypatch.setattr(jev_router, "_ask_jev", _answer("sonnet", 0.9))
    jev = jev_router._read_config(BASE_CFG)
    jev_router._decide(
        task="this is secret user content that must never be logged", jev=jev, scope="cron",
        id_for_log="j1", default_model="default-model", prev_tier=None, allow_downgrade=True,
    )
    blob = json.dumps(captured)
    assert "secret user content" not in blob
    assert set(captured) == {"ts", "scope", "id", "pick", "confidence", "rounded_up", "applied_model", "prev_tier"}
