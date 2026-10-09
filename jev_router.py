"""Opt-in "Jev model router".

Ask TypeSafe Jev (POST /v1/systemone, model ``jev-latest``, a single Choice question) which
Claude tier is the *cheapest* that will still do a task well on the first try, and route the
turn/job/task to it. Everything here is default-off and fully fail-safe: any error, timeout,
missing key, or disabled config falls back to the caller's configured default and never blocks
the turn/job/task.

The Choice question text / criteria are reused verbatim from the proven prototype at
``~/henry-ops/jev-model-picker.py`` (13/13 on its test set).

Config (all under ``routing.jev`` in config.yaml, all default off):
  enabled                  bool   master switch
  tiers                    map    {haiku|sonnet|opus: <anthropic model id>}
  min_confidence           float  below this, round the pick UP one tier (default 0.70)
  chat_mode                str    "upgrade_only" (default) | "off" -- gateway chat sessions only
  chat_floor               str    tier a gateway session starts on / resets to (default "sonnet")
  chat_reset_idle_minutes  int    idle period after which a session resets to chat_floor (default 30)
  scope                    list   subset of ["gateway", "cron", "delegate"]

Three policies, one engine (``_decide``):
  * gateway (chat):  sticky per session, upgrade-only, never downgrades mid-session (preserves
    prompt-cache warmth). Jev is asked before EVERY turn using the message + last ~2 turns.
    Resets to ``chat_floor`` after ``chat_reset_idle_minutes`` idle, or via ``reset_session()``
    (wired to /new and /reset).
  * cron / delegate: one-shot per job/task, picked from the job's/task's full prompt or
    instructions, free to move up or down (each run starts fresh -- no warm cache to lose).
    Only jobs/tasks that pin no explicit model and run on an Anthropic-or-unset provider.

Key: env ``TYPESAFE_API_KEY`` or file ``~/.hermes/config/typesafe-api-key`` (never logged).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_API_URL = "https://api.typesafe.ai/v1/systemone"
_TIMEOUT_S = 1.5
_TIER_ORDER = ("haiku", "sonnet", "opus")

# Provider strings that mean "Anthropic / Claude" (unset also counts as eligible).
_ANTHROPIC_ALIASES = {"anthropic", "claude", "claude-oauth", "claude-code"}

_DEFAULT_TIERS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

# Reused verbatim from the jev-model-picker.py prototype (do not reword -- it is tuned).
_QUESTION = {
    "type": "choice",
    "instructions": "An AI assistant must complete `task`. Which is the cheapest model tier that will still do this task well on the first try?",
    "criteria": {
        "haiku": {"covers": "Simple lookups, status checks, short factual answers, reformatting, yes/no checks, short summaries of one item, filling a template.",
                  "not_for": "Anything needing judgment across many sources, writing code, or careful drafting for important people."},
        "sonnet": {"covers": "Normal work: summarising several emails or meetings, drafting routine replies, daily briefings, research with a few sources, small code edits, running multi-step tool workflows.",
                   "not_for": "Hard debugging, architecture, high-stakes strategy or negotiation, long autonomous coding."},
        "opus": {"covers": "Hard reasoning: debugging complex systems, multi-file code changes, architecture and upgrade risk reviews, high-stakes strategy, investor or partner negotiation drafts, ambiguous problems with many trade-offs.",
                 "not_for": "Routine or simple tasks."},
    },
}

# Sticky per active gateway session: session_key -> (tier, last_seen_monotonic). A pick is
# compared against this every turn; the session only moves up, never down, and resets to
# chat_floor once idle longer than chat_reset_idle_minutes. Process-local; a restart resets,
# which is correct -- the prompt cache is cold after a restart anyway.
_session_picks: dict[str, tuple[str, float]] = {}


@dataclass
class JevConfig:
    enabled: bool
    tiers: dict
    min_confidence: float
    chat_mode: str
    chat_floor: str
    chat_reset_idle_minutes: float
    scope: list
    timeout: float


@dataclass
class JevDecision:
    """Outcome of a routing decision; ``model`` is always safe to use."""

    model: str
    tier: Optional[str]       # applied tier (None only when no tier concept applies)
    confidence: Optional[float]
    rounded_up: bool
    routed: bool  # True only when a Jev round-trip actually succeeded


def _read_config(cfg: Optional[dict]) -> JevConfig:
    r = ((cfg or {}).get("routing") or {}).get("jev") or {}
    if not isinstance(r, dict):
        r = {}
    tiers = r.get("tiers")
    scope = r.get("scope")
    try:
        min_conf = float(r.get("min_confidence", 0.70))
    except (TypeError, ValueError):
        min_conf = 0.70
    try:
        timeout = float(r.get("timeout_s", _TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = _TIMEOUT_S
    try:
        idle_minutes = float(r.get("chat_reset_idle_minutes", 30))
    except (TypeError, ValueError):
        idle_minutes = 30.0
    chat_floor = r.get("chat_floor") or "sonnet"
    if chat_floor not in _TIER_ORDER:
        chat_floor = "sonnet"
    chat_mode = r.get("chat_mode") or "upgrade_only"
    if chat_mode not in ("upgrade_only", "off"):
        chat_mode = "upgrade_only"
    return JevConfig(
        enabled=bool(r.get("enabled", False)),
        tiers=dict(tiers) if isinstance(tiers, dict) and tiers else dict(_DEFAULT_TIERS),
        min_confidence=min_conf,
        chat_mode=chat_mode,
        chat_floor=chat_floor,
        chat_reset_idle_minutes=idle_minutes,
        scope=list(scope) if isinstance(scope, list) else ["gateway", "cron", "delegate"],
        timeout=timeout,
    )


def _load_api_key() -> Optional[str]:
    """TYPESAFE_API_KEY env wins; else ~/.hermes/config/typesafe-api-key. Never logged."""
    env = os.environ.get("TYPESAFE_API_KEY")
    if env and env.strip():
        return env.strip()
    try:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "config" / "typesafe-api-key"
        if path.exists():
            key = path.read_text(encoding="utf-8-sig").strip()
            return key or None
    except Exception:
        return None
    return None


def _normalize_provider(provider: Optional[str]) -> str:
    return str(provider or "").strip().lower()


def _tier_rank(tier: Optional[str]) -> int:
    return _TIER_ORDER.index(tier) if tier in _TIER_ORDER else -1


def _pinned_and_provider_ok(pinned_model: Optional[str], provider: Optional[str]) -> bool:
    """Eligible only when no explicit model is pinned AND the provider is Anthropic or unset."""
    if isinstance(pinned_model, str) and pinned_model.strip():
        return False
    p = _normalize_provider(provider)
    return p == "" or p in _ANTHROPIC_ALIASES


def cron_eligible(job: dict, effective_provider: Optional[str] = None) -> bool:
    """A cron job is routable only when it pins no explicit model AND its effective provider
    (its own pin, else the cron fleet default passed as ``effective_provider``) is Anthropic or
    unset. Pinned models / non-Anthropic providers are left untouched."""
    if not isinstance(job, dict):
        return False
    provider = job.get("provider") if job.get("provider") else effective_provider
    return _pinned_and_provider_ok(job.get("model"), provider)


def delegate_eligible(pinned_model: Optional[str], provider: Optional[str]) -> bool:
    """A delegated sub-agent task is routable only when it pins no explicit model AND its
    provider is Anthropic or unset."""
    return _pinned_and_provider_ok(pinned_model, provider)


def _apply_round_up(tier: str, confidence: float, min_confidence: float) -> tuple[str, bool]:
    """Below ``min_confidence`` the pick is rounded UP one tier (never past opus)."""
    if tier in _TIER_ORDER and confidence < min_confidence:
        idx = _TIER_ORDER.index(tier)
        if idx < len(_TIER_ORDER) - 1:
            return _TIER_ORDER[idx + 1], True
    return tier, False


def _map_tier_to_model(tier: Optional[str], tiers: dict, default_model: str) -> str:
    """Tier name -> configured anthropic model id; unknown tier falls back to default."""
    model = tiers.get(tier) if isinstance(tiers, dict) and tier else None
    return model if isinstance(model, str) and model.strip() else default_model


def _ask_jev(key: str, task: str, timeout: float) -> tuple[str, float]:
    """One Choice question for one task. Mirrors the prototype's request shape exactly.
    Returns (tier, confidence). Raises on any HTTP / parse error (caller fail-safes)."""
    body = {
        "model": "jev-latest",
        "state": {"tasks": [task]},
        "questions": {
            "t0": {**_QUESTION, "instructions": {"task": "`tasks[0]`", "question": _QUESTION["instructions"]}},
        },
    }
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https endpoint)
        res = json.load(resp)
    answer = res["answers"]["t0"]
    return str(answer["choice"]), float(answer["confidence"])


def _append_jsonl(record: dict) -> None:
    """Append one audit line to ~/.hermes/logs/jev-router.jsonl. No message/prompt text, ever."""
    try:
        from hermes_constants import get_hermes_home

        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "jev-router.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # audit logging must never break a turn/job/task
        logger.debug("jev-router: jsonl append failed: %s", exc)


def _decide(
    *,
    task: str,
    jev: JevConfig,
    scope: str,
    id_for_log: str,
    default_model: str,
    prev_tier: Optional[str],
    allow_downgrade: bool,
) -> JevDecision:
    """Core engine shared by all three scopes: ask Jev once, round up low-confidence picks,
    then apply either upgrade-only (gateway, ``allow_downgrade=False``) or free up/down
    (cron/delegate, ``allow_downgrade=True``) policy. Logs exactly one INFO line and appends
    exactly one JSONL record, in the required format, on every call -- success or fail-safe.

    ``default_model`` is what gets returned (and logged as ``applied``'s mapped model) on any
    error/timeout/missing-key path: for gateway that is the session's *current* tier's model
    (never the global config default), for cron/delegate it is the job's/task's configured
    default. ``prev_tier`` is the tier to fall back to / compare against (None for cron/delegate,
    which have no stickiness)."""
    key = _load_api_key()
    t0 = time.monotonic()
    pick_tier: Optional[str] = None
    confidence: Optional[float] = None
    rounded_up = False
    applied_tier = prev_tier
    model = default_model
    routed = False

    if not key:
        logger.warning(
            "jev-router: no API key (set TYPESAFE_API_KEY or ~/.hermes/config/typesafe-api-key); "
            "using default model"
        )
    else:
        try:
            pick_tier, confidence = _ask_jev(key, task or "", jev.timeout)
            rounded_tier, rounded_up = _apply_round_up(pick_tier, confidence, jev.min_confidence)
            if allow_downgrade or prev_tier is None or _tier_rank(rounded_tier) > _tier_rank(prev_tier):
                applied_tier = rounded_tier
            else:
                applied_tier = prev_tier
            model = _map_tier_to_model(applied_tier, jev.tiers, default_model)
            routed = True
        except Exception as exc:
            logger.warning("jev-router: pick failed (%s); using default model %s", exc, default_model)

    ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "jev-router: scope=%s id=%s pick=%s conf=%s rounded_up=%s applied=%s ms=%d",
        scope, id_for_log, pick_tier or "-",
        f"{confidence:.2f}" if confidence is not None else "-",
        rounded_up, applied_tier or "-", ms,
    )
    _append_jsonl({
        "ts": time.time(),
        "scope": scope,
        "id": id_for_log,
        "pick": pick_tier,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "rounded_up": rounded_up,
        "applied_model": model,
        "prev_tier": prev_tier,
    })
    return JevDecision(model=model, tier=applied_tier, confidence=confidence, rounded_up=rounded_up, routed=routed)


def reset_session(session_key: Optional[str]) -> None:
    """Drop a session's sticky tier (wired to /new and /reset): the next turn starts fresh at
    chat_floor instead of keeping whatever tier the session had upgraded to."""
    if session_key:
        _session_picks.pop(session_key, None)


def route_gateway_turn(
    *,
    session_key: Optional[str],
    message: Optional[str],
    history: Any,
    cfg: Optional[dict],
    default_model: str,
    provider: Optional[str],
    has_model_override: bool,
) -> str:
    """Pick a Claude tier for a gateway chat turn; return the model string to use.

    "upgrade_only": a session starts at ``chat_floor``. Jev is asked before EVERY turn using
    the message + last ~2 turns; if its pick outranks the session's current tier, the session
    switches up and stays there (sticky) -- it never downgrades mid-session. The session resets
    to the floor after ``chat_reset_idle_minutes`` idle (or via ``reset_session()`` on /new,
    /reset). Never routes a session the user pinned with ``/model``, and only routes sessions
    already destined for Anthropic. Fail-safe to the session's current tier."""
    jev = _read_config(cfg)
    if not jev.enabled or "gateway" not in jev.scope or jev.chat_mode != "upgrade_only":
        return default_model
    if has_model_override:
        return default_model
    if _normalize_provider(provider) not in _ANTHROPIC_ALIASES:
        return default_model

    now = time.monotonic()
    prev = _session_picks.get(session_key) if session_key else None
    idle_limit_s = max(0.0, jev.chat_reset_idle_minutes) * 60
    if prev is not None and (now - prev[1]) <= idle_limit_s:
        current_tier = prev[0]
    else:
        current_tier = jev.chat_floor

    current_model = _map_tier_to_model(current_tier, jev.tiers, default_model)
    task = _gateway_task_state(message, history)
    decision = _decide(
        task=task, jev=jev, scope="gateway", id_for_log=session_key or "*",
        default_model=current_model, prev_tier=current_tier, allow_downgrade=False,
    )
    if session_key:
        _session_picks[session_key] = (decision.tier or current_tier, now)
    return decision.model


def route_cron_job(
    *,
    job: dict,
    cfg: Optional[dict],
    prompt: Optional[str],
    default_model: str,
    cron_default_provider: Optional[str] = None,
) -> str:
    """Pick a Claude tier for an eligible cron job; return the model string to use.

    Only routes jobs that pin no model and whose effective provider (its own pin, else
    ``cron_default_provider``) is Anthropic or unset. State is the job's full prompt; the pick is
    free to move up or down. Fail-safe to ``default_model``."""
    jev = _read_config(cfg)
    if not jev.enabled or "cron" not in jev.scope:
        return default_model
    if not cron_eligible(job, effective_provider=cron_default_provider):
        return default_model
    job_id = str(job.get("id") or "cron")
    decision = _decide(
        task=prompt or "", jev=jev, scope="cron", id_for_log=job_id,
        default_model=default_model, prev_tier=None, allow_downgrade=True,
    )
    return decision.model


def route_delegate_task(
    *,
    task_id: Optional[str],
    instructions: Optional[str],
    cfg: Optional[dict],
    default_model: str,
    provider: Optional[str],
    pinned_model: Optional[str],
) -> str:
    """Pick a Claude tier for a delegated sub-agent / background task; return the model string.

    Only routes tasks that pin no explicit model and whose provider is Anthropic or unset. State
    is the task's full instructions/goal; the pick is free to move up or down (each task starts
    fresh -- no prompt cache to lose). Fail-safe to ``default_model``."""
    jev = _read_config(cfg)
    if not jev.enabled or "delegate" not in jev.scope:
        return default_model
    if not delegate_eligible(pinned_model, provider):
        return default_model
    decision = _decide(
        task=instructions or "", jev=jev, scope="delegate", id_for_log=str(task_id or "task"),
        default_model=default_model, prev_tier=None, allow_downgrade=True,
    )
    return decision.model


def _msg_text(msg: Any) -> str:
    """Best-effort plain text from a transcript message (dict with str/blocks content)."""
    if isinstance(msg, str):
        return msg
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return ""


def _gateway_task_state(message: Optional[str], history: Any) -> str:
    """Task state = the last ~2 turns of context plus the user's current message."""
    parts = []
    if isinstance(history, list):
        for msg in history[-4:]:  # ~2 user/assistant turns
            text = _msg_text(msg).strip()
            if text:
                parts.append(text)
    if message:
        parts.append(message)
    return "\n".join(parts).strip() or (message or "")
