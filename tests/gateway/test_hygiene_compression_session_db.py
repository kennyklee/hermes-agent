from types import SimpleNamespace

from gateway.run import _resolve_hygiene_session_db


class FakeSessionDB:
    def create_session(self, *args, **kwargs):
        return None


def test_resolve_hygiene_session_db_unwraps_async_session_db_holder():
    sync_db = FakeSessionDB()
    async_holder = SimpleNamespace(_db=sync_db)

    assert _resolve_hygiene_session_db(async_holder) is sync_db


def test_resolve_hygiene_session_db_accepts_sync_session_db():
    sync_db = FakeSessionDB()

    assert _resolve_hygiene_session_db(sync_db) is sync_db


def test_resolve_hygiene_session_db_uses_fallback_holder_db_before_fresh_db():
    store_db = FakeSessionDB()
    created = FakeSessionDB()
    fallback_holder = SimpleNamespace(_db=store_db)

    assert (
        _resolve_hygiene_session_db(
            None,
            fallback_holder=fallback_holder,
            session_db_factory=lambda: created,
        )
        is store_db
    )


def test_resolve_hygiene_session_db_creates_db_when_gateway_holder_missing():
    created = FakeSessionDB()

    assert _resolve_hygiene_session_db(None, session_db_factory=lambda: created) is created


def test_resolve_hygiene_session_db_returns_none_when_factory_returns_none():
    assert _resolve_hygiene_session_db(None, session_db_factory=lambda: None) is None


def test_resolve_hygiene_session_db_raises_when_factory_raises_for_call_site_to_catch():
    def boom():
        raise RuntimeError("sqlite unavailable")

    try:
        _resolve_hygiene_session_db(None, session_db_factory=boom)
    except RuntimeError as exc:
        assert "sqlite unavailable" in str(exc)
    else:  # pragma: no cover - explicit assertion reads clearer than pytest import
        raise AssertionError("expected factory exception to propagate")
