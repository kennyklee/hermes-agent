import json

from tools.mcp_tool_handlers import _compact_write_receipt


def _wrap(inner):
    return json.dumps({"result": json.dumps(inner)})


def test_gbrain_capture_receipt_is_compacted():
    outcome = {"slug": "inbox/x", "status": "created_or_updated", "chunks": 1, "source_id": "kenny-brain",
               "auto_links": {"hint": "long " * 200}, "embedding_state": "queued"}
    inner = {**outcome, "outcome": outcome, "write_request": {**outcome, "outcome": outcome},
             "request_id": "r1", "state": "committed"}
    out = _compact_write_receipt(_wrap(inner))
    text = json.loads(out)["result"]
    assert text.startswith("saved ✅")
    assert '"slug": "inbox/x"' in text and '"state": "committed"' in text and '"request_id": "r1"' in text
    assert "hint" not in text and len(out) < 400


def test_reads_errors_and_plain_text_pass_through():
    read = _wrap([{"slug": "a", "chunk_text": "x"}])
    err = _wrap({"error": "write_pending", "request_id": "r", "slug": "a"})
    no_slug = _wrap({"request_id": "r", "state": "running"})
    for raw in (read, err, no_slug, json.dumps({"result": "plain text"}), "not json",
                json.dumps({"result": "{}", "structuredContent": {"a": 1}})):
        assert _compact_write_receipt(raw) == raw
