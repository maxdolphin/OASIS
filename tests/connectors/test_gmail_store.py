import os
import tempfile

import pytest

from src.connectors.gmail_store import GmailInteractionStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GmailInteractionStore(db_path=path)
    yield s
    os.remove(path)


def _row(src, dst, ts, kind="to", thread="t1", size=100, so="/Sales", do="/IT"):
    return {
        "src_email": src, "dst_email": dst, "recipient_kind": kind,
        "ts_utc": ts, "thread_id": thread, "size_bytes": size,
        "src_orgunit": so, "dst_orgunit": do,
    }


def test_insert_and_query_all(store):
    rows = [_row("a@x.com", "b@x.com", 1000), _row("a@x.com", "c@x.com", 2000)]
    assert store.insert_rows("x.com", "run1", rows) == 2
    got = store.query_window("x.com", start_ts=0, end_ts=9999)
    assert len(got) == 2
    assert {r["dst_email"] for r in got} == {"b@x.com", "c@x.com"}


def test_query_window_filters_by_time(store):
    store.insert_rows("x.com", "run1", [
        _row("a@x.com", "b@x.com", 1000),
        _row("a@x.com", "b@x.com", 5000),
    ])
    got = store.query_window("x.com", start_ts=3000, end_ts=9999)
    assert len(got) == 1
    assert got[0]["ts_utc"] == 5000


def test_query_window_scopes_by_org(store):
    store.insert_rows("x.com", "run1", [_row("a@x.com", "b@x.com", 1000)])
    store.insert_rows("y.com", "run2", [_row("a@y.com", "b@y.com", 1000)])
    assert len(store.query_window("x.com", 0, 9999)) == 1
    assert len(store.query_window("y.com", 0, 9999)) == 1


def test_insert_empty_returns_zero(store):
    assert store.insert_rows("x.com", "run1", []) == 0


def test_schema_has_no_content_columns(store):
    cols = store.column_names()
    forbidden = {"subject", "body", "snippet", "content", "text"}
    assert forbidden.isdisjoint(cols), f"metadata-only violated: {cols & forbidden}"
