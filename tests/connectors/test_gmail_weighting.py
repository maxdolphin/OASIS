import math

import pytest

from src.connectors.gmail_weighting import build_flow_matrix
from src.network_ingestion import NetworkIngestionError

DAY = 86400
WEEK = 7 * DAY
NOW = 1_000_000_000  # fixed reference; never read from the clock

ORG = {"a@x.com", "b@x.com", "c@x.com"}


def _row(src, dst, ts, kind="to", so="/Sales", do="/IT"):
    return {"src_email": src, "dst_email": dst, "recipient_kind": kind,
            "ts_utc": ts, "thread_id": "t", "size_bytes": 100,
            "src_orgunit": so, "dst_orgunit": do}


def _weight(result_pair, src, dst):
    parsed, _ = result_pair
    i = parsed.node_names.index(src)
    j = parsed.node_names.index(dst)
    return parsed.flow_matrix[i][j]


def test_decay_half_life():
    # One fresh message vs one message exactly half_life old.
    rows = [_row("a@x.com", "b@x.com", NOW),
            _row("a@x.com", "c@x.com", NOW - 30 * DAY)]
    res = build_flow_matrix(rows, org_users=ORG, now_utc=NOW,
                            window_seconds=365 * DAY, half_life_seconds=30 * DAY,
                            beta=0.0, granularity="individual")
    fresh = _weight(res, "a@x.com", "b@x.com")
    aged = _weight(res, "a@x.com", "c@x.com")
    assert math.isclose(aged, fresh * 0.5, rel_tol=1e-6)


def test_sustained_rewards_distinct_weeks():
    # Pair A->B: 2 messages in the SAME week. Pair A->C: 2 messages in DIFFERENT weeks.
    # With beta>0 and decay off (half_life huge), A->C outweighs A->B.
    big = 10 ** 9
    rows = [
        _row("a@x.com", "b@x.com", NOW - 1 * DAY),
        _row("a@x.com", "b@x.com", NOW - 2 * DAY),
        _row("a@x.com", "c@x.com", NOW - 1 * DAY),
        _row("a@x.com", "c@x.com", NOW - 2 * WEEK),
    ]
    res = build_flow_matrix(rows, org_users=ORG, now_utc=NOW,
                            window_seconds=365 * DAY, half_life_seconds=big,
                            beta=1.0, granularity="individual")
    assert _weight(res, "a@x.com", "c@x.com") > _weight(res, "a@x.com", "b@x.com")


def test_department_granularity_sums_individual_flows():
    # Two senders in /Sales both email /IT; department matrix aggregates them.
    org = {"a@x.com", "b@x.com", "z@x.com"}
    rows = [
        _row("a@x.com", "z@x.com", NOW, so="/Sales", do="/IT"),
        _row("b@x.com", "z@x.com", NOW, so="/Sales", do="/IT"),
    ]
    res = build_flow_matrix(rows, org_users=org, now_utc=NOW,
                            window_seconds=365 * DAY, half_life_seconds=10 ** 9,
                            beta=0.0, granularity="department")
    parsed, _ = res
    assert set(parsed.node_names) == {"Sales", "IT"}
    i = parsed.node_names.index("Sales")
    j = parsed.node_names.index("IT")
    assert math.isclose(parsed.flow_matrix[i][j], 2.0, rel_tol=1e-6)


def test_external_recipients_dropped_and_counted():
    rows = [
        _row("a@x.com", "b@x.com", NOW),            # internal
        _row("a@x.com", "outsider@other.com", NOW),  # external -> dropped
    ]
    parsed, dropped = build_flow_matrix(
        rows, org_users=ORG, now_utc=NOW, window_seconds=365 * DAY,
        half_life_seconds=10 ** 9, beta=0.0, granularity="individual")
    assert dropped == 1
    assert "outsider@other.com" not in parsed.node_names


def test_window_excludes_old_messages():
    rows = [
        _row("a@x.com", "b@x.com", NOW - 10 * DAY),   # in 30d window
        _row("a@x.com", "c@x.com", NOW - 100 * DAY),  # outside 30d window
    ]
    parsed, _ = build_flow_matrix(
        rows, org_users=ORG, now_utc=NOW, window_seconds=30 * DAY,
        half_life_seconds=10 ** 9, beta=0.0, granularity="individual")
    assert "c@x.com" not in parsed.node_names


def test_zero_half_life_raises():
    with pytest.raises(ValueError):
        build_flow_matrix([_row("a@x.com", "b@x.com", NOW)], org_users=ORG,
                          now_utc=NOW, window_seconds=DAY, half_life_seconds=0,
                          beta=0.0, granularity="individual")


def test_negative_beta_raises():
    with pytest.raises(ValueError):
        build_flow_matrix([_row("a@x.com", "b@x.com", NOW)], org_users=ORG,
                          now_utc=NOW, window_seconds=DAY, half_life_seconds=DAY,
                          beta=-1.0, granularity="individual")


def test_empty_rows_raise_ingestion_error():
    # No edges -> build_flow_matrix_from_edges rejects <2 nodes. Documents the contract.
    with pytest.raises(NetworkIngestionError):
        build_flow_matrix([], org_users=ORG, now_utc=NOW, window_seconds=DAY,
                          half_life_seconds=DAY, beta=0.0, granularity="individual")
