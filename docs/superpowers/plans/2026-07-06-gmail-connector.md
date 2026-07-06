# Gmail Connector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Google Workspace admin connect Gmail once and have OASIS build a weighted communication-flow network from message metadata, precompute it, and open it in the existing analysis view.

**Architecture:** Two decoupled stages behind a `src/connectors/` package. Stage 1 (`GmailConnector.sync`) pulls metadata-only headers via the Gmail API + Admin SDK and writes raw directed rows into a `gmail_interactions` SQLite table. Stage 2 (`gmail_weighting.build_flow_matrix`) is a pure function over that table — filters to a window, resolves email→node at individual or department granularity, drops external addresses, applies hybrid recency-decay × sustained-engagement weighting, and emits `(source, target, weight)` edges into the existing `build_flow_matrix_from_edges → provision_network → get_full_profile` path.

**Tech Stack:** Python 3, SQLite (`data/database/networks.db`), NumPy, pytest, `google-api-python-client` / `google-auth` / `google-auth-oauthlib`, Streamlit (`app.py`).

**Spec:** `docs/superpowers/specs/2026-07-06-gmail-connector-design.md`

**Conventions for every task:**
- Git identity: `git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit ...`
- Commit footer line: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- `data/database/networks.db` is a runtime artifact — run `git checkout -- data/database/networks.db 2>/dev/null` before every commit and NEVER `git add` it.
- Run tests from repo root `/Users/massimomistretta/Claude_Projects/Adaptive_Organization`.
- Reusable modules under `src/connectors/` must NOT call the wall clock (`datetime.now()`, `time.time()`); any "current time" or run id is passed in as an argument.

---

## File Structure

- `src/connectors/__init__.py` — package exports (`GmailConnector`, `GmailInteractionStore`, `build_flow_matrix`).
- `src/connectors/gmail_store.py` — `GmailInteractionStore`: schema + insert + query-by-window over `gmail_interactions`.
- `src/connectors/gmail_weighting.py` — pure Stage-2: `build_flow_matrix(rows, org_users, now_utc, window_seconds, half_life_seconds, beta, granularity)`.
- `src/connectors/gmail_connector.py` — `GmailConnector(BaseConnector)`: auth, org structure, `sync`, `get_flow_data` wrapper.
- `tests/connectors/__init__.py` — makes the test package importable.
- `tests/connectors/test_gmail_store.py` — table round-trip + window filter + schema assertions.
- `tests/connectors/test_gmail_weighting.py` — decay, sustained, granularity, external filtering, no-wall-clock.
- `tests/connectors/test_gmail_connector.py` — sync against a mocked Gmail/Admin client; auth failure.
- `app.py` — add `🔌 Connect Gmail` mode + `connect_gmail_interface()`.
- `src/cloud_connectors.py` — retire the stub `GoogleWorkspaceConnector.get_flow_data` body, delegate to the new package.
- `docs/requirements.txt` — add Google client libraries.

Data types used across tasks (define once, reuse):
- A **raw row** is a dict: `{"src_email","dst_email","recipient_kind","ts_utc","thread_id","size_bytes","src_orgunit","dst_orgunit"}`.
- `org_users` is a `set[str]` of lower-cased org email addresses.
- `build_flow_matrix(...)` returns the existing `ParseResult` (from `src/network_ingestion.py`) plus a `dropped_external` count, wrapped as `(ParseResult, dropped_external: int)`.

---

## Task 1: Package skeleton + dependencies

**Files:**
- Create: `src/connectors/__init__.py`
- Create: `tests/connectors/__init__.py`
- Modify: `docs/requirements.txt`

- [ ] **Step 1: Create the test package marker**

Create `tests/connectors/__init__.py` with a single line:

```python
# Test package for src.connectors
```

- [ ] **Step 2: Create the package init (exports filled in by later tasks)**

Create `src/connectors/__init__.py`:

```python
"""Self-provisioning network-source connectors (Gmail first).

Two-stage design: GmailConnector.sync() pulls metadata into GmailInteractionStore;
build_flow_matrix() turns stored rows into a weighted flow matrix for analysis.
"""

from .gmail_store import GmailInteractionStore
from .gmail_weighting import build_flow_matrix
from .gmail_connector import GmailConnector

__all__ = ["GmailInteractionStore", "build_flow_matrix", "GmailConnector"]
```

Note: this file will not import cleanly until Tasks 2–4 create those modules. That is expected; do not run it yet.

- [ ] **Step 3: Add Google client dependencies**

Append to `docs/requirements.txt`:

```
google-api-python-client>=2.100
google-auth>=2.23
google-auth-oauthlib>=1.1
```

- [ ] **Step 4: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add src/connectors/__init__.py tests/connectors/__init__.py docs/requirements.txt
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "feat(connectors): package skeleton + Google client deps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `GmailInteractionStore` (SQLite DAO)

**Files:**
- Create: `src/connectors/gmail_store.py`
- Test: `tests/connectors/test_gmail_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/connectors/test_gmail_store.py`:

```python
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
    store.insert_rows("x.com", "run1", rows)
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


def test_schema_has_no_content_columns(store):
    cols = store.column_names()
    forbidden = {"subject", "body", "snippet", "content", "text"}
    assert forbidden.isdisjoint(cols), f"metadata-only violated: {cols & forbidden}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/connectors/test_gmail_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.connectors.gmail_store'`

- [ ] **Step 3: Implement the store**

Create `src/connectors/gmail_store.py`:

```python
"""SQLite DAO for raw Gmail interaction rows (metadata only).

One row per directed message edge (a message to N recipients => N rows). No
subject/body/snippet columns exist — metadata-only is enforced by schema.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Set

DEFAULT_DB_PATH = "data/database/networks.db"

_COLUMNS = [
    "src_email", "dst_email", "recipient_kind", "ts_utc",
    "thread_id", "size_bytes", "src_orgunit", "dst_orgunit",
]


class GmailInteractionStore:
    """Read/write access to the gmail_interactions table."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gmail_interactions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_domain   TEXT    NOT NULL,
                    sync_run_id  TEXT    NOT NULL,
                    src_email    TEXT    NOT NULL,
                    dst_email    TEXT    NOT NULL,
                    recipient_kind TEXT  NOT NULL,
                    ts_utc       INTEGER NOT NULL,
                    thread_id    TEXT,
                    size_bytes   INTEGER,
                    src_orgunit  TEXT,
                    dst_orgunit  TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_gmail_org_ts "
                "ON gmail_interactions(org_domain, ts_utc)"
            )

    def insert_rows(self, org_domain: str, sync_run_id: str,
                    rows: Iterable[Dict]) -> int:
        payload = [
            (org_domain, sync_run_id,
             r["src_email"], r["dst_email"], r["recipient_kind"], int(r["ts_utc"]),
             r.get("thread_id"), r.get("size_bytes"),
             r.get("src_orgunit"), r.get("dst_orgunit"))
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO gmail_interactions "
                "(org_domain, sync_run_id, src_email, dst_email, recipient_kind, "
                " ts_utc, thread_id, size_bytes, src_orgunit, dst_orgunit) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def query_window(self, org_domain: str, start_ts: int,
                     end_ts: int) -> List[Dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM gmail_interactions "
                "WHERE org_domain = ? AND ts_utc >= ? AND ts_utc <= ? "
                "ORDER BY ts_utc",
                (org_domain, start_ts, end_ts),
            )
            return [dict(row) for row in cur.fetchall()]

    def column_names(self) -> Set[str]:
        with self._connect() as conn:
            cur = conn.execute("PRAGMA table_info(gmail_interactions)")
            return {row["name"] for row in cur.fetchall()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/connectors/test_gmail_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add src/connectors/gmail_store.py tests/connectors/test_gmail_store.py
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "feat(connectors): gmail_interactions store (metadata-only DAO)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `build_flow_matrix` — pure hybrid weighting (Stage 2)

**Files:**
- Create: `src/connectors/gmail_weighting.py`
- Test: `tests/connectors/test_gmail_weighting.py`

Reference — `build_flow_matrix_from_edges(edges)` lives in `src/network_ingestion.py`, takes an iterable of `(source, target, weight)` tuples and returns a `ParseResult` with `.flow_matrix` (np.ndarray) and `.node_names` (sorted list).

- [ ] **Step 1: Write the failing tests**

Create `tests/connectors/test_gmail_weighting.py`:

```python
import math

from src.connectors.gmail_weighting import build_flow_matrix

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/connectors/test_gmail_weighting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.connectors.gmail_weighting'`

- [ ] **Step 3: Implement the weighting**

Create `src/connectors/gmail_weighting.py`:

```python
"""Stage 2 (pure): turn stored Gmail rows into a weighted flow matrix.

Hybrid weighting per directed node pair (a -> b):
    volume(a,b)  = sum_i exp(-ln2/half_life * (now - t_i))   # recency decay
    sustain(a,b) = 1 + beta * ln(1 + A)                      # A = # distinct active ISO weeks
    weight(a,b)  = volume(a,b) * sustain(a,b)

No wall clock is read here: `now_utc` is an explicit argument.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    from network_ingestion import build_flow_matrix_from_edges, ParseResult
except Exception:  # pragma: no cover - import path when run as a package
    from src.network_ingestion import build_flow_matrix_from_edges, ParseResult

_WEEK = 7 * 86400


def _dept(orgunit: str) -> str:
    """Leaf of an orgUnitPath: '/Sales/EMEA' -> 'EMEA'; '/' or '' -> 'Root'."""
    if not orgunit:
        return "Root"
    leaf = orgunit.rstrip("/").split("/")[-1]
    return leaf or "Root"


def build_flow_matrix(
    rows: List[Dict],
    org_users: Set[str],
    now_utc: int,
    window_seconds: int,
    half_life_seconds: int,
    beta: float,
    granularity: str = "individual",
) -> Tuple[ParseResult, int]:
    """Build a weighted flow matrix from raw interaction rows.

    Args:
        rows: raw interaction dicts (see GmailInteractionStore).
        org_users: lower-cased set of known internal email addresses.
        now_utc: reference time (epoch s) for decay — supplied, never clock-read.
        window_seconds: only messages with ts_utc >= now_utc - window_seconds count.
        half_life_seconds: recency-decay half life.
        beta: sustained-engagement coefficient (>= 0).
        granularity: 'individual' (node=email) or 'department' (node=orgUnit leaf).

    Returns:
        (ParseResult, dropped_external_count).
    """
    lam = math.log(2) / float(half_life_seconds)
    cutoff = now_utc - window_seconds

    # Accumulators keyed by (src_node, dst_node).
    volume: Dict[Tuple[str, str], float] = defaultdict(float)
    weeks: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    dropped_external = 0

    for r in rows:
        ts = int(r["ts_utc"])
        if ts < cutoff or ts > now_utc:
            continue
        src_email = str(r["src_email"]).strip().lower()
        dst_email = str(r["dst_email"]).strip().lower()
        # External filtering: both endpoints must be known org users.
        if src_email not in org_users or dst_email not in org_users:
            dropped_external += 1
            continue
        if granularity == "department":
            src_node = _dept(r.get("src_orgunit"))
            dst_node = _dept(r.get("dst_orgunit"))
        else:
            src_node = src_email
            dst_node = dst_email
        if src_node == dst_node:
            continue  # ignore intra-node self-flow
        key = (src_node, dst_node)
        volume[key] += math.exp(-lam * (now_utc - ts))
        weeks[key].add(ts // _WEEK)

    edges = []
    for key, vol in volume.items():
        active_weeks = len(weeks[key])
        sustain = 1.0 + beta * math.log(1 + active_weeks)
        edges.append((key[0], key[1], vol * sustain))

    parsed = build_flow_matrix_from_edges(edges)
    return parsed, dropped_external
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/connectors/test_gmail_weighting.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add src/connectors/gmail_weighting.py tests/connectors/test_gmail_weighting.py
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "feat(connectors): pure hybrid decay x sustained weighting (Stage 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `GmailConnector` — auth + sync (Stage 1)

**Files:**
- Create: `src/connectors/gmail_connector.py`
- Test: `tests/connectors/test_gmail_connector.py`

The connector must be testable without real Google APIs. It takes an injectable
`admin_client` and `gmail_client` (duck-typed) so tests pass fakes; production builds them
from credentials. It subclasses `BaseConnector` from `src.cloud_connectors`.

- [ ] **Step 1: Write the failing tests**

Create `tests/connectors/test_gmail_connector.py`:

```python
from src.connectors.gmail_connector import GmailConnector


class FakeAdmin:
    """Mimics the two calls the connector makes on the Admin SDK."""
    def list_users(self):
        return [
            {"primaryEmail": "a@x.com", "orgUnitPath": "/Sales"},
            {"primaryEmail": "b@x.com", "orgUnitPath": "/IT"},
        ]


class FakeGmail:
    """Returns metadata headers for one sent message with To + Cc."""
    def list_sent_messages(self, user_email, start_ts, end_ts):
        if user_email != "a@x.com":
            return []
        return [{
            "ts_utc": (start_ts + end_ts) // 2,
            "thread_id": "thread-1",
            "size_bytes": 500,
            "from": "a@x.com",
            "to": ["b@x.com"],
            "cc": ["b@x.com"],
        }]


def test_get_organization_structure_maps_users_to_orgunits():
    c = GmailConnector(admin_client=FakeAdmin(), gmail_client=FakeGmail(),
                       domain="x.com")
    org = c.get_organization_structure()
    assert org["user_orgunit"]["a@x.com"] == "/Sales"
    assert org["org_users"] == {"a@x.com", "b@x.com"}


def test_sync_emits_per_recipient_rows(tmp_path):
    from src.connectors.gmail_store import GmailInteractionStore
    store = GmailInteractionStore(db_path=str(tmp_path / "t.db"))
    c = GmailConnector(admin_client=FakeAdmin(), gmail_client=FakeGmail(),
                       domain="x.com", store=store)
    n = c.sync(start_ts=1000, end_ts=3000, sync_run_id="run1")
    assert n == 2  # one To row + one Cc row
    rows = store.query_window("x.com", 0, 10 ** 12)
    kinds = sorted(r["recipient_kind"] for r in rows)
    assert kinds == ["cc", "to"]
    assert all(r["src_email"] == "a@x.com" and r["dst_email"] == "b@x.com"
               for r in rows)
    assert rows[0]["src_orgunit"] == "/Sales"
    assert rows[0]["dst_orgunit"] == "/IT"


def test_authenticate_returns_false_on_missing_credentials():
    c = GmailConnector()
    assert c.authenticate({}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/connectors/test_gmail_connector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.connectors.gmail_connector'`

- [ ] **Step 3: Implement the connector**

Create `src/connectors/gmail_connector.py`:

```python
"""Stage 1: Gmail metadata sync into GmailInteractionStore.

GmailConnector subclasses BaseConnector. Admin/Gmail clients are injected so the
logic is testable with fakes; production builds real clients from credentials.
Only metadata headers are read (From/To/Cc/timestamp/thread/size) — never body.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:
    from cloud_connectors import BaseConnector
except Exception:  # pragma: no cover
    from src.cloud_connectors import BaseConnector

from .gmail_store import GmailInteractionStore
from .gmail_weighting import build_flow_matrix

# Read-only, least-privilege scopes (metadata scope cannot read body/subject).
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/admin.directory.user.readonly",
    "https://www.googleapis.com/auth/gmail.metadata",
]


class GmailConnector(BaseConnector):
    def __init__(self, admin_client=None, gmail_client=None,
                 domain: Optional[str] = None,
                 store: Optional[GmailInteractionStore] = None):
        self.admin_client = admin_client
        self.gmail_client = gmail_client
        self.domain = domain
        self.store = store or GmailInteractionStore()

    # --- BaseConnector contract -------------------------------------------------

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Build Admin + Gmail clients from a service-account credential dict.

        Required keys: 'service_account_file', 'subject' (admin to impersonate),
        'domain'. Returns False (never raises) on any missing key or build error.
        """
        required = ("service_account_file", "subject", "domain")
        if not all(credentials.get(k) for k in required):
            return False
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                credentials["service_account_file"], scopes=GMAIL_SCOPES,
            ).with_subject(credentials["subject"])
            self.admin_client = _AdminSdkClient(
                build("admin", "directory_v1", credentials=creds))
            self.gmail_client = _GmailApiClient(credentials, GMAIL_SCOPES)
            self.domain = credentials["domain"]
            return True
        except Exception as exc:  # pragma: no cover - real-API path
            print(f"Gmail authenticate failed: {exc}")
            return False

    def get_organization_structure(self) -> Dict[str, Any]:
        users = self.admin_client.list_users()
        user_orgunit = {u["primaryEmail"].lower(): u.get("orgUnitPath", "/")
                        for u in users}
        return {
            "org_users": set(user_orgunit.keys()),
            "user_orgunit": user_orgunit,
            "total_users": len(user_orgunit),
        }

    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """BaseConnector convenience: sync the window then build with defaults."""
        start_ts, end_ts = int(start_date.timestamp()), int(end_date.timestamp())
        self.sync(start_ts, end_ts, sync_run_id=f"flowdata-{start_ts}")
        org = self.get_organization_structure()
        rows = self.store.query_window(self.domain, start_ts, end_ts)
        parsed, _ = build_flow_matrix(
            rows, org_users=org["org_users"], now_utc=end_ts,
            window_seconds=end_ts - start_ts, half_life_seconds=30 * 86400,
            beta=0.5, granularity="individual")
        return parsed.flow_matrix

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "connector": "Gmail",
            "domain": self.domain,
            "data_sources": ["Gmail metadata", "Admin SDK"],
            "privacy": "metadata-only (From/To/Cc/timestamp/thread/size)",
        }

    # --- Stage 1: sync ----------------------------------------------------------

    def sync(self, start_ts: int, end_ts: int, sync_run_id: str) -> int:
        """Pull metadata for every org user in [start_ts, end_ts]; store rows.

        Returns the number of directed rows written (one per To/Cc recipient).
        """
        org = self.get_organization_structure()
        user_orgunit = org["user_orgunit"]
        rows = []
        for sender in user_orgunit:
            for msg in self.gmail_client.list_sent_messages(sender, start_ts, end_ts):
                src = str(msg["from"]).lower()
                for kind in ("to", "cc"):
                    for rcpt in msg.get(kind, []) or []:
                        dst = str(rcpt).lower()
                        rows.append({
                            "src_email": src, "dst_email": dst,
                            "recipient_kind": kind, "ts_utc": int(msg["ts_utc"]),
                            "thread_id": msg.get("thread_id"),
                            "size_bytes": msg.get("size_bytes"),
                            "src_orgunit": user_orgunit.get(src),
                            "dst_orgunit": user_orgunit.get(dst),
                        })
        if not rows:
            return 0
        return self.store.insert_rows(self.domain, sync_run_id, rows)


class _AdminSdkClient:  # pragma: no cover - thin real-API adapter
    """Adapts the googleapiclient Admin SDK to the list_users() duck type."""
    def __init__(self, service):
        self.service = service

    def list_users(self):
        out, page = [], None
        while True:
            resp = self.service.users().list(
                customer="my_customer", maxResults=500, pageToken=page,
                projection="full").execute()
            out.extend(resp.get("users", []))
            page = resp.get("nextPageToken")
            if not page:
                break
        return out


class _GmailApiClient:  # pragma: no cover - real-API adapter, exercised via fakes in tests
    """Adapts the Gmail API to list_sent_messages(user, start_ts, end_ts).

    Uses format='metadata' so message bodies are never fetched.
    """
    def __init__(self, credentials: Dict[str, Any], scopes):
        self._credentials = credentials
        self._scopes = scopes

    def _service_for(self, user_email):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_file(
            self._credentials["service_account_file"], scopes=self._scopes,
        ).with_subject(user_email)
        return build("gmail", "v1", credentials=creds)

    def list_sent_messages(self, user_email, start_ts, end_ts):
        service = self._service_for(user_email)
        query = f"in:sent after:{start_ts} before:{end_ts}"
        results = []
        page = None
        while True:
            resp = service.users().messages().list(
                userId="me", q=query, pageToken=page).execute()
            for ref in resp.get("messages", []):
                msg = service.users().messages().get(
                    userId="me", id=ref["id"], format="metadata",
                    metadataHeaders=["From", "To", "Cc", "Date"]).execute()
                results.append(_parse_metadata(msg))
            page = resp.get("nextPageToken")
            if not page:
                break
        return results


def _parse_metadata(msg) -> Dict[str, Any]:  # pragma: no cover - real-API shape
    headers = {h["name"].lower(): h["value"]
               for h in msg.get("payload", {}).get("headers", [])}

    def addrs(v):
        return [a.strip() for a in v.split(",")] if v else []

    return {
        "ts_utc": int(msg.get("internalDate", "0")) // 1000,
        "thread_id": msg.get("threadId"),
        "size_bytes": msg.get("sizeEstimate"),
        "from": headers.get("from", ""),
        "to": addrs(headers.get("to", "")),
        "cc": addrs(headers.get("cc", "")),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/connectors/test_gmail_connector.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the whole connectors package + import check**

Run: `python -c "import src.connectors" && python -m pytest tests/connectors/ -v`
Expected: package imports cleanly; all connector tests PASS.

- [ ] **Step 6: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add src/connectors/gmail_connector.py tests/connectors/test_gmail_connector.py
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "feat(connectors): GmailConnector auth + metadata sync (Stage 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Retire the `cloud_connectors` Google stub

**Files:**
- Modify: `src/cloud_connectors.py:125-182` (the stub `GoogleWorkspaceConnector.get_flow_data`)
- Test: `tests/connectors/test_gmail_connector.py` (add a delegation test)

The legacy `GoogleWorkspaceConnector` holds raw `googleapiclient` service objects
(`admin_service`, `reports_service`) whose shapes differ from the duck-typed clients
`GmailConnector` expects (`.list_users()` / `.list_sent_messages()`). A silent "delegation"
that passed those raw services would fail at runtime and — worse — the current stub returns a
zero matrix, which reads as real data downstream. The correct deprecation is to make the old
method **refuse loudly and point to the new path**, not fabricate a matrix.

- [ ] **Step 1: Write the failing test**

Append to `tests/connectors/test_gmail_connector.py`:

```python
def test_legacy_google_stub_points_to_new_connector():
    # The legacy GoogleWorkspaceConnector must no longer fabricate a matrix; it
    # should refuse and direct callers to src.connectors.GmailConnector.
    import inspect
    import pytest
    from datetime import datetime
    from src.cloud_connectors import GoogleWorkspaceConnector

    src = inspect.getsource(GoogleWorkspaceConnector.get_flow_data)
    assert "GmailConnector" in src, "legacy stub must reference the new connector"
    assert "np.zeros" not in src, "legacy stub must not fabricate a matrix"

    with pytest.raises(NotImplementedError):
        GoogleWorkspaceConnector().get_flow_data(datetime(2026, 1, 1),
                                                 datetime(2026, 2, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/connectors/test_gmail_connector.py::test_legacy_google_stub_points_to_new_connector -v`
Expected: FAIL — current stub returns a zero matrix (no `NotImplementedError`, no `GmailConnector` reference).

- [ ] **Step 3: Replace the stub body**

In `src/cloud_connectors.py`, replace the entire `get_flow_data` method of
`GoogleWorkspaceConnector` (currently at lines 125-182) with:

```python
    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Superseded by the two-stage GmailConnector — do not fabricate data.

        The old inline Reports-API extraction returned a zero matrix, which reads
        as real (empty) data downstream. Metadata-only Gmail ingestion now lives in
        src.connectors.GmailConnector (sync -> store -> build_flow_matrix), driven
        by the 'Connect Gmail' UI. Refuse rather than mislead.
        """
        raise NotImplementedError(
            "GoogleWorkspaceConnector.get_flow_data is retired. Use "
            "src.connectors.GmailConnector (Connect Gmail in the app), which pulls "
            "metadata-only and builds a weighted flow matrix."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/connectors/test_gmail_connector.py -v`
Expected: PASS (4 passed, including the new deprecation test).

- [ ] **Step 5: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add src/cloud_connectors.py tests/connectors/test_gmail_connector.py
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "refactor(connectors): retire GoogleWorkspace stub, point to GmailConnector

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire `🔌 Connect Gmail` into the app UI

**Files:**
- Modify: `app.py` — add mode to the list (near line 986), dispatch (near line 1006), and a new `connect_gmail_interface()` function.

This task is UI glue; it is exercised manually (Streamlit) rather than via pytest. The
provision pattern to mirror is `_try_direct_analyze` at `app.py:1465-1474`.

- [ ] **Step 1: Add the mode to the sidebar list**

In `app.py`, find (near line 986):

```python
    mode_list = [
        "📊 Upload Data",
        "🧪 Use Sample Data",
        "⚡ Generate Synthetic Data"
    ]
```

Replace with:

```python
    mode_list = [
        "📊 Upload Data",
        "🧪 Use Sample Data",
        "⚡ Generate Synthetic Data",
        "🔌 Connect Gmail"
    ]
```

- [ ] **Step 2: Add the dispatch branch**

In `app.py`, find (near line 1006):

```python
    if analysis_mode == "📊 Upload Data":
        upload_data_interface()
    elif analysis_mode == "🧪 Use Sample Data":
        sample_data_interface()
    elif analysis_mode == "⚡ Generate Synthetic Data":
        synthetic_data_interface()
```

Insert a new branch immediately after the synthetic branch:

```python
    elif analysis_mode == "🔌 Connect Gmail":
        connect_gmail_interface()
```

- [ ] **Step 3: Implement `connect_gmail_interface()`**

Add this function in `app.py` immediately before `def synthetic_data_interface(` (search for that def to place it). Uses `datetime` (already imported at top of app.py) to mint the UI-layer `now_utc` and `sync_run_id` — allowed here because app.py is the UI layer, not a reusable module.

```python
def connect_gmail_interface():
    """Self-provisioning Gmail connector: admin OAuth -> sync -> build -> analyze."""
    from datetime import datetime, timedelta
    st.header("🔌 Connect Gmail")
    st.info(
        "OASIS reads only **who-emailed-whom and when** — never subjects or "
        "message contents. Requires a Google Workspace **admin** to authorize the "
        "app (domain-wide delegation)."
    )

    try:
        from src.connectors import GmailConnector, GmailInteractionStore, build_flow_matrix
    except Exception as exc:
        st.error(f"Connector unavailable: {exc}")
        return

    # 1) Credentials come from Streamlit secrets (never hard-coded / committed).
    creds = dict(st.secrets.get("gmail", {})) if hasattr(st, "secrets") else {}
    if not creds.get("service_account_file"):
        st.warning(
            "No Gmail credentials configured. Add a `[gmail]` block to "
            "`.streamlit/secrets.toml` with `service_account_file`, `subject` "
            "(admin email), and `domain`."
        )
        return

    if st.button("🔗 Connect", type="primary"):
        conn = GmailConnector()
        if conn.authenticate(creds):
            st.session_state["gmail_domain"] = creds["domain"]
            org = conn.get_organization_structure()
            st.success(
                f"Connected to **{creds['domain']}** — "
                f"{org['total_users']} users."
            )
        else:
            st.error("Authentication failed. Check the service account, admin "
                     "subject, and that domain-wide delegation is granted.")

    if not st.session_state.get("gmail_domain"):
        return

    domain = st.session_state["gmail_domain"]

    # 2) Sync controls
    st.subheader("1 · Sync mailbox metadata")
    win_days = st.selectbox("Pull window (days)", [30, 90, 180, 365], index=1)
    if st.button("⬇️ Sync now"):
        conn = GmailConnector()
        if not conn.authenticate(creds):
            st.error("Re-authentication failed.")
            return
        now = int(datetime.utcnow().timestamp())
        start = now - win_days * 86400
        run_id = f"sync-{now}"
        with st.spinner(f"Syncing last {win_days} days…"):
            n = conn.sync(start, now, sync_run_id=run_id)
        st.session_state["gmail_last_sync"] = now
        st.success(f"Synced {n} directed interactions.")

    if not st.session_state.get("gmail_last_sync"):
        return

    # 3) Build controls
    st.subheader("2 · Build the network")
    granularity = st.radio("Granularity", ["individual", "department"], index=1)
    half_life_days = st.slider("Recency half-life (days)", 7, 180, 30)
    beta = st.slider("Sustained-engagement weight (β)", 0.0, 2.0, 0.5, 0.1,
                     help="Calibration parameter — boosts relationships active "
                          "across many weeks. Not a scientific metric formula.")
    build_win_days = st.selectbox("Analysis window (days)", [30, 90, 180, 365],
                                  index=1, key="build_win")
    if st.button("🧮 Build & Analyze", type="primary"):
        store = GmailInteractionStore()
        conn = GmailConnector()
        conn.authenticate(creds)
        org = conn.get_organization_structure()
        now = int(datetime.utcnow().timestamp())
        rows = store.query_window(domain, now - build_win_days * 86400, now)
        parsed, dropped = build_flow_matrix(
            rows, org_users=org["org_users"], now_utc=now,
            window_seconds=build_win_days * 86400,
            half_life_seconds=half_life_days * 86400,
            beta=beta, granularity=granularity)
        if dropped:
            st.caption(f"Dropped {dropped} external-address interactions.")
        st.session_state.analysis_data = {
            "flow_matrix": parsed.flow_matrix,
            "node_names": parsed.node_names,
            "org_name": f"{domain} (Gmail · {granularity})",
            "source": "gmail_connector",
        }
        provision_network(st.session_state.analysis_data)
        st.session_state.current_page = "analysis"
        st.rerun()
```

- [ ] **Step 4: Syntax-check and smoke-test the app**

Run:
```bash
python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

Then confirm the new mode renders (app already running on :8501, or start it):
```bash
curl -s http://localhost:8501/ -o /dev/null -w "%{http_code}\n"
```
Expected: `200`. Manually (or via the CDP harness) select `🔌 Connect Gmail` and confirm the privacy notice + "No Gmail credentials configured" warning render without error.

- [ ] **Step 5: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add app.py
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "feat(app): Connect Gmail data-source mode (sync + build + analyze)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full-suite regression + docs note

**Files:**
- Modify: `docs/superpowers/specs/2026-07-06-gmail-connector-design.md` (mark implemented)

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q`
Expected: all tests pass (the pre-existing 301 plus the new connector tests). If any
pre-existing test fails, STOP and investigate before proceeding — the connector work is
additive and must not regress existing behavior.

- [ ] **Step 2: Add an implementation-status note to the spec**

At the top of `docs/superpowers/specs/2026-07-06-gmail-connector-design.md`, change the
`**Status:**` line to:

```markdown
**Status:** Implemented (Tasks 1–7) — see `docs/superpowers/plans/2026-07-06-gmail-connector.md`
```

- [ ] **Step 3: Commit**

```bash
git checkout -- data/database/networks.db 2>/dev/null
git add docs/superpowers/specs/2026-07-06-gmail-connector-design.md
git -c user.email=maxdolphin@gmail.com -c user.name="Massimo Mistretta" commit -m "docs(spec): mark Gmail connector implemented

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Metadata-only is a hard constraint.** Never add a subject/body/snippet column or fetch a
  message with `format='full'`. The `gmail.metadata` scope + `format='metadata'` enforce it.
- **No wall clock in `src/connectors/` reusable modules.** `now_utc` and `sync_run_id` are
  always passed in from `app.py`. Only `app.py` and the real-API adapter methods (marked
  `pragma: no cover`) may read the clock.
- **The runtime DB (`data/database/networks.db`) is never committed.** Always
  `git checkout --` it before `git add`.
- **The weighting β and half-life are calibration parameters, not scientific metric
  formulas** (per repo CLAUDE.md). They shape how the input network is *built*; no Ulanowicz
  measure is touched. Do not "optimize" any core metric while doing this work.
- **Scope is Gmail only.** Do not build Slack/M365 here — the seams (`BaseConnector`,
  `ConnectorFactory`, `MultiSourceAggregator`) already exist for later.
- **Rate-limit backoff (spec §10) is deferred to the real-API adapter.** The Gmail API
  429/backoff handling belongs in `_GmailApiClient.list_sent_messages` (the `pragma: no cover`
  real-API path), not in `sync()` — `sync()` stays pure orchestration over injected clients so
  it's testable with fakes. Add exponential backoff there when wiring live credentials; it is
  intentionally out of the TDD loop because it cannot be exercised without a live endpoint.
  This is a conscious deferral, not an omission.
