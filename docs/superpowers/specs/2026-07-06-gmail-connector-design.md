# Gmail Connector — Self-Provisioning Network Source (Design Spec)

**Date:** 2026-07-06
**Status:** Implemented (Tasks 1–7) — see `docs/superpowers/plans/2026-07-06-gmail-connector.md`. Approach B (two-stage with persisted raw-interaction store).
**Branch:** `feat/detailed-ecosystemic-report`

## 1. Goal

Let a Google Workspace admin connect their organization's Gmail once and have OASIS
automatically build a weighted communication-flow network from message **metadata**,
precompute the full metric profile, and open it in the existing analysis view — with no
manual CSV export.

This spec covers **Gmail only**. Slack and Microsoft 365 are explicitly out of scope for
this increment; the design reserves the seams for them (see §12).

## 2. Scope & Locked Decisions

| Decision | Choice |
|---|---|
| First increment | **Gmail only**, end-to-end |
| Auth model | **Admin OAuth consent** — domain-wide delegation via an admin-installed app |
| Coverage | **Org/admin-wide** (all users in the workspace) |
| Node granularity | **Both** — pull individual-level, analyze at individual *or* department roll-up |
| Flow weighting | **Hybrid** — recency decay × sustained-engagement (§6) |
| Privacy | **Metadata only** — `From`/`To`/`Cc`, timestamp, thread id, size headers. Never subject or body. |
| Time window | **Configurable** — window (30/90/180/365 days) + decay half-life, chosen at build time |

## 3. Architecture (Approach B: sync → build)

Two decoupled stages so the expensive, rate-limited network I/O runs once while the cheap,
tweakable weighting math re-runs freely.

```
Stage 1 — SYNC (network I/O, run once per pull)
  Admin OAuth  ──►  GmailConnector.sync()
                      • Admin SDK: users + orgUnitPath  → org structure
                      • Gmail API: message metadata headers (per user, windowed)
                      • write raw rows ──► gmail_interactions (SQLite)

Stage 2 — BUILD (pure math, re-run on any window/decay/granularity change)
  gmail_interactions ──► gmail_weighting.build_flow_matrix(window, half_life, beta, granularity)
                           • filter to window
                           • resolve email → node (individual OR department)
                           • hybrid decay × sustained weighting
                           • emit (source, target, weight) edges
                      ──► build_flow_matrix_from_edges()   [existing primitive]
                      ──► provision_network()               [existing precompute path]
                      ──► get_full_profile() cache          [existing]
                      ──► analysis view                     [existing]
```

Stage 2 is a **pure function over the stored table** — no Gmail calls. Re-windowing,
re-decaying, and flipping individual↔department are all cheap local recomputes.

## 4. File Structure

- **Create** `src/connectors/gmail_connector.py` — `GmailConnector` (Stage 1: auth + sync).
- **Create** `src/connectors/gmail_weighting.py` — pure Stage-2 weighting + edge emission.
- **Create** `src/connectors/gmail_store.py` — SQLite DAO for the `gmail_interactions` table
  (schema, upsert, query-by-window).
- **Create** `src/connectors/__init__.py` — package exports.
- **Create** `tests/connectors/test_gmail_weighting.py` — pure-math tests (no network).
- **Create** `tests/connectors/test_gmail_store.py` — table round-trip + window filter.
- **Create** `tests/connectors/test_gmail_connector.py` — sync with a mocked Gmail client.
- **Modify** `app.py` — add "🔌 Connect Gmail" data-source mode + `connect_gmail_interface()`.
- **Modify** `src/cloud_connectors.py` — retire the stub `GoogleWorkspaceConnector.get_flow_data`
  body in favor of delegating to the new package (keep `BaseConnector` ABC + `ConnectorFactory`).

**`BaseConnector` conformance:** `GmailConnector` subclasses `BaseConnector` and implements
`authenticate()`, `get_organization_structure()`, and `get_metadata()` directly. The
two-stage design adds an explicit `sync(window, run_id) -> int` (rows written) method that
the UI drives. To satisfy the ABC's `get_flow_data(start, end) -> np.ndarray`, `GmailConnector`
implements it as a thin convenience wrapper (`sync` the window, then `gmail_weighting.build_flow_matrix`
with defaults, return the matrix) — but the UI uses the explicit `sync` + `build` methods so
the two stages stay independently invokable.
- **Modify** `docs/requirements.txt` — add `google-api-python-client`, `google-auth`,
  `google-auth-oauthlib`.

Rationale for a new `src/connectors/` package rather than growing `cloud_connectors.py`:
that file is a flat POC with three provider stubs in one module; Gmail now needs three
cooperating units (auth/sync, storage, weighting) that are individually testable. Keeping
them in a focused package matches the "files that change together live together" principle.

## 5. Data Model — `gmail_interactions`

One row per observed directed message edge (a message with N recipients yields N rows).
Stored in the existing `data/database/networks.db` (a **runtime artifact** — never staged;
`git checkout -- data/database/networks.db` before commits, as elsewhere in this repo).

```sql
CREATE TABLE IF NOT EXISTS gmail_interactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    org_domain   TEXT    NOT NULL,   -- workspace domain, scopes a tenant
    sync_run_id  TEXT    NOT NULL,   -- groups rows from one sync (passed in, not generated in-lib)
    src_email    TEXT    NOT NULL,   -- sender
    dst_email    TEXT    NOT NULL,   -- one recipient (To or Cc)
    recipient_kind TEXT  NOT NULL,   -- 'to' | 'cc'
    ts_utc       INTEGER NOT NULL,   -- message epoch seconds (UTC)
    thread_id    TEXT,               -- Gmail thread id (dedupe / reply grouping)
    size_bytes   INTEGER,            -- message size header (optional volume signal)
    src_orgunit  TEXT,               -- sender orgUnitPath at sync time
    dst_orgunit  TEXT                -- recipient orgUnitPath at sync time
);
CREATE INDEX IF NOT EXISTS ix_gmail_org_ts ON gmail_interactions(org_domain, ts_utc);
```

Notes:
- **No subject, no body, no snippet columns exist** — metadata-only is enforced by schema,
  not just by convention.
- Department is captured as the `orgUnitPath` *at sync time* (people move teams; we record
  the structure as it was when the message flowed).
- `ts_utc` is passed in from the caller / Gmail header; the library never calls
  `Date.now()`-style clocks itself (repo constraint: no wall-clock in reusable libs — the
  "current time" reference for decay is an explicit `now_utc` argument, §6). Likewise the
  `sync_run_id` and the `now_utc` decay reference are **generated by the app.py UI layer**
  (which may use the clock) and passed down, never minted inside the reusable modules.

### 5.1 Node resolution & external-address filtering

- **Individual granularity:** node label = the person's **email address** (lower-cased).
- **Department granularity:** node label = the **leaf of `orgUnitPath`** (e.g. `/Sales/EMEA`
  → `EMEA`), matching the existing `GoogleWorkspaceConnector` convention; a department edge
  aggregates (sums) every individual edge whose endpoints resolve to those units.
- **External-address filtering (correctness-critical):** only recipients that resolve to a
  **known org user** in the Admin SDK directory become nodes. Emails to/from outside the
  workspace domain are **dropped** at Stage 2 and their dropped-edge count is reported as a
  warning (they are not part of the *internal* organizational flow network). The org-user
  set comes from `get_organization_structure()` captured during the same sync.

## 6. Weighting Math (hybrid decay × sustained)

For a directed node pair (a → b), let its messages have UTC timestamps `t_1..t_k` and let
`now` be an explicit reference time supplied by the caller.

**Recency decay (per message):** exponential half-life decay — a standard, defensible
recency kernel.
```
λ = ln(2) / half_life_seconds
decay_i = exp(-λ · (now - t_i))          # t_i ≤ now; clamp (now - t_i) ≥ 0
volume(a→b) = Σ_i decay_i                 # decayed message volume
```

**Sustained engagement (per pair):** reward relationships that recur across many distinct
active periods rather than a single burst of equal decayed volume.
```
A(a→b) = count of distinct active buckets (ISO weeks) in which a→b sent ≥1 message
sustain(a→b) = 1 + β · ln(1 + A(a→b))     # β ≥ 0, tunable
```

**Hybrid edge weight:**
```
w(a→b) = volume(a→b) · sustain(a→b)
```

Parameters and their status:
- `half_life` — user-configurable (default 30 days). Recency kernel.
- `β` (sustained coefficient) — **calibration parameter**, default `0.5`, documented as such.
- Bucket granularity for `A` — ISO week (fixed for v1).

**Scientific note (per repo CLAUDE.md):** exponential half-life recency is an established
kernel and needs no new justification. The *sustained-engagement multiplier* and its
default `β` are a **design/calibration choice for network construction**, not an Ulanowicz
scientific formula — it changes how the network is *built*, not how any validated metric is
*computed*. It must be documented as a calibration parameter and, before being sold as
"correct," validated against organizations with known collaboration structure. This spec
does **not** touch any core metric formula.

## 7. Authentication (admin OAuth, domain-wide delegation)

- The app is registered as a Google Cloud project with a **service account** granted
  **domain-wide delegation**; a Workspace **admin authorizes** the app's client id with the
  minimal read-only scopes (one-time admin consent — the self-provisioning story).
- **Scopes (read-only, least privilege):**
  - `https://www.googleapis.com/auth/admin.directory.user.readonly` — users + orgUnitPath
  - `https://www.googleapis.com/auth/gmail.metadata` — message **metadata only** (this scope
    cannot read subject or body by construction)
- Credentials are supplied by the admin (service-account JSON path + `subject` admin email
  + domain) and are **read from Streamlit secrets / env**, never hard-coded and never
  committed. `GmailConnector.authenticate(credentials: dict) -> bool` mirrors the existing
  `BaseConnector` contract.
- The connector impersonates each user via delegation to read that user's metadata; it never
  stores a user-level long-lived token.

## 8. Privacy Posture

- Only header fields listed in §5 are ever requested (`gmail.metadata` scope makes body
  access impossible).
- The UI states plainly, before connecting: *"OASIS reads only who-emailed-whom and when —
  never subjects or message contents."*
- Stored rows contain email addresses and org units; the network the analyst sees can be
  rendered at department granularity to avoid surfacing individuals when not needed.

## 9. UI Flow (`connect_gmail_interface()`)

Added as a new sidebar mode `🔌 Connect Gmail`, alongside Upload / Sample / Synthetic.

1. **Explainer + privacy statement** (metadata-only), and a "requires Workspace admin" note.
2. **Connect** — validates credentials from secrets/env via `authenticate()`; shows the
   resolved domain + user count on success.
3. **Sync controls** — pick the pull window (30/90/180/365 days) → runs `sync()` with a
   progress indicator; reports rows ingested and users covered.
4. **Build controls** — choose analysis granularity (Individual / Department), decay
   half-life (default 30d), and β (advanced, default 0.5) → runs Stage 2, calls
   `provision_network()`, sets `st.session_state.current_page = 'analysis'` and navigates —
   exactly like every other provision path. Re-running Build with new settings does **not**
   re-pull Gmail.
5. Respect the existing "Back to Data Selection" contract (clears
   `selected_dataset_name` + `full_profile`).

## 10. Error Handling

- **Auth failure** (bad key, delegation not granted, missing scope): return `False`, surface
  a specific, actionable message (which scope/step is missing); never crash the app.
- **Gmail rate limits / 429**: exponential backoff with a capped retry count in `sync()`;
  partial syncs write what they got and report how many users completed.
- **Empty result** (no messages in window): raise the same `NetworkIngestionError` path the
  CSV flow uses ("Total flow is zero…") so the UI message is consistent.
- **< 2 nodes after resolution**: reuse `build_flow_matrix_from_edges`'s existing guard.
- **Provision failure**: `provision_network` already swallows and falls back to lazy compute;
  no new failure mode introduced.

## 11. Testing

Pure Stage-2 math is the correctness core and is tested without any network:

- `test_gmail_weighting.py`
  - decay: a message `half_life` seconds old contributes exactly `0.5` of a fresh one.
  - sustained: two pairs with equal decayed volume but different active-week counts rank by
    `A` (more distinct weeks ⇒ higher weight).
  - granularity: same raw rows produce a larger individual matrix and a correctly
    department-aggregated matrix that sums the constituent individual flows.
  - external filtering: rows whose recipient is outside the known org-user set are dropped
    and counted; only internal edges reach the matrix.
  - `now_utc` is an explicit argument (no wall-clock in the library).
- `test_gmail_store.py`: insert rows across two windows, query-by-window returns only the
  in-window rows; metadata-only schema (assert no subject/body columns).
- `test_gmail_connector.py`: `sync()` against a **mocked** Gmail/Admin client fixture emits
  the expected rows; a `message_sent` with To+Cc yields the right per-recipient rows with
  correct `recipient_kind`; auth failure returns `False`.

Target: all new tests green; full existing suite (301 tests) still green.

## 12. Out of Scope / Future Seams

- **Slack, Microsoft 365** — increment 2+. They implement the same `BaseConnector` and reuse
  `gmail_weighting`'s generic core (rename to `interaction_weighting` when the second
  provider lands; premature now — YAGNI).
- **Incremental sync** (pull only messages newer than last `sync_run_id`) — the
  `sync_run_id` + `ts_utc` index already support it; not built in v1.
- **Drive/Calendar signals** — metadata sources beyond email; not in v1.
- **Token vault / background workers / multi-tenant job queue** (approach C) — added only
  when multiple providers/tenants demand it. The existing `MultiSourceAggregator` /
  `ConnectorFactory` reserve that seam.
- **Subject/topic tagging** — deliberately excluded by the metadata-only decision.

## 13. Non-Goals for Formula Integrity

No core Ulanowicz measure, threshold constant, or composite formula is touched. This feature
only *constructs* an input network from a new source; it feeds the identical
`build_flow_matrix_from_edges → provision_network → get_full_profile` path that CSV upload
already uses.
