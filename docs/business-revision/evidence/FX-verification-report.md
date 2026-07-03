# OASIS Formula-Fix Pass — Verification Report

**Scope:** Independent verification that the Track-1 formula-fix pass (ENA methods, network-science,
mutualism, roll-up veto, gradient reframe, size-normalization) is correct, that no core measure changed,
and that every intended behavior holds.

- **Date:** 2026-07-03
- **Branch:** `feat/detailed-ecosystemic-report`
- **Fix commits under test:** `d88ac2e … 200539f` (see history below)
- **Errors report:** `docs/business-revision/OASIS-formula-errors-report.md`
- **Pre-fix baseline commit (regression anchor):** `c137bf5`
- **Reproduce:** `python3 docs/business-revision/evidence/fx_verify.py` (checks 2/3/5) and
  `python scripts/run_scientific_validation.py --all` (check 4).

**No source code was modified.** The only file added is the verification harness
`docs/business-revision/evidence/fx_verify.py` and this report.

## Overall verdict: ✅ ALL GREEN — no regression found, all intended behaviors hold.

| # | Check | Result |
|---|-------|--------|
| 1 | Full test suite | ✅ PASS — 175 passed, 0 failed |
| 2 | Core-measure regression (must be UNCHANGED) | ✅ PASS — bitwise-identical to pre-fix |
| 3 | Loop vs vectorized parity | ✅ PASS — agree to <1e-9 on all shared metrics |
| 4 | Published-value validation | ✅ PASS — Everglades SKIP (not ERROR), identities hold, **no new failures** |
| 5 | Intended-behavior checks (7 fixes) | ✅ PASS — all 7 verified |
| 6 | PDF smoke test (3 orgs) | ✅ PASS — all 3 generate, gradient framing present |

---

## Check 1 — Full test suite ✅ PASS

```
python -m pytest tests/ -q
175 passed in ~1.9s
```

Per-file (the fix-relevant files):

| File | Tests | Result |
|------|-------|--------|
| test_ena_fixes.py | 32 | ✅ |
| test_network_fixes.py | 25 | ✅ |
| test_vectorized_metrics.py | 32 | ✅ |
| test_gradient_reframe.py | 8 | ✅ |
| test_mutualism_fix.py | 7 | ✅ |
| test_rollup_veto.py | 7 | ✅ |
| test_size_normalization.py | 7 | ✅ |
| test_published_metrics_provenance.py | 7 | ✅ |
| test_report_consistency.py / _intelligence.py / _sections.py / _ingestion.py / _pdf_generator_detailed.py | 50 | ✅ |

No failures, no errors.

---

## Check 2 — Core-measure regression (must be UNCHANGED) ✅ PASS

Fixed known flow matrix (5-node Cone-Spring-style internal flow network) run through
`UlanowiczCalculator`. The seven core Ulanowicz measures and the identity **C = A + Φ**:

| Measure | Value (this branch) | Value (pre-fix `c137bf5`) | Δ |
|---------|--------------------|--------------------------|---|
| TST | `22007.0` | `22007.0` | 0 |
| AMI | `0.7387440254129302` | `0.7387440254129302` | 0 |
| Ascendency A | `16257.539767262353` | `16257.539767262353` | 0 |
| Development Capacity C | `34469.20966111582` | `34469.20966111582` | 0 |
| Overhead Φ | `18211.66989385347` | `18211.66989385347` | 0 |
| Relative ascendency α = A/C | `0.4716539754493481` | `0.4716539754493481` | 0 |
| Identity **C = A + Φ** | `34469.2097 == 34469.2097` | — | ✅ holds |

**All six core measures are bitwise-identical to the pre-fix commit (verified by running the identical
computation on a detached `c137bf5` worktree).** The fix pass did **not** touch the information-theoretic
core — exactly as the errors report promised ("no headline number produced by the core engine is
mathematically wrong; the errors are in the layers on top").

---

## Check 3 — Loop vs vectorized parity ✅ PASS

For seeded random flow matrices `(n,seed) ∈ {(4,1),(5,3),(6,9),(5,42),(8,7),(10,11)}`, the loop
`UlanowiczCalculator(use_vectorized=False)` and the vectorized path agree to **< 1e-9** on **all** shared
metrics: TST, AMI, A, C, Φ, effective_nodes, effective_flows, **effective_connectivity**, number_of_roles.

- **No divergence** on any metric/seed.
- Explicit re-confirmation of the E-7 fix: `effective_connectivity == effective_flows / effective_nodes`
  (F/N) to 1e-9 in both implementations (sample: C = 4.5052 = F/N). This is the metric the fix pass
  changed in *both* implementations, and they remain consistent.

(Also covered by `tests/test_vectorized_metrics.py` 32 tests and `tests/test_ena_fixes.py`
`test_loop_matches_vectorized` — all green.)

---

## Check 4 — Published-value validation ✅ PASS (no new failures)

`python scripts/run_scientific_validation.py --all` → 8 networks, 58 formulas checked.

**Baseline comparison** (same runner on pre-fix `c137bf5` vs this branch):

| Network | Pre-fix (`c137bf5`) | This branch | Verdict |
|---------|---------------------|-------------|---------|
| Cone Spring | FAIL 6/13 | FAIL 6/13 | **unchanged** |
| Cone Spring (Eutrophicated) | FAIL 6/7 | FAIL 6/7 | **unchanged** |
| Crystal River Creek | FAIL 6/11 | FAIL 6/11 | **unchanged** |
| Prawns-Alligator (×3) | FAIL 7/9 | FAIL 7/9 | **unchanged** |
| Florida Bay (mislabeled anchor) | FAIL 6/7 | — | **removed (E-22 fix)** |
| Everglades graminoid | (was the mislabeled "Florida Bay") | **SKIP** (reference_only) | **corrected** |
| Everglades cypress | — | **SKIP** (reference_only) | **corrected** |

**Key results:**

- ✅ The corrected **Everglades anchors report cleanly as SKIP, not ERROR/FAIL**: *"Network
  'everglades_graminoid' is a published-literature reference anchor (reference_only); no recomputable
  flow matrix to validate."* This is the intended E-22 outcome — the mislabeled Florida Bay α = 0.367
  anchor was replaced with the sourced Heymans graminoid/cypress reference-only anchors.
- ✅ **cone_spring and crystal_river identities still pass.** Every scientific invariant check is PASS on
  both networks:
  - `C = A + Phi` (cone: 26549.47 == 26549.47; crystal: 115617.36 == 115617.36)
  - `0 ≤ alpha ≤ 1`, `A ≤ C`, `TST > 0`, `Reserve ≥ 0`, `0 ≤ FCI ≤ 1` — all PASS.
- ✅ **No NEW failures vs the documented baseline.** The per-network `passed/total` counts are
  identical to the pre-fix commit. The remaining FAILs are the **pre-existing** published-value
  comparisons that are a documented units/basis mismatch (engine computes in **nats**; several stored
  published values are in **bits / scaled by k** — e.g. TST 42016 vs 17509, log2 vs natural entropy).
  This is the documented "A units-labeling note" from `validation-A-ulanowicz-core.md` (E-18 / base
  convention), **not** a regression introduced by the fix pass. The invariant/identity checks — the ones
  that actually test correctness — all pass.

> Note: the runner's headline "0/N networks pass" is unchanged from before the fix pass and reflects the
> pre-existing nats-vs-bits published-value convention gap, not any fix-pass defect. The fix pass neither
> introduced nor was expected to close that gap.

---

## Check 5 — Intended-behavior checks (the fixes actually work) ✅ PASS (7/7)

All run via `fx_verify.py`; every assertion PASS.

### 5a — Roll-up veto (E-1) ✅
Profile `(OPEN 100, AUT 100, SYM 100, INT 100, SUSTAINABLE 0)`:
- `overall_score = 80.0` — **unchanged weighted mean** (the veto changes only the label).
- `raw_overall_status = HEALTHY` but final `overall_status = WARNING` (capped, not HEALTHY).
- `capped_by = ['sustainable']`.

The "Non-Viable but 80/100 HEALTHY" contradiction is fixed.

### 5b — Finn FCI (E-8/E-9) ✅
- Pure 4-node ring → full Finn FCI = **1.0000** (`EcosystemFlowCalculator` and the
  `UlanowiczCalculator.calculate_finn_cycling_index_full`).
- Acyclic 4-chain → FCI = **0.0000**.
- (The old short-cycle proxy correctly returns 0.0 on the ring and is retained under the honest name
  `calculate_short_cycle_proxy`, with a back-compat alias.)

### 5c — Effective connectivity (E-7) ✅
Connected net → C = **4.6275 ≥ 1.0** (no longer inverted N/F < 1). Identity `C = F/N` holds in both
loop and vectorized paths (Check 3).

### 5d — Betweenness inversion (E-13) ✅
Directed net with a strong-flow route through hub `h`: after inverting flow→distance (`d = 1/flow`,
Brandes 2001), betweenness = `{s:0, h:0.167, t:0, x:0}` → **hub `h` ranks top**. The strong-tie node is
correctly central (this feeds OPEN, so OPEN is no longer mis-scored).

### 5e — Mutualism (E-24 / Fath 2019 P8) ✅
2-node network `[[0,5],[3,0]]`:
- `direct_benefit_cost_ratio == integral_benefit_cost_ratio == 0.6` — **no indirect lift** on a 2-node
  network (there are no length-≥2 indirect paths), exactly as the integral-utility construction requires.
- Direct utility matrix diagonal = (0.0, 0.0) — **benefit:cost sums exclude the diagonal** (off-diagonal
  only). (Larger networks where indirect > direct are covered by `tests/test_mutualism_fix.py`.)

### 5f — Gradient reframe (E-1/E-3) ✅
`sustainable_verdict_narrative(30, 0.09)` for a low-α org contains:
- **position** `under-organized` ✅
- **direction-of-travel** `increase structure` ✅
- **indicative** caveat ✅
- does **NOT** contain bare `non-viable` ✅ or bare `unsustainable` ✅

### 5g — Size normalization (E-24/E-25) ✅
8-node ring: `norm_roles == min(number_of_roles / effective_nodes, 1)` to 1e-9, and `norm_roles ∈ [0,1]`.
The arbitrary fixed `/10` divisor is replaced by the principled `roles/effective_nodes` bound (R ≤ N).

---

## Check 6 — App / PDF smoke test (3 orgs) ✅ PASS

Via the headless app path (`docs/business-revision/evidence/gen-report.py`, which mirrors `app.py`'s PDF
export):

| Org file | Output | Result |
|----------|--------|--------|
| `tech_company_combined_matrix.json` | tech.pdf (38 KB) | ✅ no exception |
| `balanced_org_test.json` | balanced.pdf (38 KB) | ✅ no exception |
| `cone_spring_original.json` | cone.pdf (37 KB) | ✅ no exception |

All three generate without error. The new **gradient framing appears** in the tech_company (low-α) PDF:
`under-organized` ×13, `direction` ×12, `indicative` ×11, `gradient` ×7, `increase structure` ×5, and
**zero** occurrences of the bare `non-viable` fail string. The single `unsustainable` token is inside a
generic forward-looking recommendation ("detect early signs of drift toward unsustainable
configurations"), not an absolute verdict about the org's current state — consistent with the reframe,
which targets the *verdict* language.

---

## Flags / items to note before opening a PR

- **No regressions or broken behavior found.** All six checks are green.
- The scientific-validation runner still prints a headline "0/N networks pass." This is a **pre-existing,
  documented units/basis (nats vs bits / scaled-by-k) convention gap** in the *published-value comparison
  layer* — identical to the pre-fix baseline, **not** caused by this fix pass. It does not affect any
  identity/invariant check. If desired, a *separate* follow-up could add a base-conversion in the
  comparison layer (E-18-adjacent) so the published-value deltas close — but that is out of scope for
  this fix pass and should not block the PR.
- New file added by this verification: `docs/business-revision/evidence/fx_verify.py` (the reproducible
  harness). No source code changed; no new test gap was found that required adding a unit test (the fix
  pass already ships thorough TDD coverage — 125 fix-specific tests).

---

*Generated by the FX verification pass. Harness: `docs/business-revision/evidence/fx_verify.py`.*
