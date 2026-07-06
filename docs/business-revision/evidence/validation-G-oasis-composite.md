# Validation G — OASIS 5-Dimension Composite (Internal Design Logic Audit)

**Scope:** Family G (O1–O13) of `formula-inventory.md` + Issue 1 (roll-up floor) + Issue 4 (Network Efficiency def).
**Method:** Internal design-logic audit only. The OASIS composite is **proprietary** — there is *no* peer-reviewed paper defining the sub-weights, caps, or normalization constants. Where the design references established science (Fath et al. 2019; Ulanowicz 2009) the mapping is noted, but the *composition* is a product artifact and is audited for internal consistency, not literature conformance.
**Primary file:** `src/oasis_calculator.py` (read in full).
**No code was modified.** Every fix below is a recommendation only.

Legend for the "Class" column:
- **BUG** — clear logic error or code/doc mismatch; fix does not require a product decision.
- **DESIGN-CHOICE** — proprietary tuning/composition; changing it needs a product/business decision, not a bug fix.

---

## 1. Per-formula logic-audit table

| ID | Formula (loc) | Sub-weights sum to 1.0? | Logic sound? | Arbitrary constants | Class | Severity |
|----|---------------|:---:|---|---|---|:---:|
| **O1** OPEN | `0.25·conn + 0.30·normFD + 0.25·avgBetween + 0.20·clustering` `:333-338` | **YES** (0.25+0.30+0.25+0.20 = 1.00) | Sound as a convex combination. But `avgBetweenness` is a raw mean of betweenness centralities (typically << 0.1 for sparse orgs) mixed on equal footing with connectance/clustering (also small) and a *normalized* FD (0–1) — the four inputs are on **incommensurate scales**, so the 0.25 weight on betweenness contributes far less than 25% of realized variance. Weights are nominal, not effective. | `max_flow_diversity = log(n²)` normalizer (defensible: theoretical H max). Cap `0.6` (see O6). | DESIGN-CHOICE | MED |
| **O2** AUTONOMOUS | `0.35·min(FCI,1) + 0.25·recip + 0.25·normAMI + 0.15·autocat` `:406-411` | **YES** (0.35+0.25+0.25+0.15 = 1.00) | Sound convex combination. `FCI` clamped to ≤1 (fine). `normAMI = AMI/log(n²)` defensible. `reciprocity` falls back to `mutualism_ratio` when the network analyzer is absent — a **silent input substitution** (two different quantities feed the same slot depending on config). | FCI default `0.1` when missing `:386`; `autocatalytic_index` uses `·10` scaling and `expected_cycles = n(n-1)/2` `:185-188`. Cap `0.5` (O6). | DESIGN-CHOICE | MED |
| **O3** SYMBIOTIC | `0.30·(1−gini) + 0.25·min(mod,1) + 0.25·min(nodeRatio,1) + 0.20·mutualism` `:484-489` | **YES** (0.30+0.25+0.25+0.20 = 1.00) | Sound. `1−gini` correctly inverts inequality. `modularity` default `0.3` when analyzer absent `:472` — another silent substitution that props the score up. `min(·,1)` clamps guard against out-of-range. | modularity default `0.3`; cap `0.7` (O6). | DESIGN-CHOICE | MED |
| **O4** INTELLIGENT | `0.35·normRoles + 0.25·normDivers + 0.20·normRolesPerNode + 0.20·normCondEntropy` `:556-561` | **YES** (0.35+0.25+0.20+0.20 = 1.00) | Sound. All four inputs pre-normalized to 0–1, so this dimension is the most scale-consistent of the five. | `roles/10` `:538`, `rolesPerNode/2` `:548` (see O12). Cap `0.6` (O6). | DESIGN-CHOICE | LOW-MED |
| **O5** SUSTAINABLE | **code:** `0.30·normRob + 0.20·inWindow + 0.20·normRegen + 0.30·alphaOpt` `:633-638` | **YES** (0.30+0.20+0.20+0.30 = 1.00) | Convex combination sound. **BUG: docstring/code mismatch** — the docstring `:599-600` states `0.30·robustness + 0.25·is_in_window + 0.20·regen + 0.25·alpha_optimality` (0.30/0.25/0.20/0.25). Code uses 0.30/**0.20**/0.20/**0.30**. Both sum to 1.0, but the *published* weights differ from the *executed* weights. `is_in_window` weight is documented at 0.25 but runs at 0.20; `alpha_optimality` documented 0.25 but runs 0.30. Anyone auditing from the docstring gets the wrong model. | `regen/0.3` `:619`, `fitness/0.4` `:630` (computed but unused in score), α-target `0.37` `:623`. Cap `0.8` (O6). | **BUG** (docstring) + DESIGN-CHOICE (weights) | MED |
| **O6** normalize-to-100 + per-dim caps | `_normalize_to_100(raw, 0, cap)` with caps OPEN 0.6 / AUT 0.5 / SYM 0.7 / INT 0.6 / SUS 0.8 `:99-104, 341, 414, 492, 564, 641` | n/a | See §3 — caps are **undocumented, un-sourced**, differ per dimension, and cause saturation. | five caps `0.5–0.8` | DESIGN-CHOICE | **HIGH** |
| **O7** overall roll-up | `Σ dim·weight`, 20% each `:695-698` | **YES** (5×0.20 = 1.00) | Flat weighted arithmetic mean. Mathematically fine; **the problem is what it enables** — see §2 (Issue 1). No floor/veto. | equal 0.20 weights | DESIGN-CHOICE | **CRITICAL** (via O8) |
| **O8** overall band | HEALTHY ≥60 / WARNING ≥40 / CRITICAL `:713-718` | n/a | **Bands the mean independently of per-dimension status (O9).** This is the mechanism of the Issue-1 bug: a single dimension at 0 cannot pull the overall out of HEALTHY if the other four are strong. | 60 / 40 thresholds | **BUG** (see §2) | **CRITICAL** |
| **O9** per-dim asymmetric thresholds | `HEALTH_THRESHOLDS` `:50-56`, applied `:701-708` | n/a | Per-dimension healthy/warning/critical bands are asymmetric (e.g. SUSTAINABLE healthy≥60, AUTONOMOUS healthy≥40). Internally consistent, but the thresholds themselves are un-sourced tuning. `get_status` only checks the **lower** bound of each band `:703-708`; the upper bounds in the tuples (e.g. `(50,85)`) are **never used** — dead data. | 15 threshold constants | DESIGN-CHOICE | MED |
| **O10** α-optimality | `max(0, 1 − |α−0.37|/0.37)` `:624-626` | n/a | Logic sound: triangular kernel peaking at α=0.37, reaching 0 at α=0 and α=0.74. **Asymmetric penalty**: because it divides by 0.37, α above 0.74 is floored at 0 while the window of viability extends to 0.60 — fine, but the "0" region (α≥0.74) coincides with over-rigid systems, which is intended. Uses `0.37` not `1/e=0.3679` (see O5/Issue 3). | target `0.37` | DESIGN-CHOICE | LOW |
| **O11** norm_robustness | `R / (1/e)` `:609-610` | n/a | **CORRECT.** R = −α·ln(α) has its maximum 1/e at α=1/e, so R/(1/e) ∈ [0,1] with 1.0 exactly at the robustness optimum. Proper normalization to the theoretical max. Consistent with the (natural-log) robustness in `ulanowicz_calculator.py:549`. | `1/e` (theoretical, not arbitrary) | ✅ VALID | — |
| **O12** sub-metric norm constants | `roles/10` `:538`, `rolesPerNode/2` `:548`, `regen/0.3` `:619`, `fitness/0.4` `:630`, autocat `·10` `:188` | n/a | Each divisor asserts an "expected max" for the metric with **no cited basis**. `roles/10` assumes 10 roles is the ceiling; `regen/0.3` assumes regen capacity tops at 0.3; `fitness/0.4` matches the theoretical max of the Ulanowicz-2009 fitness fn (defensible) but `fitness` is **computed and never used** in the score `:629-630, 656-657`. | 10, 2, 0.3, 0.4, 10 | DESIGN-CHOICE | MED |
| **O13** recommendation triggers | α<0.2 / α>0.6 CRITICAL `:920,928`; gini>0.5 `:896`; roles<3 `:907`; open<50/<30, auto<40/<25 `:875,885` | n/a | Trigger thresholds are internally reasonable (α<0.2/>0.6 matches the window-of-viability bounds 0.20–0.60) but the score cutoffs (50/30/40/25) are un-sourced. **Note:** recommendation logic keys off *raw metrics* (α, gini, roles), not the dimension *scores* — so it can fire correctly even when O6 saturation hides the problem in the headline score. This is actually a partial mitigation of Issue 1 (the CRITICAL SUSTAINABLE recommendation still appears), but it does **not** fix the overall HEALTHY label. | 0.2, 0.6, 0.5, 3, 50, 30, 40, 25 | DESIGN-CHOICE | MED |

### Weights-sum-to-1.0 summary (Audit question 1)
**All five dimensions (O1–O5) have sub-weights that sum to exactly 1.00.** No dimension silently biases its own score through mis-summed weights. The dimension roll-up (O7) also sums to 1.00 (5×0.20). The **only** weight-related defect is the **O5 docstring/code mismatch** (documented 0.30/0.25/0.20/0.25 vs executed 0.30/0.20/0.20/0.30) — a documentation BUG, not a scoring bug (the executed weights still sum to 1.0).

---

## 2. The Roll-up Floor Problem (Issue 1) — KEY DELIVERABLE

### 2.1 Confirmation of the bug

**Confirmed.** O7 (`:695-698`) computes a flat weighted arithmetic mean; O8 (`:713-718`) bands that mean with fixed cutoffs (≥60 HEALTHY) **completely independently** of the per-dimension status computed in O9 (`:701-708`).

Worked example (the audit's stated scenario):

```
OPEN=100, AUTONOMOUS=100, SYMBIOTIC=100, INTELLIGENT=100, SUSTAINABLE=0
overall = 0.20·100 ×4 + 0.20·0 = 80.0
O8:  80 ≥ 60  →  overall_status = "HEALTHY"
O9:  sustainable=0  →  dimension_status['sustainable'] = "CRITICAL"  (0 < 40)
```

**Result: a system whose sustainability dimension is CRITICAL (score 0) is reported as overall HEALTHY (80/100).** There is **no floor, no veto, and no worst-dimension rule** anywhere in `get_oasis_profile`. The arithmetic mean lets four strong dimensions fully mask one collapsed dimension.

Why this matters more than an edge case: **SUSTAINABLE is the Window-of-Viability dimension** — in the underlying ecological model (Ulanowicz 2009; Fath 2019 Principle 6) a system outside the window of viability is, by definition, **non-viable**. Non-viability is not a weakness to be averaged away; it is a *necessary condition for survival*. Averaging it against OPEN/AUTONOMOUS/etc. is a category error: you cannot compensate for being non-viable by being well-connected. **A Non-Viable organization can currently be labeled HEALTHY.** That is the #1 business-credibility risk in the composite.

This is compounded by the O6 saturation effect (§3): the audit observed **3 dimensions pinned at 100/100**. Saturation makes the masking *worse* — the four "carrier" dimensions are not just high, they are maxed, so SUSTAINABLE has to fall essentially to 0 before the mean even dips below the 60 HEALTHY line, and even then it lands in WARNING, never CRITICAL, no matter how catastrophic SUSTAINABLE is.

**Class:** BUG for the *label*, DESIGN-CHOICE for *which* fix. The averaging itself is a legitimate design; the absence of any viability gate is the defect.

### 2.2 Design-fix options (pick one — product decision required)

#### Option A — CRITICAL floor / veto rule (minimal, recommended)
Keep the weighted mean as the *score*, but override the *status* so the overall status can never outrank the worst dimension by more than one band, with a hard rule: **overall status cannot be HEALTHY if any dimension is CRITICAL.**

Sketch (design only, not implemented):
```
worst = min over dimensions of status-rank
overall_status = band(overall)          # existing 60/40 logic
if any dimension is CRITICAL:  overall_status = at most WARNING
if SUSTAINABLE is CRITICAL:    overall_status = CRITICAL   # viability veto
```
- **Pros:** Smallest change; preserves the familiar 0–100 headline number; directly kills the "Non-Viable = HEALTHY" case; transparent and explainable to clients ("we never call you healthy while a pillar is critical"). Leaves the per-dimension scores and all downstream visuals untouched.
- **Cons:** The headline *number* (80) still looks good next to a CRITICAL label — mild score/label dissonance. Requires defining the veto policy (any-CRITICAL vs SUSTAINABLE-only). Two knobs to govern (score vs status) instead of one.
- **Best when:** you want to ship a credibility fix fast without re-tuning the whole composite.

#### Option B — Geometric or harmonic mean roll-up
Replace the arithmetic mean in O7 with a **geometric mean** (`(∏ dimᵢ^wᵢ)`) or **harmonic mean**. Both are dominated by their *smallest* input, so a single collapsed dimension drags the overall down hard.

Worked example, geometric mean, same inputs (100,100,100,100,0):
```
geo = (100·100·100·100·0)^(1/5) = 0     → overall 0 → CRITICAL
```
Even a milder case (100,100,100,100,20) gives geo ≈ 45.9 (WARNING) vs arithmetic 84 (HEALTHY).
- **Pros:** Mathematically principled — encodes "all dimensions must be adequate" (Cobb-Douglas / low-substitutability semantics), which matches the ecological reality that viability is non-compensatory. No separate veto policy needed; the aggregation itself enforces the floor.
- **Cons:** Harsh at zero (any dimension at exactly 0 → overall 0), which needs a small floor (e.g. clamp inputs to ≥1) to avoid a discontinuity. Changes the meaning and distribution of the headline number → re-baselines every historical/benchmark score; the 60/40 O8 bands would need re-calibration. Larger blast radius across reports and benchmarks.
- **Best when:** you're willing to re-baseline and want the aggregation to *inherently* express non-substitutability.

#### Option C — Gate SUSTAINABLE as a necessary condition (multiplicative gate)
Treat SUSTAINABLE (Window of Viability) as a **gating multiplier** on an arithmetic mean of the other four:
```
core = mean(OPEN, AUTONOMOUS, SYMBIOTIC, INTELLIGENT)
overall = core · g(SUSTAINABLE)     where g rises from 0→1 across the viability band
```
- **Pros:** Encodes the strongest theoretical claim — nothing else counts if you are non-viable — while leaving the other four fully compensatory among themselves. Directly mirrors Fath Principle 6 as a *precondition*, not a co-equal average term.
- **Cons:** Singles out one dimension as special → a design/philosophical commitment the business must own and defend. Discards the "20% each" symmetry. If SUSTAINABLE's own O6 cap/saturation is noisy, the gate inherits that noise and can over-penalize. Requires designing the shape of `g()`.
- **Best when:** the product's core thesis is explicitly "viability first."

### 2.3 Recommendation
**Adopt Option A now** (CRITICAL floor + SUSTAINABLE viability veto) as a BUG-class credibility fix — it is the smallest change that eliminates "Non-Viable labeled HEALTHY," and it is fully explainable to clients. Evaluate **Option B (geometric mean)** as a subsequent DESIGN-CHOICE if/when the business is prepared to re-baseline scores and re-calibrate the O8 bands, because it fixes the masking at the aggregation layer rather than patching the label. Reserve **Option C** for the case where "viability-first" becomes an explicit brand promise. **Do not implement without a product decision** — A changes only the status logic (low risk), B and C change the headline number (require re-baselining and re-calibration of O6 caps and O8 bands).

---

## 3. Normalization caps (O6) — saturation assessment (Audit question 3)

Each dimension's raw 0–1 weighted sum is mapped to 0–100 via `_normalize_to_100(raw, 0, cap)` where **cap** differs per dimension:

| Dimension | Cap | Raw value that yields 100 |
|---|---|---|
| OPEN | 0.6 `:341` | raw ≥ 0.60 → 100 |
| AUTONOMOUS | 0.5 `:414` | raw ≥ 0.50 → 100 |
| SYMBIOTIC | 0.7 `:492` | raw ≥ 0.70 → 100 |
| INTELLIGENT | 0.6 `:564` | raw ≥ 0.60 → 100 |
| SUSTAINABLE | 0.8 `:641` | raw ≥ 0.80 → 100 |

**Are the caps justified?** No source, comment, or derivation accompanies any of the five values. They are **arbitrary tuning constants** chosen so that "realistic" raw scores spread across 0–100. Because the caps are well below 1.0 (the theoretical max of each convex combination), **any organization whose raw score reaches the cap saturates at exactly 100** and all further differentiation above the cap is discarded.

**Do they cause saturation?** Yes, and the audit's field observation (**3 dimensions at 100/100**) is the direct symptom. AUTONOMOUS has the lowest cap (0.50) — its four inputs (FCI clamped to ≤1, reciprocity, normAMI, autocat) only need to average 0.50 to peg at 100, and with the `modularity` and `reciprocity`/`FCI` **defaults** (0.3, 0.1) plus the autocat `·10` amplifier, mid-range networks routinely clear it. SYMBIOTIC (cap 0.70) is inflated by the `modularity=0.3` default and `1−gini` (which is high whenever flows are even). OPEN/INTELLIGENT (0.60) saturate whenever normalized inputs cluster near their own maxima.

**Consequence — this is the mechanism that lets the roll-up mask SUSTAINABLE.** Saturation pins the four "carrier" dimensions at or near 100, so in the O7 mean the only remaining variance lives in SUSTAINABLE (which has the *highest* cap 0.80 and is therefore hardest to saturate). The mean is then `≈ (4·100 + SUS)/5 = 80 + SUS/5`, which stays ≥60 (HEALTHY) for **any** SUSTAINABLE ≥ −100, i.e. always. Saturation + arithmetic mean = SUSTAINABLE can never move the overall out of HEALTHY on its own. **§3 and §2 are the same bug viewed from two angles.**

**Class:** DESIGN-CHOICE (the caps are proprietary tuning), but they are **un-justified** and should be either (a) documented with an empirical basis (percentile of a reference corpus), or (b) replaced by a principled normalization (e.g. cap = theoretical max of each convex combination, or a corpus-derived P95), and (c) revisited jointly with the Issue-1 fix, since fixing the mean without de-saturating the inputs only half-solves the masking.

---

## 4. Network Efficiency definition (Issue 4) — resolution

**Confirmed code/doc mismatch.**

| Source | Definition | Location |
|---|---|---|
| **Engine (authoritative)** | `network_efficiency = A / C = α` | `ulanowicz_calculator.py:562, 567-570` |
| Vectorized engine | `network_efficiency = relative_ascendency` (alias) | `vectorized_metrics.py:508` |
| In-app docs registry | `Efficiency = α = A / C` | `docs_registry.py:432` |
| PDF KPI / tables | consume `metrics['network_efficiency']` (= α) directly | `pdf_generator.py:399, 696-697` |
| **Report Appendix (OUTLIER)** | `Network Efficiency: A / (C x log2(n))` | `publication_report.py:432` |

**Which is intended?** The **engine value (`A/C = α`) is correct**; the **Appendix text is the outlier and is wrong.** Three independent confirmations:

1. **The bands only work with α.** The efficiency assessment `_assess_efficiency` (`ulanowicz_calculator.py:1256-1263`) classifies LOW <0.2 / OPTIMAL / HIGH >0.6, and `_assess_robustness` reuses the same 0.2/0.6 cutoffs (`:1238-1240`). These bands are the empirically-derived Window-of-Viability bounds (0.20–0.60) which apply to **α ∈ [0,1]**. The Appendix formula `A/(C·log2 n)` divides α by `log2(n)`, which for any n>4 shrinks the value well below 0.2 — so a perfectly viable α=0.4 network would register as LOW efficiency. The bands are meaningless unless the fed value is α. The engine feeds α. Therefore α is intended.
2. **Every other doc surface agrees on α** (`docs_registry.py`, the α-based interpretations, `publication_report.py:420` which itself defines `alpha = A/C`). Only the one Appendix line at `:432` diverges — and it even contradicts its *own* report, which lists `alpha = A/C` twelve lines earlier at `:420`.
3. **`A/(C·log2 n)` is not a standard Ulanowicz quantity.** α = A/C ("relative ascendency" / "degree of order") is the canonical efficiency ratio in Ulanowicz (2009). The `log2(n)` divisor appears to be a stray conflation with the *redundancy* normalizer `H_max = log2(n)` used two lines up at `:426`.

**Recommendation (DESIGN-CHOICE / documentation BUG):** Make code and docs agree by **fixing the Appendix text** `publication_report.py:432` to read `Network Efficiency: α = A / C` (matching the engine and every other surface). Do **not** change the engine. This is a BUG-class doc fix — the executed number is already correct; only the printed Appendix formula is wrong and could mislead an analyst reproducing the calc. (Not implemented per audit-only scope.)

---

## 5. Minor consistency flags

- **O10/Issue 3 — 0.37 vs 1/e:** `alpha_optimality` (`:623`) and `regenerative_capacity` docs target **0.37**, while `norm_robustness` (`:609`) uses the exact **1/e = 0.36788**. These differ by ~0.6%. `0.37` is a rounded presentation of 1/e; harmless numerically, but for internal consistency the two should reference the same constant (recommend `1/math.e` everywhere, or document 0.37 as "≈1/e"). DESIGN-CHOICE / LOW.
- **Unused computed values:** `fitness_for_evolution`/`norm_fitness` (`:629-630`) and the upper bounds of `HEALTH_THRESHOLDS` tuples (`:51-55`) are computed/stored but never affect any score or status. Dead logic — flag for cleanup, no scoring impact.
- **Silent input substitutions:** O2 reciprocity → mutualism fallback (`:389-394`); O3 modularity default 0.3 (`:472`); O2 FCI default 0.1 (`:386`). When the network analyzer is absent these defaults **inflate** scores toward the (already low) saturation caps. Flag: the composite behaves differently with vs without the network analyzer, and the difference is upward-biasing.

---

## 6. Severity roll-up

| Finding | Class | Severity |
|---|---|---|
| Issue 1 — roll-up has no floor/veto; Non-Viable → HEALTHY (§2) | BUG (label) | **CRITICAL** |
| O6 caps arbitrary + cause saturation, which enables the masking (§3) | DESIGN-CHOICE (unjustified) | **HIGH** |
| Issue 4 — Appendix `A/(C·log2 n)` contradicts engine `α=A/C` (§4) | BUG (doc) | **HIGH** (mislead risk; number is correct) |
| O5 docstring weights ≠ code weights (§1) | BUG (doc) | MED |
| O12/O13 magic numbers un-justified (§1) | DESIGN-CHOICE | MED |
| O1 scale-incommensurate inputs (nominal ≠ effective weights) | DESIGN-CHOICE | MED |
| Silent input substitutions inflate scores (§5) | DESIGN-CHOICE | MED |
| O10 0.37 vs 1/e inconsistency; dead computed values (§5) | DESIGN-CHOICE | LOW |
| O11 norm_robustness = R/(1/e) | ✅ VALID | — |
| Weights O1–O5 all sum to 1.0 | ✅ VALID | — |

## 7. Arbitrary magic numbers inventory (all "needs empirical justification")

| Constant | Value | Location | Role |
|---|---|---|---|
| OPEN cap | 0.6 | `:341` | normalization ceiling |
| AUTONOMOUS cap | 0.5 | `:414` | normalization ceiling |
| SYMBIOTIC cap | 0.7 | `:492` | normalization ceiling |
| INTELLIGENT cap | 0.6 | `:564` | normalization ceiling |
| SUSTAINABLE cap | 0.8 | `:641` | normalization ceiling |
| roles divisor | 10 | `:538` | "expected max roles" |
| roles-per-node divisor | 2 | `:548` | "expected max roles/node" |
| regen divisor | 0.3 | `:619` | "expected max regen" |
| fitness divisor | 0.4 | `:630` | fitness norm (unused in score) |
| autocatalysis amplifier | ·10 | `:188` | `min(1, cycle_flow_ratio·10)` |
| FCI missing-default | 0.1 | `:386` | fallback |
| modularity missing-default | 0.3 | `:472` | fallback |
| overall bands | 60 / 40 | `:713-717` | O8 status cutoffs |
| per-dim thresholds | 15 values | `:50-56` | O9 status cutoffs |
| α-optimality target | 0.37 | `:623` | vs 1/e=0.3679 |
| recommendation triggers | 0.2, 0.6, 0.5, 3, 50, 30, 40, 25 | `:875-928` | O13 |

*None of the caps, divisors, amplifiers, or defaults carries a citation or derivation. The two genuinely theoretical constants — `1/e` in O11 (`:609`) and `0.4` matching the Ulanowicz-2009 fitness max (`:630`) — are the exceptions and are sound (though `0.4`'s consumer is unused).*
