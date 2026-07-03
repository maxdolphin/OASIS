# Validation I & H — Stored Published Reference Values + Report-Layer Verdict Bands

**Scope:** families I (P1–P9) and H (H2–H13) of `formula-inventory.md`.
**Method:** validation only — no source modified. Arithmetic identities computed in Python;
paper cross-checks against the local `_papers/` corpus.
**Working dir:** `/Users/massimomistretta/Claude_Projects/Adaptive_Organization`
**Files audited:**
`src/services/published_metrics_db.py`, `src/services/scientific_validation_agent.py`,
`src/services/new_metric_checklist.py`, `src/publication_report.py`, `src/pdf_generator.py`,
`src/latex_report_generator.py`, `src/report_intelligence.py`, `src/ecosystem_flow_calculator.py`,
`src/main.py`.

**Severity legend:** CRITICAL (wrong number / broken identity) · MAJOR (mislabelled source or
cross-file contradiction) · MINOR (cosmetic / rounding / documentation) · OK.

---

## Part 1 — Published reference values: internal identities + paper match

All values computed from the stored numbers in `published_metrics_db.py`.

| Network (id) | Stored α | A | C | Φ | TST | A+Φ=C? | α=A/C? | A≤C? | Matches paper? | Severity |
|---|---|---|---|---|---|---|---|---|---|---|
| **cone_spring_original** `:58` | 0.505 | 68191 | 135000 | 66809 | 42016 | ✅ 68191+66809=135000 (0.000%) | ✅ 0.50512≈0.505 | ✅ | citation-only (Ulanowicz&Norden 1990 not in corpus); AMI=A/TST=1.6230✅, H=C/TST=3.2131✅ all self-consistent | **OK** |
| **cone_spring_eutrophicated** `:111` | 0.529 | — | — | — | — | n/a (only α stored) | n/a | n/a | ✅ **CONFIRMED** — Ulanowicz 2009 p5: *"The ensuing value of a is 0.529 (>a_opt)"*; note "optimal 0.460" matches p5 *"a_opt (≈0.460)"* & p5 *"a=0.4596"* | **OK** |
| **crystal_river_creek** `:135` | 0.552 | 112891 | 204355 | 91464 | 97916 | ✅ 112891+91464=204355 (0.000%) | ✅ 0.55243≈0.552 | ✅ | citation-only (Ulanowicz 1986 book, not in corpus); internally consistent | **OK** |
| **florida_bay** `:179` | 0.367 | — | — | — | — | n/a (only α stored) | n/a | n/a | ❌ **NOT FOUND / MISLABELLED** — see finding below | **MAJOR** |
| **prawns_alligator_original** `:201` | — | 53.9 | (implied 175.2) | 121.3 | 102.6 | n/a (C not stored) → A+Φ=175.2 | implied α=0.3076 | ✅ 53.9≤175.2 | citation-only (Ulanowicz 2009 Fig.1); self-consistent | **OK** |
| **prawns_alligator_efficient** `:235` | — | 100.3 | (implied 100.3) | 0.0 | 121.8 | n/a → A+Φ=100.3 | implied α=1.000 (Φ=0) | ✅ | citation-only (Fig.2); Φ=0 → zero-reserve case, internally consistent | **OK** |
| **prawns_alligator_adapted** `:270` | — | 44.5 | (implied 112.7) | 68.2 | 99.7 | n/a → A+Φ=112.7 | implied α=0.3949 | ✅ 44.5≤112.7 | citation-only (Fig.3); self-consistent | **OK** |

### Internal-identity verdict
**Every stored published value passes its own internal identities.** No data-entry bug in the
arithmetic: `A+Φ=C` holds to 0.000% where all three are stored (cone_spring_original,
crystal_river_creek); `α=A/C` holds to <0.001 in both cases; `A≤C` holds everywhere; Φ≥0 and TST>0
everywhere. The cone-spring AMI (1.623) and H (3.213) also reproduce exactly from A/TST and C/TST.

### P4 florida_bay — MAJOR: value cannot be sourced and label is wrong
- Stored: `florida_bay`, `source="Heymans et al. 2002"`, α=**0.367**, note "subtropical
  seagrass-dominated ecosystem / shallow marine environment."
- **The cited paper (`_papers/Heymans.pdf`) is not about Florida Bay.** Its title is *"Network
  analysis of the South Florida Everglades **graminoid marshes** and comparison with nearby
  **cypress** ecosystems"* (Heymans, Ulanowicz, Bondavalli — Ecological Modelling 149, 2002).
- The relative-ascendancy values it actually reports (p15) are **graminoids ≈ 52%** and
  **cypress ≈ 34%** — *"the relative ascendancy of 52% for the graminoids… the relative
  ascendancy of 34% reported for the cypress."* **Neither is 0.367.**
- A corpus-wide regex sweep for any α≈0.36–0.37 tied to "ascendency/relative/Florida Bay/bay"
  returned **zero matches** across all 11 PDFs.
- The stored ecosystem description ("seagrass-dominated / shallow marine") does not match the
  freshwater graminoid marsh / cypress swamp of the cited paper.
- **Coincidence to flag, not confirm:** 0.367 is numerically identical to the robustness optimum
  `1/e = 0.367879…` hard-coded elsewhere (`report_intelligence.py:15`, `oasis_calculator.py:609`).
  This raises the possibility the "Florida Bay α" was transcribed from the robustness-optimum
  constant rather than from a measurement. **Not asserted — flagged for human source-tracing.**
- **Verdict:** citation does not support the stored number, and the network label/description do
  not match the cited paper. **MAJOR** (a benchmark anchor with an unverifiable/likely-wrong value
  and a mismatched source). Recommend replacing with the paper's actual graminoid (0.52) or
  cypress (0.34) figure and correcting the label, OR sourcing a genuine Florida Bay α from
  Ulanowicz et al. 1998 (not in corpus) — pending human decision. **No code changed.**

### P9 EXAMPLE_METRICS (`new_metric_checklist.py:459-500`)
Embedded values `cone_spring_original`: A=68191, C=135000, α=0.505; `cone_spring_eutrophicated`
α=0.529; `crystal_river_creek` α=0.552; `prawns_alligator_original` A=53.9. **All match the
primary DB and their own identities. OK.** (Florida Bay α is NOT embedded here, so the P4 issue is
localized to `published_metrics_db.py`.)

---

## Part 2 — Base & conversion (P6, P7)

| Item | Finding | Severity |
|---|---|---|
| **P7 log2↔ln direction** | ✅ **CORRECT.** Engine (`UlanowiczCalculator`) computes information terms in **nats** (natural log). `scientific_validation_agent._compute_metrics` (`:159-169`) converts to bits for LOG2 papers by **dividing by ln2** (`metric / ln2`, `ln2=math.log(2)`). Since log2(x)=ln(x)/ln(2) and ln(2)<1, dividing nats by ln2 *increases* the magnitude → nats→bits, the correct direction. Sanity check: cone-spring published AMI is 1.623 **bits**; 1.623 bits = 1.125 nats, and 1.125/ln2 = 1.623 bits ✓. | **OK** |
| **P6 tolerances** | Default 0.05 (`:44`); crystal_river & florida_bay 0.10 (`:140,183`); fundamental C=A+Φ 0.001 (`:388` and re-checked at `scientific_validation_agent.py:231`). WARNING band = within 2× tolerance (`:202`). Internally consistent with the inventory. | **OK** |
| **Base tagging** | cone_spring_original & crystal_river_creek tagged `LogBase.LOG2`; eutrophicated, florida_bay, all prawns tagged `LogBase.NATURAL`. The prawns/eutrophicated networks store only **α** (a dimensionless ratio) and/or TST — α is **base-invariant** (A/C cancels the log base), so their NATURAL tag is harmless for α comparison. Note the crystal_river base is a stated *assumption* (`:139` "Assumes log base 2 based on era"), not paper-confirmed; with tol 0.10 this is acceptable but should stay flagged as assumption. | **MINOR** (crystal base is assumed) |
| **P8 invariants** | 0≤α≤1, A≤C, TST>0, Φ≥0, 0≤FCI≤1 defined at `:391-414` and enforced at `scientific_validation_agent.py:240-302`. All stored published values satisfy them. | **OK** |

**Base conclusion:** the log2/ln conversion is applied in the **right direction**, so published-value
comparisons are not systematically wrong. This was the highest-risk item and it passes.

---

## Part 3 — Report-layer verdict bands: cross-file self-consistency

### 3a. α / "network efficiency" bands — CONTRADICTION (the same quantity, three verdict schemes)

`network_efficiency` is an explicit alias of α=A/C (`vectorized_metrics.py:508`; engine
`calculate_network_efficiency` returns A/C). Yet three files re-band **this same α** with different
thresholds *and opposite value-framing*:

| File / method (H-id) | Thresholds on α | Labels (low→high α) | Framing of HIGH α |
|---|---|---|---|
| `publication_report._categorize_efficiency` (H5) `:645-651` | 0.2 / 0.4 / 0.6 | Low → Moderate → High → **Very High** | **positive** (higher = better) |
| `latex_report_generator._categorize_efficiency` (H5) `:376-383` | 0.2 / 0.4 / 0.6 | Low → Moderate → High → **Very High** | **positive** (identical to publication) |
| `publication_report._interpret_position` (H8) `:668-680` | 0.2 / 0.35 / 0.45 / 0.6 | Under-organized → Developing → **Optimal** → Efficient → **Over-constrained** | **negative** (higher = pathological) |
| `report_intelligence.build_risk_view` (H9) `:110-158` | band [0.2, 0.6] + 0.05 edge | under-organized / balanced / **over-organized (HIGH risk)** | **negative** (α>0.6 = brittle) |
| `pdf_generator` eff_status (H5-variant) `:400` | 0.2 ≤ eff ≤ 0.6 → "Optimal" else "Sub-optimal" | Sub-optimal / Optimal / Sub-optimal | **balanced** (band = good) |
| `main.py` CLI (H13) `:166-172` | 0.2 / 0.6 | underutilized / sustainable / **over-optimized & brittle** | **negative** (α>0.6 = brittle) |

**Contradiction (MAJOR).** Take α = 0.65 (over the viability window):
- `_categorize_efficiency` → **"Very High"** efficiency, presented as a **strength**.
- `_interpret_position` → **"Over-constrained"** (a problem).
- `build_risk_view` / `main.py` → **over-organized / brittle**, emitted as **HIGH-severity risk**.
- `pdf_generator` → **"Sub-optimal."**

So one section of the very same report can call α=0.65 "Very High (good) network efficiency" while
another section (and the risk register) flags it as over-organized and brittle. This is exactly the
"α=0.4 called High efficiency in one file vs sub-optimal in another" class of contradiction the task
asked to surface — and it is real for the **upper** tail (α>0.6). It is the presentation-layer
manifestation of **Issue 4** (efficiency mislabelled) compounded by an ascendency-is-monotonically-
good framing in `_categorize_efficiency` that contradicts the Ulanowicz window logic used everywhere
else. **Recommend** aligning `_categorize_efficiency`/`_categorize` labels with the window model so
α>0.6 is not called "Very High (good)". **No code changed.**

Note also `_categorize_efficiency` (0.2/0.4/0.6) and `_interpret_position` (0.2/0.35/0.45/0.6) use
**different breakpoints** for the same axis, so their zone boundaries don't even line up
(e.g. α=0.42 is "High" efficiency but only "Developing" position).

### 3b. Robustness R bands — INCONSISTENT thresholds across files

R = −α·ln(α), theoretical max ≈0.531 at α=1/e.

| File / method (H-id) | Thresholds on R | Labels |
|---|---|---|
| `publication_report` strengths/risks (H4) `:283-288` | 0.20 / 0.15 | >0.20 strong · >0.15 adequate · else low |
| `pdf_generator` rob_status (H4) `:398` | 0.2 / 0.15 | >0.2 High · >0.15 Moderate · else Low |
| `latex_report_generator` text (H4) `:265-266` | **0.25** / 0.15 | >0.25 "exceeds high-resilience threshold" · >0.15 approaches · else below |
| `main.py` CLI (H13) `:174-179` | 0.1 / 0.25 | <0.1 lacks robustness · >0.25 strong |

**Contradiction (MINOR→MAJOR).** The "high robustness" threshold is **0.20** in
publication_report and pdf_generator, but **0.25** in latex_report_generator and main.py. A system
with R=0.22 is reported as **"strong/High robustness"** by the ReportLab PDF path and as **below the
high-resilience threshold** by the LaTeX path and CLI — a direct cross-file disagreement on the same
metric. Since all four are stated to the reader as resilience verdicts, this is user-visible;
graded **MAJOR** for the 0.20-vs-0.25 split (the 0.15 lower rung is consistent).

### 3c. Gini / redundancy bands — consistent

| Quantity | Files | Thresholds | Verdict |
|---|---|---|---|
| Gini (H6) | `publication_report:235`, `pdf_generator:852-854` | 0.3 / 0.6 → equal / moderate / high inequality | ✅ **consistent** across both files |
| Redundancy (H7) | `publication_report:707-713` | 0.3 / 0.6 → Low / Moderate / High backup | ✅ single source, internally coherent |
| Overhead ratio Φ/C (H4-adjacent) | `publication_report:290-293` | >0.4 substantial · <0.3 limited | OK (no competing file) |

### 3d. H12 ecosystem-health bands (ecosystem_flow_calculator) — self-contained

Respiration 0.3/0.6/0.7, FCI 0.1/0.2/0.5, import 0.2/0.5 (`:237-261`). These band **flow-based**
quantities (respiration_ratio, finn_cycling_index, import_dependency), not α, and appear only in this
one module. **No cross-file contradiction.** Note the "overall_health HEALTHY" rule uses
`respiration_ratio < 0.6` while the per-axis energy band calls <0.3 "HIGH efficiency" / >0.7 "LOW" —
the 0.6 cutoff sits inside the "MODERATE" zone, which is coherent (not contradictory). OK.

### 3e. H2 — Network Efficiency definition mismatch (documentation vs engine)

`publication_report.py` Appendix defines *"Network Efficiency = A/(C·log₂ n)"* while the engine
computes `network_efficiency = A/C = α` (no log₂n factor; alias at `vectorized_metrics.py:508`).
Confirmed as **Issue 4** in the inventory — the printed methodology and the computed value are **not
the same expression**. The bands in 3a all operate on the engine's A/C value, so the Appendix text is
the outlier. **MAJOR** (documented formula contradicts the number the reader sees). No code changed.

---

## Cross-file band-consistency summary table

| Quantity | Consistent across files? | Discrepancy | Severity |
|---|---|---|---|
| α / network_efficiency | ❌ NO | Very-High(good) vs Over-constrained/brittle(bad) for same α>0.6; breakpoints 0.2/0.4/0.6 vs 0.2/0.35/0.45/0.6 | **MAJOR** |
| Robustness R "high" threshold | ❌ NO | 0.20 (publication/pdf) vs 0.25 (latex/main.py) | **MAJOR** |
| Robustness R "low/adequate" (0.15) | ✅ yes | — | OK |
| Gini inequality (0.3/0.6) | ✅ yes | — | OK |
| Redundancy backup (0.3/0.6) | ✅ yes | single source | OK |
| Viability window α ∈ [0.2,0.6] | ✅ yes | consistent everywhere (report_intelligence, oasis, main.py, pdf) | OK |
| Network-Efficiency definition (text vs engine) | ❌ NO | A/(C·log₂n) documented vs A/C computed | **MAJOR** (Issue 4) |
| Ecosystem-health flow bands (H12) | ✅ n/a | self-contained module | OK |

---

## Bottom line

1. **Internal identities:** ALL stored published values pass `A+Φ=C` (0.000%), `α=A/C` (<0.001),
   `A≤C`, `Φ≥0`, `TST>0`. No arithmetic/data-entry bug. cone-spring AMI & H also reproduce exactly.
2. **Paper match:** cone_spring_eutrophicated (α=0.529, opt 0.460/0.4596) is **confirmed verbatim**
   in Ulanowicz 2009. cone_spring_original, crystal_river_creek, and all prawns are **citation-only**
   (papers not in corpus) but internally consistent. **florida_bay (α=0.367) is MAJOR**: the cited
   Heymans 2002 paper is about Everglades graminoid/cypress (α = 0.52 / 0.34), reports no 0.367, and
   the stored "seagrass/marine" description doesn't match; 0.367 suspiciously equals 1/e used
   elsewhere.
3. **log2/ln conversion:** **CORRECT direction** (nats ÷ ln2 → bits); published-value comparisons are
   not systematically wrong.
4. **Verdict bands:** **two genuine cross-file contradictions** — (a) α/efficiency framing (Very-High
   "good" vs over-organized "brittle" for the same α>0.6) and (b) robustness "high" threshold 0.20
   vs 0.25 — plus the documented Network-Efficiency formula (A/(C·log₂n)) contradicting the engine
   (A/C). Gini, redundancy, and the viability window are consistent.

**No source files were modified. This document was not committed.**
