# OASIS — Identified Formula Errors Report

**Scientific validation of the OASIS computational engine**

Date: 2026-07-03 · Branch: `feat/detailed-ecosystemic-report` · Scope: all 99 inventoried formulas in `src/`

---

## How to read this report

This report documents every computation error identified by an exhaustive validation of the OASIS
codebase, in which each of the 99 scientific/mathematical formulas was cross-checked against the
peer-reviewed papers in `_papers/` (Ulanowicz 2009; Zorach & Ulanowicz 2003; Fath 2019; Finn 1976;
Levine 1980; and canonical network-science references) or, for the proprietary OASIS composite, against
internal design logic.

Each error carries a **verification status**:

- **✅ Confirmed** — high confidence; a logic error, a documentation/consistency contradiction, or a
  data-provenance problem that does not depend on a contested scientific interpretation.
- **✅ Expert-confirmed** — a scientific/mathematical correction that a three-member adversarial expert
  panel (mathematician, ecosystem-dynamics theorist, ENA methodologist) independently re-derived and
  confirmed against the papers.
- **⚖️ Expert-reviewed — reclassified** — a claim the panel examined and **downgraded** from a "fix" to a
  design/science *decision* (it is not unambiguously demanded by the literature). Per the project rule
  *"no formula change without peer-reviewed support,"* these are **not** auto-implemented.

**Governing rule:** scientific formulas are not changed unless a peer-reviewed paper unambiguously
supports the correction.

---

## Executive summary

**The core mathematics is sound.** All 11 core Ulanowicz information-theoretic measures — Total System
Throughput, Average Mutual Information, Ascendency (A), Development Capacity (C), Reserve/Overhead (Φ),
relative ascendency (α), flow diversity, and the identity **C = A + Φ** — are **correct and
paper-faithful**. Marginal sums are not swapped, zero-flow terms are handled correctly, and the two
independent implementations (loop and vectorized) agree to machine precision. **No headline number
produced by the core engine is mathematically wrong.**

**The errors are concentrated in the layers built *on top of* that core** — the derived metrics, the
threshold constants, the proprietary 5-dimension composite, and the report/presentation layer.

**27 distinct defects** were identified:

| Severity | Count | Nature |
|----------|-------|--------|
| **Critical** | 3 | Directly corrupt a headline verdict or the sustainability score |
| **Major** | 14 | Wrong derived metric, wrong network-science formula, or self-contradicting report |
| **Minor** | 10 | Documentation, labeling, and normalization-transparency issues |

**The single most important finding** is not a formula at all — it is a **design flaw in how the five
dimensions are combined**: the overall health verdict can read "HEALTHY" while the organization is
"Non-Viable," because a flat average lets three strong dimensions mask a collapsed one. This is the root
cause of the credibility problem and is fixable with a small, explainable change to the roll-up logic.

**The most consequential scientific finding — after adversarial expert review — is not a fix, it is a
caution.** An initial pass proposed globally re-targeting the "optimal α" from 0.37 to **0.4596**,
citing Ulanowicz (2009). A three-member expert panel (mathematician, ecosystem-dynamics theorist, ENA
methodologist) **overturned that as a clean fix.** They established that (i) 0.4596 is an *empirically
calibrated* value, not a mathematical theorem (the exponent β = 1.288 is back-solved from it, so the two
are circular); (ii) the organization-facing paper, **Fath (2019), itself defines robustness as −α·log α
and maximizes it at 1/e ≈ 0.37**, so 0.37 is *not* simply wrong; and (iii) a global swap would make the
robustness curve's peak (1/e) and the "optimum" (0.4596) contradict each other. **Most importantly, the
ecologist found that these ecological viability optima are not established to transfer to *organizations*
at all** — Fath (2019) explicitly notes economic/organizational networks are more redundant, sit in a
different region of the curve, and that their calibration is an open research question. This means the
product's "every organization is unsustainable" pattern is most likely a **mis-calibrated window
artifact, not a true diagnosis** — a finding that matters more for a sellable product than any single
formula. Details in §1 (E-2) and the panel verdicts below.

---

## 1. Critical errors

### E-1 · The composite roll-up has no viability floor ✅ Confirmed
**Where:** `oasis_calculator.py:695-698, 713-718` · **Classification:** proprietary-design decision

**What happens.** The overall OASIS health score is a flat weighted average of the five dimensions
(20% each), and the overall HEALTHY/WARNING/CRITICAL band is applied to that average *independently* of
any single dimension's status. So an organization scoring `(OPEN 100, AUTONOMOUS 100, SYMBIOTIC 100,
INTELLIGENT 100, SUSTAINABLE 0)` averages to **80 → "HEALTHY"** even though its SUSTAINABLE dimension is
CRITICAL and the organization is Non-Viable. There is no floor, veto, or worst-dimension rule anywhere.

**Why it matters.** This is the direct cause of the product's most damaging contradiction — a report
that simultaneously says "Non-Viable" and "76/100 HEALTHY." It is amplified by the normalization caps
(see E-24), which pin the other three dimensions near 100 and let them outvote a collapsed SUSTAINABLE.

**Fix (design decision required).** Add a viability veto: the overall status cannot be HEALTHY if any
dimension — especially SUSTAINABLE / Window-of-Viability — is CRITICAL. This is the smallest, most
client-explainable change; it corrects the headline verdict without altering any underlying number.
Alternative roll-ups (geometric/harmonic mean, multiplicative sustainability gate) are stronger but
require re-baselining and are deferred as product choices.

### E-2 · The "optimal α" target and the organizational-calibration question ⚖️ Expert-reviewed — reclassified
**Where:** `oasis_calculator.py:623-626` (α-optimality) · **Classification:** proprietary/scientific design decision (NOT a clean paper-backed fix)

**What happens.** The α-optimality sub-score rewards proximity to **α = 0.37** (≈ 1/e), the peak of the
robustness proxy R = −α·ln α. An initial validation proposed re-targeting this to **0.4596** (Ulanowicz
2009's "window of vitality" center) as a paper-backed correction.

**Expert-panel verdict (adversarial review).** The panel **partially refuted** that proposal:

- **Mathematician:** 0.4596 = e^(−1/β) with β = 1.288, but β is itself *back-solved* from the 0.4596
  window center — the two are circular by construction. So 0.4596 is an *empirical calibration*, not a
  mathematically forced optimum. The paper hedges it as provisional.
- **Ecosystem-dynamics theorist:** the claim that theory "rejects 1/e" is **refuted** — the same
  author's *Dual Nature* paper (2009) calls **α = 1/e "the point of natural sustainability,"** and
  **Fath (2019) — the paper that extends this to economics/organizations — defines "Systemic Robustness
  = −α·log α" and maximizes it, i.e. at 1/e.** So 0.37 is a defensible target, especially for
  organizations.
- **Internal-consistency catch:** if the code keeps R = −α·ln α (which peaks at 1/e) but moves the
  *target* to 0.4596, the robustness peak and the "optimum" contradict each other — they come from two
  different kernels.

**The deeper finding (why this matters most).** The bigger issue is not 0.37-vs-0.4596 but **whether an
ecological viability window applies to organizations at all.** Fath (2019) explicitly states economic
and organizational networks are *more redundant, less efficient, and sit in a different region of the
curve* than ecosystems, and that their calibration is an **open research question**. The observed
pattern — every organization sample (α ≈ 0.07–0.10) reads "unsustainable," only a literal wetland
(α ≈ 0.58) passes — is therefore most consistent with a **mis-transferred window / scale artifact, not
genuine organizational dysfunction.**

**Recommendation.** Do **not** perform a global 0.37 → 0.4596 swap (scientifically indefensible and
self-contradicting). Instead treat the organizational optimum and window as a **calibration decision**:
keep 1/e where it legitimately normalizes the robustness proxy (the `R/(1/e)` normalization, which the
mathematician confirmed is correct), keep the ecosystem-only
β = 1.288 path where it already correctly uses 0.4596, and **re-derive or explicitly caveat the
organizational viability window** rather than inheriting the ecological one. This is the highest-value
scientific decision for making the product credible.

### E-3 · Regenerative-capacity center 0.37 + α/efficiency variable naming ⚖️ Expert-reviewed
**Where:** `ulanowicz_calculator.py:877-887` · **Classification:** tied to the E-2 calibration decision

**What happens.** Regenerative capacity is `R · (1 − |α − 0.37|)`; the α value arrives via a function
named "network efficiency" (the value *is* α = A/C, but the name misleads).

**Verdict.** The `0.37` here is subject to the **same E-2 calibration decision** — it is **not** an
independent paper-backed fix (the panel refuted the blanket 0.4596 swap). The variable-naming confusion
(efficiency vs α) is a genuine, separable clarity fix. The `R · (1 − |Δ|)` blend shape is proprietary.

---

## 2. Major errors

### Derived ecological metrics (Ulanowicz / ENA methods) ✅ Expert-confirmed

**Verified by the ENA methodologist and the mathematician** against Finn (1976), Levine (1980), and
Zorach & Ulanowicz (2003) — all confirmed, one strengthened.

| ID | Metric | Where | What's wrong | Correct form (citation) |
|----|--------|-------|--------------|--------------------------|
| **E-7** | Effective connectivity inverted | `ulanowicz_calculator.py:1084-1086` | Computes N/F (< 1) — the reciprocal of connectivity — and violates the hard "connectivity ≥ 1" floor. **Root cause (found by the panel): a dropped negative sign in the exponent.** Confirmed numerically: code = 0.31 where F/N = 3.22. | Connectivity **C = F/N** (flows per node, ≥ 1) — Zorach & Ulanowicz 2003, p.72/76 |
| **E-8** | "Finn Cycling Index" counts only short cycles | `ulanowicz_calculator.py:719-729` | Counts self-loops + 2-cycles only; **confirmed to return 0.0 for a pure 4-node ring whose true cycling → 100%**. | Relabel as short-cycle proxy; use the full Finn index (E-9) |
| **E-9** | Full Finn Cycling Index mis-normalized | `ecosystem_flow_calculator.py:140-144` | Normalizes by the scalar total instead of column throughflow, and sums the wrong Leontief entries → a **systematic underestimate** of cycling (panel measured 0.2–1.5× the canonical value across test networks — *not* the clean "2×" an earlier pass claimed, but wrong in direction and magnitude). | FCI = TSTc/TST via column-stochastic Leontief inverse — Finn 1976; Ulanowicz 2004 §5 |
| **E-10** | Trophic depth uses topological hops | `ulanowicz_calculator.py:628` | Uses unweighted average shortest-path length, ignoring flow magnitudes; cannot produce fractional effective trophic levels. | Flow-weighted effective trophic level from the structure matrix — Levine 1980; Ulanowicz 2004 §4 |
| **E-11** | "Lindeman efficiency" mislabeled | `ecosystem_flow_calculator.py:194-196` | `1 − respiration/(TST+imports)` is a system-wide retention ratio, not Lindeman between-level (≈10%) transfer efficiency. | Transfer efficiency from the Lindeman spine — Lindeman 1942; or rename the metric |

### Network-science metrics on directed flow graphs ✅ Expert-confirmed

**Verified by the mathematician** (sympy/numpy re-derivations) against canonical references. All
confirmed; the Gini coefficient was checked and found **correct** (equals the mean-absolute-difference
Gini to machine precision — no change needed).

| ID | Metric | Where | What's wrong | Correct form (citation) |
|----|--------|-------|--------------|--------------------------|
| **E-13** | Betweenness/closeness treat flow as distance | `network_analyzer.py:86,103` | `weight='weight'` makes shortest paths *minimize* flow, so strong high-flow ties are treated as long/far — inverted. **Betweenness feeds the OPEN dimension**, so this mis-scores OPEN. | Invert to cost `d = 1/flow` — Brandes 2001 |
| **E-14** | Small-world random baseline corrupted | `network_analyzer.py:230-231` | Mean degree is read from the wrong function (returns average-neighbour-degree of degree-1 nodes), corrupting the random path-length baseline → σ, ω, and the small-world verdict are unreliable. | `⟨k⟩ = 2m/n` — Fronczak et al. 2004 |
| **E-12** | Freeman centralization normalizer | `ulanowicz_calculator.py:956-963` | Uses the undirected star maximum `(n−1)(n−2)` on directed degrees; the value can exceed 1. | Directed normalizer `(n−1)²` — Freeman 1979 |
| **E-15** | Small-world ω uses wrong clustering baseline | `network_analyzer.py:244` | Uses random-graph clustering where the ω coefficient requires *lattice* clustering. | `ω = L_rand/L − C/C_lattice` — Telford / Bassett 2011 |
| **E-16** | Rich-club coefficient unnormalized | `network_analyzer.py:314-320` | `normalized=False` yields a monotone, uninterpretable curve; degree cutoff is arbitrary. | `normalized=True` (ratio to degree-preserving randomization) — Colizza 2006 |
| **E-18** | Flow-diversity utilization mixes log bases | `publication_report.py:266-267` | Divides a natural-log (nats) diversity by a base-2 (bits) denominator → understates utilization by ~31%. | Match bases (use `log(n²)` in nats) |

### Report/presentation contradictions ✅ Confirmed

Code is correct; only the rendered text/bands are wrong or inconsistent.

| ID | Where | What's wrong |
|----|-------|--------------|
| **E-19** | `publication_report.py`, `report_intelligence.py`, `main.py`, `pdf_generator.py` | The **same α** is called "Very High / good efficiency" in one section and "over-constrained / brittle, HIGH risk" in another; the breakpoints also differ across files (0.2/0.4/0.6 vs 0.2/0.35/0.45/0.6). |
| **E-20** | `publication_report.py`, `latex_report_generator.py`, `main.py` | The robustness "high" threshold is **0.20 on the PDF path but 0.25 on the LaTeX/CLI path**, so R = 0.22 flips verdict depending on which export you run. |
| **E-21** | `publication_report.py` Appendix | The methodology appendix prints **"Network Efficiency = A/(C·log₂ n)"**, but the engine computes **A/C = α**. The printed formula contradicts the number shown. (Engine is correct; fix the text.) |

### Benchmark data provenance ✅ Confirmed (needs source-tracing)

| ID | Where | What's wrong |
|----|-------|--------------|
| **E-22** | `services/published_metrics_db.py:179-186` | The stored **Florida Bay α = 0.367** cannot be sourced: the cited Heymans 2002 paper is about Everglades graminoid/cypress ecosystems (reporting ≈ 0.52 / 0.34, never 0.367), the "seagrass/marine" label mismatches, and 0.367 suspiciously equals 1/e used elsewhere. A benchmark anchor should not ship with an unverifiable value. |

---

## 3. Minor errors ✅ Confirmed

| ID | Where | What's wrong | Note |
|----|-------|--------------|------|
| **E-23** | `oasis_calculator.py:599-600 vs 633-638` | SUSTAINABLE docstring weights (0.30/0.25/0.20/0.25) differ from the executed weights (0.30/0.20/0.20/0.30). Both sum to 1.0 — scoring unaffected, but auditors reading the docstring get the wrong model. | Doc fix |
| **E-24** | `oasis_calculator.py` (per-dimension caps 0.5–0.8) | The normalization caps are **necessary size-scaling devices** (many metrics grow with network size, so caps make a 0–100 score comparable across a 5-node org and a 40-node ecosystem) — *not* arbitrary bugs. But they are undocumented and *fixed*, causing saturation that amplifies E-1. | Size-relative refinement + documentation |
| **E-25** | `oasis_calculator.py` (divisors roles/10, rolesPerNode/2, regen/0.3) | Same principle: these gauge size, but a **fixed** divisor assumes a size and mis-gauges very small/large networks. | Make size-relative (relative to n) |
| **E-26** | `ulanowicz_calculator.py:815-818` | Autocatalytic index uses a `·10` amplifier and an `n(n−1)/2` normalizer with no theoretical basis; any network with >10% cycle flow saturates. | Proprietary blend; report raw components |
| **E-27** | `precompute_pipeline.py:117-118` | Two "density" definitions coexist (`m/n²` and `m/(n(n−1))`). | Pick one denominator |

Additional minor items (Katz centrality parameter not adaptive, low simulation counts, direct-only
mutualism, base-convention labeling) are documented in the underlying validation files.

---

## 4. Two tracks for remediation

The errors split cleanly into two remediation tracks.

### Track 1 — Paper-backed / canonical corrections (expert-confirmed, safe to implement)
Each is unambiguously specified by a peer-reviewed paper or canonical reference **and confirmed by the
expert panel** — permissible under the "no formula change without peer-reviewed support" rule:

- Effective connectivity = F/N (E-7) — Zorach & Ulanowicz 2003 ✅
- Full Finn Cycling Index; relabel the short-cycle proxy (E-8/E-9) — Finn 1976; Ulanowicz 2004 ✅
- Flow-weighted trophic depth (E-10) — Levine 1980 ✅
- Lindeman efficiency (relabel or replace) (E-11) — Lindeman 1942 ✅
- Betweenness/closeness distance inversion (E-13) — Brandes 2001 ✅
- Freeman `(n−1)²`, small-world `⟨k⟩=2m/n`, ω lattice clustering, rich-club normalization, log-base
  consistency (E-12, E-14, E-15, E-16, E-18) — canonical network science ✅
- Mutualism should include *indirect* (integral-utility) effects, not direct-only — Fath 2019 Principle 8 ✅
- Report/label consistency: efficiency labels, robustness threshold, appendix formula, docstring
  weights (E-19, E-20, E-21, E-23) — ✅ code already correct, fix the text

### Track 2 — Scientific & proprietary design decisions (require a product/science call, not a literature fix)
- **The organizational α-optimum and viability window (E-2/E-3)** — the panel refuted a blanket
  0.37 → 0.4596 swap; decide the organizational calibration (keep 1/e for the robustness proxy;
  re-derive or caveat the org window). **This is now a design decision, not a Track-1 fix.**
- The roll-up viability veto (E-1) — which rule to adopt
- Size-relative redesign of the normalization caps and divisors (E-24, E-25) — the basis to use
- Whether to keep the [0.2, 0.6] α-window heuristic or move to the paper's (c, n) formulation — the
  panel found [0.2, 0.6] is **not** in the primary literature and, applied to organizations, is what
  "manufactures" the near-universal fail
- The autocatalysis blend and general magic-number tuning (E-26 and the overall/threshold bands)

---

## 5. What changes if these are fixed (regression note)

Fixes that **change computed numbers** (and therefore require re-baselining and test updates):
the SUSTAINABLE dimension (via the α-optimum), OPEN (via betweenness), AUTONOMOUS (via cycling and
connectivity), the overall verdict (via the veto), the Finn Cycling Index (≈ doubles), effective
connectivity (inverts), small-world coefficients, Freeman centralization, trophic depth, rich-club, the
flow-diversity utilization %, and the Florida Bay benchmark anchor.

Fixes that **change only rendered text** (no number moves): the efficiency labels, the robustness
threshold wording, the appendix formula, and the docstring weights.

The published-value validation suite (`services/`) and any unit tests over `oasis_calculator`,
`ulanowicz_calculator`, `vectorized_metrics`, `ecosystem_flow_calculator`, and `network_analyzer` must
be re-run, and loop-vs-vectorized parity re-confirmed after the connectivity fix.

---

## Verification status & panel outcome

The adversarial expert panel — a **mathematician**, an **ecosystem-dynamics theorist**, and an **ENA
methodologist** — has **completed** its review. Each independently re-derived results and re-read the
papers to *refute* rather than confirm. Outcome:

- **Confirmed (safe to implement, Track 1):** the ENA-method corrections (effective connectivity = F/N,
  full Finn Cycling Index, flow-weighted trophic depth, Lindeman relabel), all directed-network-science
  fixes (Freeman, betweenness inversion, small-world baseline, ω, rich-club, log base), and the mutualism
  indirect-utility extension. The Gini coefficient was checked and is **correct** (no change).
- **Confirmed (Track 1, code correct — text only):** the report/label contradictions (E-19–E-21, E-23).
- **Refuted / reclassified (now Track 2 — a decision, not a fix):** the blanket α-optimum swap to 0.4596
  (E-2/E-3). The panel showed 0.4596 is an empirical calibration (not a theorem), that 1/e is a
  defensible target for organizations (Fath 2019 uses −α·log α), and that a global swap would be
  internally self-contradicting.
- **Elevated finding:** the ecological viability window ([0.2, 0.6], optimum) is **not established to
  apply to organizations**; the "every organization is unsustainable" result is most likely a
  calibration artifact. This is the top scientific item to resolve for product credibility.

**Bottom line:** the engine's core mathematics is sound; the confirmed Track-1 corrections can proceed
under the peer-reviewed-support rule; and the two highest-value items — the roll-up viability veto (E-1)
and the organizational calibration of the viability window (E-2) — are **product/science decisions** that
should be made deliberately, not auto-fixed. The adversarial review paid off precisely by stopping a
plausible-but-wrong "correction" from being applied.

---

*Full per-formula validation detail is in `docs/business-revision/evidence/validation-*.md`, the
consolidated `validation-SYNTHESIS.md`, and the three panel reports `evidence/expert-*.md`. This report
documents identified errors only; no source code has been modified.*
