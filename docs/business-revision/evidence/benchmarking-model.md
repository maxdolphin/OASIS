# OASIS Benchmarking-Basis Model (Business Revision)

**Purpose.** Resolve the audit finding that OASIS's "Benchmarking" section has **no organizational peer basis** — it currently positions a company against four published wetlands (Cone Spring, Cone Spring Eutrophicated, Crystal River Creek, Florida Bay) and nothing else. This document recommends a **credible, defensible, layered benchmarking model** and specifies exactly how each headline metric is contextualized on-screen and in the report.

**Scope.** Presentation / framing / information-architecture only. No formula changes. Calibration and validity questions are flagged **(formula-validator)** and handed off, never resolved here.

**Audit anchors.** `audit-pm.md` Q2 (every sampled org reads "unsustainable" / near-universal fail) and Q3 (benchmark basis = theoretical band + swamps, no peer set). `scored-matrix.md` gap #3 (no peer basis, R14 Bench = 1) and gap #6 (near-universal fail / binary framing). Bench/context (dim 4) is the most *structurally* pervasive gap in the product: 🟥 across **both** the dashboard and PDF families (column avg 2.49; 25 of 41 scored cells ≤ 2).

---

## 0. Ground truth — values verified in the code

Every number below was read from source, not the narrative. **Read these before trusting any framing built on top of them.**

| Quantity | Implemented value | Where | Matches the narrative? |
|----------|-------------------|-------|------------------------|
| Window-of-Viability lower bound (α) | **0.2** | `src/report_intelligence.py:13` (`VIABILITY_LOWER = 0.2`); `src/ulanowicz_calculator.py:379` (`lower_bound = 0.2 * development_capacity`) | **Yes** — 0.2 as stated. |
| Window-of-Viability upper bound (α) | **0.6** | `src/report_intelligence.py:14` (`VIABILITY_UPPER = 0.6`); `src/ulanowicz_calculator.py:380` (`upper_bound = 0.6 * development_capacity`) | **Yes** — 0.6 as stated. The docstring at `ulanowicz_calculator.py:314` also says "Optimal range: 0.2 - 0.6". |
| Robustness optimum (α) | **1/e ≈ 0.367879441** | `src/report_intelligence.py:15` (`ROBUSTNESS_OPTIMUM = 0.367879441  # 1/e`) | **Mostly** — the report-intelligence layer uses the exact 1/e, i.e. **0.368**, not the rounded 0.37. |
| Robustness optimum (α) — second constant | **0.37** (rounded) | `src/ulanowicz_calculator.py:314, 522, 880` (`optimal_ratio = 0.37`; "Peak robustness: ~0.37") | **Finding:** the codebase carries **two** optimum constants — `0.367879441` (1/e, used by the benchmark/report layer) and a rounded `0.37` (used inside `calculate_regenerative_capacity`). Same theoretical point (α = 1/e maximizes R = −α·ln α); different rounding. Present the optimum as **α ≈ 0.37 (= 1/e)** on-screen so both agree. This ~0.001 discrepancy is a code-hygiene note, not a benchmarking blocker. |
| Robustness formula | R = −α·ln(α) | `src/ulanowicz_calculator.py:549` | R peaks at α = 1/e ≈ 0.368, R_max ≈ 0.368 (`:526` "max ~0.368"). |
| Fitness / window-of-**vitality** center | α = e^(−1/β) ≈ **0.4596** for β = 1.288 | `src/ulanowicz_calculator.py:522, 836`; `src/oasis_calculator.py:266` | A *different* β-tuned optimum (Ulanowicz window of vitality). Not the same as the 0.37 robustness optimum — do not conflate the two in exec copy. |

**Finding on the α reference band (important).** The **engine viability band is 0.2–0.6** (verified above), and the exec narrative should quote **0.2–0.6** to stay faithful to what the tool actually computes. But the report prose cites a *narrower food-web band* — `src/latex_report_generator.py:274`: "Ecological food webs: α ∈ [0.20, 0.50] (Ulanowicz, 2009)." So the codebase already ships **two slightly different α reference ranges** (engine 0.2–0.6 vs. cited literature 0.20–0.50). This is a consistency defect to reconcile in copy; it does not change the recommendation. **The band the tool enforces is 0.2–0.6.**

**Fath 2019 org-level α reference — CONFIRMED REAL in the codebase.** The claim "high-performing organizations show α ≈ 0.30–0.45 (Fath et al., 2019)" is not a report-only footnote; it is **wired into three live surfaces**:

- `src/latex_report_generator.py:275` — verbatim: `High-performing organizations: $\alpha \in [0.30, 0.45]$ (Fath et al., 2019)`, with line 276 classifying the current system as "aligns with" vs. "deviates from" that band.
- `src/pdf_generator.py:408` — the **Executive-Summary KPI card** for Rel. Ascendency labels α **`'Optimal' if 0.30 <= alpha <= 0.45 else 'Warning'`**.
- `src/pdf_generator.py:750` — the **Core Metrics table** grades α **`'Optimal' if 0.30 <= alpha <= 0.45`**.

So the org-level band **already drives the on-screen "Optimal/Warning" verdict** — it is simply never surfaced as a *named comparator* in §5 Benchmarking, which instead shows only the four wetlands. **The primary anchor the audit asks for already exists in code; it just needs to be promoted into the benchmark section and the wetlands demoted.**

**Tier-2 dataset inventory.** `data/ecosystem_samples/*.json` = **22 files**. They span wetlands/food webs (`cone_spring_original`, `cone_spring_eutrophicated`, `crystal_river_creek`, `florida_bay`, `cypress_wetland`, `graminoid_everglades`, `mondego_estuary`, `chesapeake_bay_simplified`, `baltic_sea`, `prawns_alligator_*`) **and** non-ecological networks (`us_airport_network`, `bitcoin_transaction_network`, `dblp_coauthorship_network`, `manufacturing_network`, `pharma_development_network`, `enzyme_network`, `protein_structure_network`, `molecular_compound_network`, `mutag_supply_chain_network`). The published α values are looked up at runtime via `services/published_metrics_db` (`report_intelligence.py:79`, `get_published_metric(net_id, 'relative_ascendency')`). The four wetlands currently shown in §5 are a *subset* of these 22; the non-ecological networks are available but unused as anchors.

---

## Part 1 — The layered benchmarking model (recommendation)

Three tiers, shipped in sequence. Each tier is honest about what it can and cannot claim.

### Tier 1 — Theoretical norms (SHIP NOW)

**Basis:** the Ulanowicz Window-of-Viability band **α ∈ [0.2, 0.6]** and the robustness optimum **α ≈ 0.37 (= 1/e)**, exactly as implemented (`report_intelligence.py:13–15`). Every headline metric is framed against its own theoretical band or optimum (see Part 2 table).

- **Strength:** zero data cost, fully self-contained, mathematically defensible from first principles (the robustness curve R = −α·ln α has a single analytic maximum at 1/e). Nothing to license, seed, or anonymize. Defensible to a skeptic *as theory*.
- **Limit:** it answers **"viable vs. not,"** never **"better vs. peer."** It cannot tell an exec whether α = 0.34 is top-quartile or bottom-quartile among comparable companies — only that it sits inside the theoretical band and close to the robustness optimum.
- **Framing rule:** call this "**position relative to the theoretical viability range**," never "benchmarking." (See Tier 3.)

### Tier 2 — Reference anchors (NEAR-TERM)

**Basis:** the shipped datasets as illustrative **"you-are-here" anchors on the α scale**, clearly labeled cross-domain/illustrative — *not* organizational targets.

**CRITICAL correction to the current product** (directly addresses audit Q3 / gap #3):

1. **Promote the org-level α reference to the PRIMARY anchor.** Put **"High-performing organizations: α ≈ 0.30–0.45 (Fath et al., 2019)"** at the *top* of the benchmark exhibit as the headline comparator. It already exists in code (`latex_report_generator.py:275`, `pdf_generator.py:408/750`) and already drives the Optimal/Warning verdict — it must therefore be the named reference an exec sees. This is the *only* organizational (rather than ecological) comparator in the product; it is the board-credible one.
2. **Demote the wetlands to a methodology footnote.** Cone Spring / Crystal River / Florida Bay stop being the headline table (`pdf_generator.py:1022–1045`, §5) and become a small "how the viability scale was validated in ecology" note. Comparing a software company to a tidal bay in the *headline* invites the exact ridicule the audit names ("compared to a swamp"). They stay in the product as scale-validation provenance, not as the exec's comparator.
3. **Optionally add cross-domain human-system anchors** from the 22-file set that are *not* wetlands (e.g. `us_airport_network`, `manufacturing_network`, `pharma_development_network`, `dblp_coauthorship_network`) as "same math, other domains" illustration — still labeled illustrative, still not targets — because an airport or supply-chain network is a more intuitive analog to an org than a marsh.

- **Strength:** gives the "you are here on the α line" picture real reference points, led by an *organizational* one.
- **Limit:** still not a same-sector, same-size peer set. Every anchor must carry the label **"illustrative reference point — not an organizational target."**

### Tier 3 — Peer cohort (DEFERRED, flagged)

**State plainly: this does not exist yet, and until it does, the exec framing must not say "benchmarking."**

- **What it would require:** an **anonymized cohort of real organizations run through the identical OASIS pipeline** (same ingestion, same Ulanowicz/OASIS calculators), tagged by size band and sector, so a new org can be placed at a **percentile** within its cohort. Percentiles are only honest above a minimum cohort size — recommend **N ≥ 30 per (sector × size) cell** before quoting quartiles/percentiles, and **N ≥ 8–10** before quoting even a coarse "below / around / above the cohort median" band; below that, show the cohort as individually plotted anonymized points, not a distribution.
- **Why fake peer benchmarks are rejected:** a fabricated or synthetically-generated "peer average" would manufacture authority the tool has not earned — precisely the unearned-authority failure the audit flags as the product's #1 risk. A board that discovers the "peer benchmark" was invented discards the entire diagnosis. Better to ship an honest "no peer basis yet" than a fake one.
- **Interim exec framing (until Tier 3 exists):** the section is titled and spoken as **"Position relative to the theoretical viability range,"** *not* "Benchmarking." The word "benchmark" is reserved for when a real cohort with percentiles ships. This is the single most important framing change: it converts an unkeepable promise into an honest, defensible statement.

---

## Part 2 — Per-metric contextualization table

Reference bands are the ones **implemented in code** (cited inline). On-screen labels and "so-what" sentences are the recommended presentation. "α" = relative ascendency = A/C.

| Metric | Reference band (from code) | On-screen label | "So-what" sentence |
|--------|----------------------------|-----------------|--------------------|
| **Relative Ascendency (α = A/C)** | Viability band **0.2–0.6** (`report_intelligence.py:13–14`); robustness optimum **≈0.37 = 1/e** (`:15`); **org anchor 0.30–0.45, Fath 2019** (`pdf_generator.py:408/750`, `latex_report_generator.py:275`) | "Coordination balance — α = {value} (viability 0.2–0.6; high-performing orgs 0.30–0.45; sweet spot ≈0.37)" | How much of your capacity is locked into fixed structure vs. kept as flexible reserve; too low = diffuse and chaotic, too high = rigid and brittle, and healthy organizations cluster around 0.30–0.45. **Tier-1 honesty caveat: the 0.2–0.6 band is calibrated on ecological food webs; whether those exact bounds are valid for organizational flow networks is an open calibration question (formula-validator) — every sampled org lands below 0.2, which may be a calibration artifact, not a universal failure.** |
| **Robustness (R)** | Peaks at **α = 1/e ≈ 0.368**, **R_max ≈ 0.368** (`ulanowicz_calculator.py:526, 549`); report classes R > 0.2 High / 0.15–0.2 Moderate / <0.15 Low (`pdf_generator.py:398`); calculator: >0.3 HIGH, <0.1 LOW (`:1234–1237`) | "Resilience — R = {value} of a theoretical max ≈ 0.37 ({High/Moderate/Low})" | Your system's capacity to absorb shocks without collapsing; it is highest when order and flexibility are balanced (α ≈ 0.37), so R is read *together with* α, not alone. **(Note: two different R-band thresholds exist in code — reconcile to one on-screen band; presentation issue, not formula.)** |
| **Total System Throughput (TST)** | **No theoretical band** (scale quantity, units = flow) | "Total activity — {value} units (scale indicator, no good/bad band)" | The gross volume of flow through the network — a size/activity measure, not a health verdict; it contextualizes the other metrics (all ratios are relative to this) but is never itself "pass/fail." |
| **AMI (Average Mutual Information)** | **No standalone band**; interpreted only via α = A/C where A = TST·AMI | "Flow organization — {value} bits (feeds α; not judged alone)" | How constrained/organized the flow pattern is; higher AMI means more structured routing, but it is only meaningful relative to capacity — which is exactly what α captures, so judge α, not AMI in isolation. |
| **Ascendency (A)** | **No standalone band**; A = TST·AMI, judged only as the ratio A/C = α | "Organized activity — {value} (numerator of α; judge as α)" | The portion of total activity that is organized/directed; on its own it is a raw magnitude — its health meaning comes entirely from A/C = α against the 0.2–0.6 band. **(Audit gap #5: never print A on a 0–1 α scale beside bounds in raw ascendency units — that scale-mismatch is the report's central defect; formula-validator owns the units, presentation owns not mixing scales.)** |
| **Development Capacity (C)** | **No theoretical band**; C = A + Φ (`ulanowicz_calculator.py:325–351`, C = A + reserve) | "Total capacity — {value} (the 100% that α is a fraction of)" | The system's total organizational potential (organized + reserve); it is the denominator of α, so it defines the ceiling — a metric to *contextualize* α, never to pass/fail on its own. |
| **OASIS Overall Score** | **0–100 composite**; banded by OASIS status (HEALTHY etc.), weighted across 5 dimensions | "Overall health — {score}/100 ({status})" | A single roll-up of the five OASIS dimensions; **must be reconciled on-screen with the viability verdict** — today an org can read "76/100 HEALTHY" while simultaneously "Non-Viable," which is a 30-second trust-killer (audit gap #1). Present as one headline with viability as a named sub-component, not a co-equal second headline. **(The weighting that lets three 100s outvote a CRITICAL pillar is formula-validator; the on-screen reconciliation is presentation.)** |
| **SUSTAINABLE dimension score** | **0–100**; formula `SUS = 0.30·R_norm + 0.20·W + 0.20·RC_norm + 0.30·α_opt` (`oasis_calculator.py:599`, `docs_registry.py:703`) | "Sustainability pillar — {score}/100 (built from robustness + viability + α-optimality)" | The dimension that carries the viability verdict into the OASIS roll-up; because it is 60% driven by robustness and α-optimality, a low α (below the 0.2–0.6 band) pulls it down hard — this is the pillar that should *lead* the reconciled headline, not the overall average that masks it. |

---

## Part 3 — The "gradient, not pass/fail" reframe

**Problem (audit Q2 / gap #6):** the tool currently renders viability as a binary verdict, and because the α band (0.2–0.6) is food-web-calibrated, essentially every real organization lands *below* it and reads "Non-Viable / FAIL." A diagnostic that tells almost every company "you fail" is commercially dead and reads as miscalibrated — the more so because a literal wetland (Cone Spring, α 0.577) is the only "pass."

**Reframe: present position as a direction-of-travel on a gradient, not a binary.**

- **Show the α line, mark the org's dot, name which way to move.** Instead of "α = 0.066 → Non-Viable (FAIL)," render the α axis with three zones — **← diffuse/chaotic (α < 0.2) · viable (0.2–0.6, sweet spot ≈0.37) · rigid/brittle (α > 0.6) →** — plot the organization's dot, and state the *vector*: e.g. **"Your α is left of the viability band — coordination is diffuse. Direction of travel: add structure (clearer roles, fewer redundant flows) to move toward the band."** For a high-α org the mirror applies: "right of the band — over-organized/brittle; introduce redundancy and slack."
- **Replace FAIL/PASS words with position + move.** "Below the band, tending chaotic — move toward more structure" reads as *guidance*; "FAIL" reads as a *verdict*. Same underlying number, opposite reception. The `build_benchmark_view` output already computes `position` ∈ {below, within, above} and `distance_to_optimum` (`report_intelligence.py:53–70`) — the data for a gradient exists; only the *rendering* is binary.
- **Anchor the destination on the org comparator, not the wetland.** "Move toward 0.30–0.45, where high-performing organizations cluster (Fath 2019)" is a credible target; "move toward Florida Bay's 0.367" is not. Use the Tier-2 primary anchor as the arrow's destination.
- **Carry the calibration caveat as an honesty line, not a formula edit.** One sentence: *"Viability bounds are calibrated on ecological networks; treat your position as a direction of travel rather than an absolute grade (calibration for organizational networks is under review)."* This defuses the "the swamp passed and I failed" objection without touching the math. **(formula-validator owns whether the bounds should be re-calibrated; presentation owns reading them as a gradient.)**

Net effect: the benchmarking section reads as **"here is where you sit and which way to move,"** not **"you fail" —** turning a near-guaranteed death sentence into actionable guidance while staying fully honest about the theoretical (not peer) basis.

---

*Scope: presentation, framing, information architecture only. No formula changes proposed. Items marked (formula-validator) carry a calibration/validity root cause handed to that agent; their business framing is retained here. All cited values verified against source on the working branch.*
