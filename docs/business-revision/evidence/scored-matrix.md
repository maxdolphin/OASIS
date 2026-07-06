# OASIS Business Revision — Reconciled Scored Matrix & Gap Heatmap

Synthesis of three independent audits into one reconciled scored matrix, gap
heatmap, and ranked gap list. Sources:

- `audit-uiux.md` — authoritative for dashboards **D1–D21** on all 7 dimensions.
- `audit-report.md` — authoritative for the PDF report **R1–R21** on all 7 dimensions.
- `audit-pm.md` — second opinion on **dim 1 (Decision relevance)** and **dim 2 (So-what)** for every surface.
- `surface-inventory.md` — canonical surface list.

**Scale:** 1 = fails badly · 3 = mediocre · 5 = consultant-grade. Cells **≤2 are gaps.**
**Dimensions:** 1 Decision relevance (TIEBREAKER) · 2 So-what clarity · 3 Interpretability · 4 Benchmark/context · 5 Credibility/defensibility · 6 Narrative flow · 7 Visual effectiveness.

**Reconciliation rule.** The domain audit sets all 7 dims for its own surfaces.
The PM audit supplies a second read on dims 1 and 2. **Where the PM and domain
scores disagree by ≥2 points on dim 1 or dim 2, the LOWER score is taken and a
footnote records both values** — conservative, because a gap flagged by either
lens is a real handoff risk. Cells inferred from prose (no explicit number in the
domain audit) are prefixed `~`. Surfaces "not captured" are marked `n/c` and
excluded from all averages.

---

## Section A — Reconciled matrix

One row per surface (D1–D21, R1–R21). "Avg" = mean of the 7 dims, 1 decimal.

| ID | Surface type | 1 DecRel | 2 So-what | 3 Interp | 4 Bench | 5 Cred | 6 Narr | 7 Visual | Avg |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| D1  | Dashboard | 4 | 3 | 3 | 4 | 4 | 3 | 4 | 3.6 |
| D2  | Dashboard | 4 | 3 | 3 | 3 | 3 | 3 | 4 | 3.3 |
| D3  | Dashboard | 3 | 2 | 2 | 1 | 4 | 2 | 2 | 2.3 |
| D4  | Dashboard | 5 | 4 | 3 | 3 | 3 | 4 | 3 | 3.6 |
| D5  | Dashboard | 2[^d5] | 3 | 2 | 4 | 4 | 3 | 3 | 3.0 |
| D6  | Dashboard | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 2.0 |
| D7  | Dashboard | 2 | 2 | 2 | 2 | 3 | 2 | 2 | 2.1 |
| D8  | Dashboard | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 2.9 |
| D9  | Dashboard | 2 | 2 | 2 | 2 | 3 | 2 | 3 | 2.3 |
| D10 | Dashboard | 3 | 2 | 3 | 2 | 3 | 3 | 3 | 2.7 |
| D11 | Dashboard | 3 | 2 | 3 | 2 | 3 | 3 | 3 | 2.7 |
| D12 | Dashboard | 3 | 2 | 3 | 2 | 3 | 3 | 4 | 2.9 |
| D13 | Dashboard | 4 | 3 | 3 | 5 | 4 | 3 | 4 | 3.7 |
| D14 | Dashboard | n/c | n/c | n/c | n/c | n/c | n/c | n/c | — |
| D15 | Dashboard | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 2.6 |
| D16 | Dashboard | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3.0 |
| D17 | Dashboard | 4 | 3 | 3 | 3 | 2 | 3 | 3 | 3.0 |
| D18 | Dashboard | 4 | 3 | 4 | 4 | 2 | 3 | 4 | 3.4 |
| D19 | Dashboard | 4 | 4 | 3 | 4 | 3 | 4 | 3 | 3.6 |
| D20 | Dashboard | 3[^d20] | 4 | 4 | 3 | 3 | 4 | 4 | 3.6 |
| D21 | Dashboard | 4 | 3 | 3 | 3 | 4 | 4 | 2 | 3.3 |
| R1  | Report | 4 | 3 | 3 | 2 | 3 | 4 | 2 | 3.0 |
| R2  | Report | 4 | 3 | 2 | 2 | 2 | 3 | 1 | 2.4 |
| R3  | Report | 2 | 1 | 2 | 1 | 2 | 2 | 1 | 1.6 |
| R4  | Report | 3 | 4 | 4 | 3 | 3 | 4 | 1 | 3.1 |
| R5  | Report | 3 | 3 | 3 | 4 | 4 | 4 | 1 | 3.1 |
| R6  | Report | 3 | 2 | 3 | 2 | 3 | 3 | 1 | 2.4 |
| R7  | Report | 4 | 3 | 2 | 2 | 3 | 3 | 2 | 2.7 |
| R8  | Report | 5 | 3 | 2 | 1 | 1 | 3 | 2 | 2.4 |
| R9  | Report | 2[^r9] | 1 | 1 | 1 | 1 | 1 | 1 | 1.1 |
| R10 | Report | 3 | 3 | 3 | 2 | 3 | 3 | 1 | 2.6 |
| R11 | Report | 5 | 3 | 3 | 2 | 2 | 3 | 2 | 2.9 |
| R12 | Report | 4 | 3 | 4 | 2 | 2 | 3 | 2 | 2.9 |
| R13 | Report | 4 | 4 | 4 | 2 | 3 | 3 | 2 | 3.1 |
| R14 | Report | 3[^r14] | 3 | 3 | 1 | 2 | 3 | 2 | 2.4 |
| R15 | Report | 5 | 4 | 4 | 3 | 3 | 4 | 2 | 3.6 |
| R16 | Report | 5 | 4 | 4 | 3 | 3 | 4 | 2 | 3.6 |
| R17 | Report | 4 | 3 | 3 | 3 | 3 | 3 | 2 | 3.0 |
| R18 | Report | 3 | 3 | 4 | 3 | 2 | 4 | 1 | 2.9 |
| R19 | Report | 4 | 4 | 4 | 3 | 3 | 4 | 1 | 3.3 |
| R20 | Report | 2 | 2 | 3 | 3 | 5 | 3 | 1 | 2.7 |
| R21 | Report | 3 | 2 | 3 | 2 | 3 | 3 | 2 | 2.6 |
| **Column avg** | (41 scored) | **3.4** | **2.8** | **2.9** | **2.5** | **2.9** | **3.1** | **2.3** | **2.85** |

**Overall product average (41 scored surfaces): 2.85 / 5** — mediocre; no surface
reaches consultant-grade (≥4.0), and only 8 of 41 clear 3.4.

**Weakest rubric dimension across the whole product: dim 7 Visual effectiveness
(column avg 2.29)**, driven almost entirely by the PDF — R1–R21 score Visual 1–2
on 18 of 21 rows because `pdfimages` confirms zero embedded images. The **next
weakest is dim 4 Benchmark/context (2.49)**, and this one is the more structural
failure: it is red or near-red across *both* surface families, not just the PDF.

**n/c note:** **D14 (Multi-Metric Comparison radar)** was *not captured* by the
uiux audit — the visualizations screenshots end at the Window-of-Viability chart
and the radar (`app.py:2848`) is below the fold in all three captures. Its cells
are `n/c` and it is excluded from every average.

**Footnotes (PM vs. domain disagreements ≥2 on dim 1 or 2 — lower taken):**

[^d5]: **D5 Decision relevance — took 2 (PM) over 4 (uiux).** uiux rated the WoV
bounds decision-relevant because the "Outside the window" verdict is defensible;
PM rated it 2 because the bounds are printed in raw throughput units (2.76K–8.27K)
that "mean nothing to an exec," so no decision changes for the target reader as
shown. Conservative: a surface an exec can't read is not decision-relevant to them.

[^d20]: **D20 Decision relevance — took 3 (PM) over 5 (uiux).** uiux rated the
Recommendations cards a 5 (Critical/Medium, action-oriented); PM rated 3 because
the actions are generic ("increase structure, standardize processes") and
metric-name-leaky, so they inform but don't yet drive a specific decision.

[^r9]: **R9 Decision relevance — took 2 (PM) over 5 (report).** The report audit
scored dim 1 a 5 (visualizations *would* be highly decision-relevant); PM scored 2
because the ReportLab path renders nothing (zero embedded images), so the surface
as delivered carries no decision. Conservative: an empty section decides nothing.

[^r14]: **R14 Decision relevance — took 3 (PM) over 5 (report).** The report audit
rated the Benchmarking section's *intent* a 5; PM rated 3 because the only
comparators are four wetlands explicitly disclaimed as "not organizational
targets," leaving the exec no peer position to act on.

---

## Section B — Gap heatmap

🟥 = cell ≤2 (gap) · 🟨 = cell = 3 · 🟩 = cell ≥4. Same rows/columns as Section A.

| ID | 1 DecRel | 2 So-what | 3 Interp | 4 Bench | 5 Cred | 6 Narr | 7 Visual |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| D1  | 🟩 | 🟨 | 🟨 | 🟩 | 🟩 | 🟨 | 🟩 |
| D2  | 🟩 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟩 |
| D3  | 🟨 | 🟥 | 🟥 | 🟥 | 🟩 | 🟥 | 🟥 |
| D4  | 🟩 | 🟩 | 🟨 | 🟨 | 🟨 | 🟩 | 🟨 |
| D5  | 🟥 | 🟨 | 🟥 | 🟩 | 🟩 | 🟨 | 🟨 |
| D6  | 🟥 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 | 🟥 |
| D7  | 🟥 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 | 🟥 |
| D8  | 🟨 | 🟨 | 🟨 | 🟥 | 🟨 | 🟨 | 🟨 |
| D9  | 🟥 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 | 🟨 |
| D10 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟨 |
| D11 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟨 |
| D12 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟩 |
| D13 | 🟩 | 🟨 | 🟨 | 🟩 | 🟩 | 🟨 | 🟩 |
| D14 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| D15 | 🟨 | 🟥 | 🟥 | 🟥 | 🟨 | 🟨 | 🟨 |
| D16 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 |
| D17 | 🟩 | 🟨 | 🟨 | 🟨 | 🟥 | 🟨 | 🟨 |
| D18 | 🟩 | 🟨 | 🟩 | 🟩 | 🟥 | 🟨 | 🟩 |
| D19 | 🟩 | 🟩 | 🟨 | 🟩 | 🟨 | 🟩 | 🟨 |
| D20 | 🟨 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟩 |
| D21 | 🟩 | 🟨 | 🟨 | 🟨 | 🟩 | 🟩 | 🟥 |
| R1  | 🟩 | 🟨 | 🟨 | 🟥 | 🟨 | 🟩 | 🟥 |
| R2  | 🟩 | 🟨 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 |
| R3  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| R4  | 🟨 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R5  | 🟨 | 🟨 | 🟨 | 🟩 | 🟩 | 🟩 | 🟥 |
| R6  | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟥 |
| R7  | 🟩 | 🟨 | 🟥 | 🟥 | 🟨 | 🟨 | 🟥 |
| R8  | 🟩 | 🟨 | 🟥 | 🟥 | 🟥 | 🟨 | 🟥 |
| R9  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| R10 | 🟨 | 🟨 | 🟨 | 🟥 | 🟨 | 🟨 | 🟥 |
| R11 | 🟩 | 🟨 | 🟨 | 🟥 | 🟥 | 🟨 | 🟥 |
| R12 | 🟩 | 🟨 | 🟩 | 🟥 | 🟥 | 🟨 | 🟥 |
| R13 | 🟩 | 🟩 | 🟩 | 🟥 | 🟨 | 🟨 | 🟥 |
| R14 | 🟨 | 🟨 | 🟨 | 🟥 | 🟥 | 🟨 | 🟥 |
| R15 | 🟩 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R16 | 🟩 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R17 | 🟩 | 🟨 | 🟨 | 🟨 | 🟨 | 🟨 | 🟥 |
| R18 | 🟨 | 🟨 | 🟩 | 🟨 | 🟥 | 🟩 | 🟥 |
| R19 | 🟩 | 🟩 | 🟩 | 🟨 | 🟨 | 🟩 | 🟥 |
| R20 | 🟥 | 🟥 | 🟨 | 🟨 | 🟩 | 🟨 | 🟥 |
| R21 | 🟨 | 🟥 | 🟨 | 🟥 | 🟨 | 🟨 | 🟥 |

### At-a-glance readout

- **Weakest dimension overall: 7 Visual effectiveness (column avg 2.29).** The PDF
  column is a near-solid 🟥 wall — 18 of 21 report rows score Visual ≤2 — because
  the ReportLab path embeds **zero images** (confirmed via `pdfimages -list`). The
  dashboards fare far better on Visual (only D3/D6/D7/D21 are 🟥). So "Visual is
  weakest" is really "the PDF has no pictures."
- **Most *structurally* pervasive gap: 4 Benchmark/context (2.49).** Unlike Visual,
  Bench is 🟥 across *both* surface families — 25 of 41 scored cells are ≤2. Raw
  numbers appear with no good/bad band on D3/D6/D7/D9/D15 and R7/R8/R10–R14, and
  the one section literally named "Benchmarking" (R14) scores Bench = 1 because its
  only comparators are wetlands. **Benchmark/context is 🟥 across nearly every
  interpretive surface** — the single clearest pattern in the heatmap.
- **Weakest surfaces overall:** **R9 (1.1)** — the empty Visualizations section;
  **R3 (1.6)** — a Table of Contents that matches no real body heading and has no
  page numbers; then **D6 (2.0)**, **D7 (2.1)**, and a tie at **2.3** among
  **D3 / D9**. Four of the five worst are the raw-ecological-telemetry blocks
  (D3/D6/D7/D9) plus the two broken PDF front-/mid-matter surfaces (R3/R9).
- **Cross-cutting patterns:**
  - **Credibility (dim 5) collapses precisely on the OASIS roll-up and the viability
    table** — 🟥 on D17, D18, R2, R8, R9, R11, R12, R14, R18 — the surfaces where the
    "76/100 HEALTHY vs Non-Viable" self-contradiction and the α-vs-bounds scale
    mismatch live. Credibility is fine where the science is quoted straight (R20 = 5,
    D1/D13 = 4).
  - **So-what (dim 2)** is a broad 🟥/🟨 band: the tool computes far more than it
    interprets. Strong so-what clusters only in the intelligence layer (R15/R16) and
    the OASIS prose (D19/R12/R13).
  - The **top-left quadrant (Decision relevance) is the healthiest column (3.4)** —
    the surfaces are *about* the right things; the product's failure is in
    explaining, benchmarking, and drawing them, not in choosing what to measure.

---

## Section C — Ranked gap list (top 10)

Ranked by **lowest score × surface prominence / decision-weight**: a board-facing
surface failing dim 1 (Decision relevance) or dim 5 (Credibility) outranks an
analyst-only surface failing dim 7 (Visual). Cross-cutting findings the three
audits converged on are folded in. Gaps whose *root* cause is math/calibration are
marked **(root: formula-validator)** — the *business framing* stays here;
presentation-only.

### 1. Self-contradicting headline: "Non-Viable / CRITICAL" vs "76–79/100 HEALTHY"
- **Surfaces / dims:** D17 (Cred 2), D18 (Cred 2), R11 (Bench 2, Cred 2), R12 (Cred 2); echoed on D21 and R21/A2.
- **Evidence:** `techflow-oasis-health.png` and `techflow-report.pdf` p.7–8 — overall "76/100 HEALTHY" (green) with OPEN/AUTONOMOUS/SYMBIOTIC pinned at 100/100, while the same org's cover verdict, D4 banner, and appendix A2 all read "Non-Viable / UNSUSTAINABLE / SUSTAINABLE 35 CRITICAL." Balanced identical at 79/100. All three audits flagged this as their #1 or #2 gap.
- **Business consequence:** An exec skims the big green "HEALTHY 76" and three perfect 100s and concludes the org is fine — the exact opposite of the diagnosis. A 30-second, deal-killing trust failure that blocks the operator→exec handoff outright. **(root: formula-validator** — the weighted roll-up that lets 3×100 outvote a CRITICAL pillar, and the 46/49-labeled-HEALTHY banding, are calibration questions; the on-screen *reconciliation* of the two verdicts is the presentation fix.**)**

### 2. Credibility keystone (org = ecosystem analogy) buried on PDF p.4, absent in-app
- **Surfaces / dims:** R4 (the only place the analogy is argued; Visual 1 but Cred/So-what carried by prose); app-wide absence.
- **Evidence:** PM Q1 — §1.1/§1.2 argue the ecosystem→org transfer once, competently, on page 4, after the cover, exec summary, and TOC; it appears **nowhere in the dashboards**. §4.2's org-level reference (α 0.30–0.45, Fath 2019) is stranded on page 14.
- **Business consequence:** The entire product's authority rests on this one leap, and a skeptical CFO's first question — "why does a swamp metric grade my company?" — has no answer they will reach. Every downstream verdict, benchmark, and recommendation inherits this unearned-authority risk.

### 3. "Benchmarking" has no organizational peer basis — only wetlands
- **Surfaces / dims:** R14 (Bench 1, Cred 2); mirrored on D5 (Bench-in-raw-units).
- **Evidence:** `techflow-report.pdf` p.10 — the sole benchmark table is four published ecosystems (Cone Spring, Cone Spring Eutrophicated, Crystal River Creek, Florida Bay 0.367), self-disclaimed as "reference points … not organizational targets." No peer cohort, no percentile, no industry basis.
- **Business consequence:** "Benchmarking" is the word that sells this to a board, and the product can't keep the promise — a "Benchmarking & Position" section that positions a software company against a tidal bay and then says don't use it as a target gives the exec nothing to act on and invites ridicule.

### 4. Zero embedded visualizations in the PDF
- **Surfaces / dims:** R9 (all seven dims 1, Avg 1.1 — the single lowest-scoring surface); drags Visual to ≤2 across R1–R21.
- **Evidence:** `pdfimages -list` returns **zero embedded images in all three PDFs** — no network diagram, no Sankey, no Window-of-Viability curve, no OASIS radar, no gauges. The section title exists in the IA; it renders nothing.
- **Business consequence:** An ecological-flow-network diagnosis whose entire thesis is a picture ("your position in a window," "the shape of your flows") is delivered as prose and number tables. Every "position in a window" claim must be taken on faith, and it is the biggest single miss versus a consultant deck.

### 5. The viability table compares two different scales (α vs. ascendency-unit bounds)
- **Surfaces / dims:** R8 (Bench 1, Cred 1 — the report's central diagnostic exhibit); recurs in R2 and R18.
- **Evidence:** `techflow-report.pdf` p.6 — "Current Position (α) = 0.066" compared against "Lower Bound = 2756.558 FAIL / Upper Bound = 8269.674 PASS," i.e. a 0–1 ratio judged against bounds in the thousands, with a "FAIL lower / PASS upper" status for a system declared *below* the window. §6 (R15) quotes the same lower bound as **0.2**.
- **Business consequence:** A client's CFO spots in five seconds that "0.066 cannot be below 2756," and the report's most important exhibit reads as a bug or sloppiness — torpedoing the viability verdict. **(root: formula-validator** — units/scale correctness and the coherence of "Lower FAIL / Upper PASS" are computation questions; the fix here is not printing two scales in one table.**)**

### 6. Near-universal "fail" verdict / binary pass-fail framing
- **Surfaces / dims:** D4, D17/R11 (verdict framing); product-wide.
- **Evidence:** PM Q2 — every sampled org is Non-Viable/outside the window (TechFlow α 0.066, "Balanced" α 0.095), and only the literal wetland (Cone Spring, α 0.577) passes. Two designed orgs — including one built to be balanced — both fail.
- **Business consequence:** A diagnostic that tells virtually every real company "you fail" is commercially dead and reads as miscalibrated. The presentation fix is to reframe the pass/fail as a *position on a gradient with a direction of travel* ("your coordination is diffuse relative to sustainable systems; move this way") and surface the calibration caveat honestly. **(root: formula-validator** — whether the α Window-of-Viability bounds, calibrated on food webs, are valid for organizational networks is a calibration question; no formula change proposed here.**)**

### 7. Raw ecological telemetry with no reference band and untranslated jargon
- **Surfaces / dims:** D6 (Avg 2.0), D7 (2.1), D3 (2.3), D9 (2.3) — all failing Bench (1–2) and So-what (1–2); R7 (Interp 2, Bench 2).
- **Evidence:** `techflow-core-metrics.png` — Ascendency 4.29, Overhead 0.45, AMI, α=A/C, "Effective Roles 73.00," Structural Info 0.31, Effective Link 0.06 as bare numbers with only unit micro-captions, no good/bad band. The blocks that actually *explain why* the org is unsustainable (too much redundancy, too little organization) are the least legible on the page. Only the OASIS dimension expanders (D19) translate anything.
- **Business consequence:** The causal story is present in the math but invisible to the reader; an exec cannot act on "Ascendency = 4.29" and a consultant must hand-annotate every figure, violating the core "no ecology PhD" constraint.

### 8. Table of Contents matches no real section; front-/mid-matter numbering leaks
- **Surfaces / dims:** R3 (Avg 1.6 — second-lowest surface; DecRel 2, So-what 1, Bench 1, Visual 1); R6 (body jumps 3.2→3.4); R18/R19 heading leaks.
- **Evidence:** `*-report.pdf` p.3 — TOC lists "3.1 Network Structure / 3.3 System Organization / …," none of which match the real body ("3.1 Core Network Metrics," "3.2 Sustainability Assessment," then a jump to "3.4"), and the TOC carries **no page numbers**. Section 9 contains sub-headers numbered "4.1/4.2/4.3"; Section 10 contains "5.1/5.2/5.3."
- **Business consequence:** A TOC that doesn't describe its own document, plus mis-numbered headings, are an immediate tell that the report was auto-assembled and unproofed — undermining trust in everything downstream before the content is even read.

### 9. Exec Summary is internally inconsistent, un-anchored, and mis-colored
- **Surfaces / dims:** R2 (Interp 2, Bench 2, Cred 2, Visual 1); D21 mirror (green up-arrows).
- **Evidence:** `techflow-report.pdf` p.2 — "Non-Viable" rendered in **green** (traffic-light failure, PM); the word split as "Non-Viabl/e"; two KPI cards print the *same* 0.066 under two different labels ("Network Efficiency" and "Rel. Ascendency α"); Balanced's summary praises "high resilience (R=0.223)" of a system it labels Non-Viable, with no reconciling sentence. In-app D21 puts green ▲ up-arrows next to "Sub-optimal / Non-Viable."
- **Business consequence:** The one page the board actually reads contradicts itself and gives no visual anchor for "how bad is bad" — credibility is won or lost here, and it currently loses it. **(Cred overclaim and the identical-0.066 labels are partly root: formula-validator** — confirm whether Network Efficiency and α are intended to be the same quantity; the green "Non-Viable" and split word are pure layout.**)**

### 10. ESG crosswalk is a superficial one-to-one code lookup
- **Surfaces / dims:** R17 (Cred 3, Visual 2).
- **Evidence:** `techflow-report.pdf` p.13 — each OASIS dimension maps to one GRI code, one ESRS code, one TCFD pillar, with no disclosure text, data-point ID, or materiality logic; some mappings stretch (SUSTAINABLE / Window-of-Viability → GRI 201-2 climate financial implications). Explicitly caveated as "indicative … not a compliance attestation."
- **Business consequence:** For a CSRD-conscious board this is box-ticking in the buyer's own language (its high sales value is why it ranks); it will not survive a sustainability lead's review and risks an ESG-washing charge. The non-attestation caveat is doing all the credibility work.

---

*Scope: presentation, framing, information architecture, and narrative only. No
formula changes are proposed. Items marked **(root: formula-validator)** carry a
math/calibration root cause handed to that agent; their business framing is
retained above. D14 excluded as `n/c` (not captured).*
