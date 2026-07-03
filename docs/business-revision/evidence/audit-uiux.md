# OASIS In-App Dashboard — Business-Utility Audit (UI/UX)

**Job to be done:** *diagnose & benchmark org health* for a C-suite exec who must trust and act without an ecology PhD.
**Operator:** consultant / sustainability lead. **Audience:** C-suite.
**Evidence base:** 15 screenshots — 3 orgs (TechFlow = red/unsustainable, Balanced = red/unsustainable, Cone Spring = green/viable) × 5 sections (core-metrics, network-analysis, visualizations, oasis-health, detailed-report), read at full resolution plus zoomed crops of the Sustainability Assessment, Window-of-Viability, and OASIS score/dimension blocks.

Scale: 1 = fails badly · 3 = mediocre · 5 = consultant-grade. Cells ≤2 are gaps. Dimensions:
1. Decision relevance (TIEBREAKER) · 2. So-what clarity · 3. Interpretability · 4. Benchmark/context · 5. Credibility/defensibility · 6. Narrative flow · 7. Visual effectiveness.

---

## 1. Scoring table (one row per surface)

| ID | Surface | 1 DecRel | 2 So-what | 3 Interp | 4 Bench | 5 Cred | 6 Narr | 7 Visual | avg |
|----|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| D1 | Core Metrics header + KPIs (WoV robustness curve) | 4 | 3 | 3 | 4 | 4 | 3 | 4 | 3.6 |
| D2 | System Health Dashboard (Efficiency/Robustness/Viability/Roles cards) | 4 | 3 | 3 | 3 | 3 | 3 | 4 | 3.3 |
| D3 | Ulanowicz Core Metrics (Process Principles row) | 3 | 2 | 2 | 1 | 4 | 2 | 2 | 2.3 |
| D4 | Sustainability Assessment (verdict banner) | 5 | 4 | 3 | 3 | 3 | 4 | 3 | 3.6 |
| D5 | Window of Viability Bounds (A_min/A/A_opt/A_max/α) | 4 | 3 | 2 | 4 | 4 | 3 | 3 | 3.3 |
| D6 | Extended Network Metrics (Flow-based Metrics) | 2 | 2 | 2 | 1 | 3 | 2 | 2 | 2.0 |
| D7 | Balance Indicators (redundancy/flexibility/AMI) | 2 | 2 | 2 | 2 | 3 | 2 | 2 | 2.1 |
| D8 | Health Assessments (5 colored dots) | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 2.9 |
| D9 | Network Roles & Functional Specialization | 2 | 2 | 2 | 2 | 3 | 2 | 3 | 2.3 |
| D10 | Overall System Health (viz tab intro) | 3 | 2 | 3 | 2 | 3 | 3 | 3 | 2.7 |
| D11 | Network Diagram (spring + directed) | 3 | 2 | 3 | 2 | 3 | 3 | 3 | 2.7 |
| D12 | Interactive Sankey / Directed Flow diagram | 3 | 2 | 3 | 2 | 3 | 3 | 4 | 2.9 |
| D13 | Window of Viability robustness curve (viz tab) | 4 | 3 | 3 | 5 | 4 | 3 | 4 | 3.7 |
| D14 | Multi-Metric Comparison radar | not captured | | | | | | | — |
| D15 | Network Analysis (topology/centrality/community/robustness) | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 2.6 |
| D16 | System Health Dashboard / Network Health Summary radar-dots | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3.0 |
| D17 | OASIS Overall Health Assessment (score box + radar) | 4 | 3 | 3 | 3 | 2 | 3 | 3 | 3.0 |
| D18 | OASIS Dimension Status (per-dim list + radar) | 4 | 3 | 4 | 4 | 2 | 3 | 4 | 3.4 |
| D19 | OASIS Dimension Details (expanders + WoV position) | 4 | 4 | 3 | 4 | 3 | 4 | 3 | 3.6 |
| D20 | OASIS Recommendations (Critical/Medium cards) | 5 | 4 | 4 | 3 | 3 | 4 | 4 | 3.9 |
| D21 | Analysis Report tab (in-app export/preview) | 4 | 3 | 3 | 3 | 4 | 4 | 2 | 3.3 |

Notes on captures:
- **D14 (Multi-Metric Comparison radar)** — *not captured.* The visualizations screenshots end at the Window-of-Viability chart; the radar referenced at `app.py:2848` is below the fold in all three `-visualizations.png` files. Cannot score.
- D1/D13 are the same WoV robustness curve rendered in two tabs; scored separately because context differs (core-metrics header vs. dedicated viz).
- D3, D6, D7, D9 are the "Process Principles / Extended Network Metrics / Balance Indicators / Specialization" rows on the core-metrics page — all captured for all three orgs.

---

## 2. Every score ≤3 — surface, failing dimension(s), evidence, business consequence

### D17 — OASIS Overall Health Assessment · Credibility = 2  ★ HIGHEST-IMPACT GAP
- **Evidence:** `techflow-oasis-health.png` (zoomed). The Overall Score box reads **"76 /100 — ✅ HEALTHY"** in green, and the Dimension Status list shows **OPEN 100, AUTONOMOUS 100, SYMBIOTIC 100** (all green), INTELLIGENT 46, SUSTAINABLE 35 (red/CRITICAL). Meanwhile the SAME org's core-metrics verdict is **"UNSUSTAINABLE"** and the detailed-report KPI banner says **"🔴 Non-Viable"**, and this very page's Window Status = **"❌ Outside."** Balanced org is identical: **79/100 HEALTHY** with three 100s while its verdict is Non-Viable.
- **Business consequence:** A C-suite exec skims the big green "HEALTHY 76" and three perfect 100s and concludes the org is fine — the exact opposite of the tool's actual diagnosis. A consultant cannot stake their reputation on a screen that greenlights a failing system. This is a credibility-destroying self-contradiction and the single biggest reason an exec would distrust or misread the tool. (The weighted-average math that lets 3×100 outvote a critical central dimension is **for formula-validator**; the *presentation* failure — no reconciliation between "76 HEALTHY" and "Non-Viable/Outside" on the same screen — is the UI gap.)

### D6 — Extended Network Metrics (Flow-based Metrics) · avg 2.0
- **Dimensions failing:** So-what (2), Interpretability (2), Benchmark (1), Narrative (2), Visual (2).
- **Evidence:** `techflow-core-metrics.png`. Shows "Structural Info 0.31, Effective Link 0.06, Trophic Depth 1.00, Regen. Capacity 0.12" as bare numbers with only micro-captions ([H(nats)], [%Utilized]). No target band, no red/green, no "is 0.06 good?" cue. Same layout on all three orgs; the only way to tell TechFlow (bad) from Cone Spring (good) is to already know the direction of each metric.
- **Business consequence:** Raw ecological telemetry with zero interpretation. An exec cannot act on it and a consultant must hand-annotate every figure. Pure "data for data's sake."

### D7 — Balance Indicators (redundancy / flexibility / AMI) · avg 2.1
- **Dimensions failing:** So-what (2), Interpretability (2), Benchmark (2), Narrative (2), Visual (2).
- **Evidence:** `techflow-core-metrics.png` shows "0.07 α=A/C Chaotic / 0.93 flexibility / 0.07 Effect. Balance." The term **AMI** and **α** appear with no plain-language gloss. Flexibility 0.93 has no band saying whether 0.93 is dangerously high or healthy.
- **Business consequence:** These are the metrics that actually *explain* why the org is unsustainable (too much redundancy, too little organization), yet they're the least legible block on the page. The causal story is buried under jargon.

### D3 — Ulanowicz Core Metrics / Process Principles · avg 2.3
- **Dimensions failing:** So-what (2), Interp (2), Benchmark (1), Narrative (2), Visual (2).
- **Evidence:** `techflow-core-metrics.png` "Process Principles" row: Ascendency 4.29 [A(nats)], Overhead 0.45, Reserve Cap. 0.55, Efficiency 0.07, Balance 0.18 — five raw numbers, no reference band, no translation of **Ascendency** or **Overhead** into business language anywhere on screen.
- **Business consequence:** "Ascendency = 4.29" is meaningless to an exec. Without a band or plain-English label, this row consumes prime real estate while communicating nothing decision-relevant.

### D9 — Network Roles & Functional Specialization · avg 2.3
- **Dimensions failing:** DecRel (2), So-what (2), Interp (2), Benchmark (2), Narrative (2).
- **Evidence:** `techflow-core-metrics.png` "Number of Roles 1.33, Effective Nodes 9.84, Effective Roles 73.00, Connectivity 0.13" plus a "Specialization Analysis" with "Low Specialization: system lacks functional differentiation." "Effective Roles 73.00" against 10 nodes is confusing on its face and has no benchmark.
- **Business consequence:** Marginal to the diagnose-&-benchmark job; reads like network-science trivia. Competes for attention with the verdict without supporting it.

### D15 — Network Analysis section (topology / centrality / community / robustness) · avg 2.6
- **Dimensions failing:** So-what (2), Interp (2), Benchmark (2).
- **Evidence:** `techflow-network-analysis.png` / `balanced-network-analysis.png`. Dozens of graph-theory metrics (Density, Clustering, Small World, Modularity, Assortativity, Rich Club, Path Redundancy 65.00) as bare numbers. Subheader itself flags "independent of ecological theory" — i.e. disconnected from the OASIS verdict. Only the bottom "Network Health Summary" (dots + "MODERATE / GOOD") offers any so-what.
- **Business consequence:** An entire top-level section that a consultant would have to hide from a C-suite deck. It's analyst-grade exploration, not exec decision support; risks making the tool look like an academic toy.

### D10 / D11 / D12 — Visualizations (intro, network diagrams, Sankey) · avg 2.7–2.9
- **Dimensions failing:** So-what (2), Benchmark (2) for D10/D11; So-what (2), Benchmark (2) for D12.
- **Evidence:** `techflow-visualizations.png` / `viable-cone-spring-visualizations.png`. The network diagram, heatmap, and directed-flow Sankey are visually strong, but captions describe *what the chart is* ("Color-coded matrix showing flow intensity… dark = strong flow") not *what it means for the business*. No annotation of where the problem is. The green (Cone Spring) vs red (TechFlow) orgs produce structurally similar-looking diagrams; the diagram alone does not telegraph healthy-vs-sick.
- **Business consequence:** Beautiful but inert. An exec asks "so what am I looking at?" and the caption answers with cartography, not diagnosis.

### D2 — System Health Dashboard cards · So-what/Interp/Bench/Cred/Narr = 3
- **Evidence:** `techflow-core-metrics.png` four cards: Efficiency 0.07 ❌, Robustness 0.18 ❌, Viability NO ❌, Roles 1.33 ❌ (red X's, "0.06 Chaotic – Weak," "17.6% Weak"). The red X iconography *does* communicate bad — good. But "Efficiency 0.07" has no band on the card itself (the band is only on the separate WoV chart), and "Roles 1.33 ❌" is opaque. Solid but not consultant-grade.
- **Business consequence:** The color cues rescue interpretability, but the exec still can't tell *how* bad or *what good looks like* from the card alone.

### D5 — Window of Viability Bounds · Interpretability = 2
- **Evidence:** `techflow-core-metrics.png` (zoom): "Lower Bound 2.76K · Current Ascendency 909.4 · Optimal Zone 0.35–0.40 · Upper Bound 8.27K · Current α 0.07 ❌ Outside." Mixes flow-nats magnitudes (2.76K, 8.27K, 909.4) with dimensionless α (0.07, 0.35–0.40) in one row, so the reader can't see that 909.4 sits below the 2.76K lower bound without doing arithmetic. Units [flow-nats] / [dimensionless] are jargon.
- **Business consequence:** The "Outside the window" conclusion is correct and defensible, but the row makes the reader work to see it. Benchmark exists (bounds are shown) so dim 4 is fine; the failure is legibility.

### D8 — Health Assessments (5 dots) · Benchmark = 2
- **Evidence:** `techflow-core-metrics.png` five labeled dots (Sustainability red, Robustness yellow, Resilience red, Efficiency yellow, Regen. Potential yellow) with one-line captions. No numeric band behind the color; a yellow "Efficiency" gives no sense of distance-to-target.
- **Business consequence:** Traffic-light summary is directionally useful but not quantified; fine for a glance, thin for a decision.

### D16 — Network Health Summary radar/dots · all 3s
- **Evidence:** `techflow-network-analysis.png` bottom "Overall Network Health: MODERATE (0.46/1.0)" with 5 dots. Adequate but generic; "MODERATE 0.46" has a scale but no peer/threshold context.

### D17/D18 — OASIS radar · Credibility = 2 (D17), Cred = 2 (D18)
- **Evidence:** `techflow-oasis-health.png` radar shows Current Profile hugging the outer ring on OPEN/AUTONOMOUS/SYMBIOTIC (100) with a deep notch on SUSTAINABLE (35). Because three axes are pinned at 100, the radar *looks* mostly full/healthy, visually reinforcing the false "76 HEALTHY" impression rather than flagging the critical collapse. Credibility scored 2 for the same self-contradiction as D17.
- **Business consequence:** The one chart that should scream "central dimension has collapsed" instead reads as a mostly-complete shape. Reinforces the misleading headline.

### D21 — Analysis Report tab · Visual = 2
- **Evidence:** `techflow-detailed-report.png`. Under "Key Performance Indicators" the Viability Status correctly shows **"🔴 Non-Viable,"** but every KPI carries a **green ▲ up-arrow** — "▲ α=0.07," "Robustness ▲ Moderate," "Network Efficiency ▲ Sub-optimal," "Total Throughput ▲ 10 nodes." Green up-arrows are delta/trend iconography that reads as "improving / good" and directly conflicts with a Non-Viable verdict.
- **Business consequence:** Mixed signals on the export surface a consultant would actually paste into a board deck. An exec sees green arrows next to "Sub-optimal" and is confused about direction of travel.

### Cross-cutting: jargon translation & reference bands
- **Jargon (verify request):** Ascendency, AMI, overhead, α, flow-nats, "A = A/C" appear **untranslated** across D3/D5/D6/D7 and D19's "Fath et al. Principle" box. The OASIS dimension expanders (D19) are the *only* place with real plain-language translation ("under-organized and chaotic," "clearer role definitions"). Tooltips (ⓘ) exist on some labels but content wasn't verifiable from static screenshots — flag to confirm whether hover text translates the terms.
- **Reference bands (verify request):** Surfaces showing raw numbers with **NO** reference band: **D3 (Process Principles), D6 (Flow-based Metrics), D7 (Balance Indicators partial), D9 (Roles), D15 (most Network Analysis metrics).** Surfaces that DO benchmark well: **D1/D13** (WoV curve plots your org vs. viability band vs. optimum — the gold standard here), **D5** (bounds shown), **D18** (dimension list vs. 0–100 with critical/warning/healthy zones).

### Red-vs-green consistency (verify request) — mostly GOOD
- The red (TechFlow/Balanced) vs green (Cone Spring) result **is** communicated consistently on the primary verdict surfaces: D4 banner (red "UNSUSTAINABLE – Too chaotic" vs green "VIABLE – Good organization"), D2 cards (red X's vs green checks), D13 WoV curve (red dot left of band vs green dot inside band). This is the tool's strength. The consistency **breaks only at D17/D21** (OASIS "HEALTHY 76" and green up-arrows on a Non-Viable org).

### "Too chaotic" vs "over-rigid" label check (verify request) — LABEL IS CORRECT
- TechFlow α = **0.07** and Balanced α = **0.09**; both sit **below** the 0.2 lower bound, and the red dot is plotted to the **LEFT** of the green viability band on the robustness curve (`techflow-core-metrics` WoV zoom). Low α = excess overhead/redundancy relative to organization = genuinely *under-organized / chaotic*. So the verdict **"Too chaotic (α < 0.2)"** and the fix **"Increase structure and coordination"** are **correctly applied** here, not mis-labeled. (No over-rigid case appears in these three orgs to test the opposite label.)
- **One data inconsistency to flag for formula-validator (not a UI fix):** the SAME TechFlow org shows α = **0.07** on core-metrics (D5) but α = **0.066** in the OASIS SUSTAINABLE note / WoV-position plot (D19). Minor, but two different α values for one org on two screens undermines credibility. Flagged **for formula-validator / data-pipeline**, not proposed here.

---

## 3. Top 5 dashboard gaps (highest business impact first)

1. **The "HEALTHY 76 / three 100s" vs "Non-Viable" contradiction (D17/D18, echoed on D21).** A failing org is greenlit with a big green HEALTHY badge, three perfect dimension scores, and a mostly-full radar, on the same page that says Window Status = Outside. This is the highest-impact gap because it can make an exec reach the *wrong decision* and makes the whole tool indefensible. (Underlying weighting = for formula-validator; the on-screen reconciliation/labeling = UI.)

2. **No reference bands on the interpretive metrics (D3, D6, D7, D9).** The very blocks that explain *why* the org is unsustainable are shown as bare numbers with no good/bad band and heavy jargon — so the causal story ("too much redundancy, too little organization") is present in the math but invisible to the reader. Every one of these fails dimension 4.

3. **Untranslated ecology jargon on exec-facing surfaces (Ascendency, AMI, overhead, α, flow-nats).** Only the OASIS dimension expanders translate anything. A C-suite audience cannot read D3/D5/D6/D7/D15 without a glossary, violating the core "no ecology PhD" constraint.

4. **An entire analyst-grade section (D15 Network Analysis) with no so-what.** Density, assortativity, rich-club, small-world, path-redundancy 65.00 as raw numbers, explicitly labeled "independent of ecological theory." It's the section most likely to make the tool look like an academic toy in front of a board.

5. **Misleading green up-arrow iconography on the export/report KPI banner (D21).** Green ▲ next to "Sub-optimal / Non-Viable" reads as "improving/good," contradicting the red verdict on the surface a consultant would actually paste into a deck.

---

*Scope note: this audit covers presentation, information architecture, framing, and narrative only. Items touching the OASIS weighting math or the α = 0.07 vs 0.066 discrepancy are flagged **for formula-validator / data-pipeline** and no formula changes are proposed here.*
