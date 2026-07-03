# OASIS Formula-Validation — Consolidated Synthesis & Paper-Backed Fix Plan

**Purpose:** Fold the six raw formula-validation reports (A, B, C&D, E&F, G, I&H) and the
99-formula inventory into a single findings document with a precise, per-defect classification and a
two-track fix plan. **This document gates a code-fix pass — no source code was modified in producing it.**

**Working dir:** `/Users/massimomistretta/Claude_Projects/Adaptive_Organization`
**Branch:** `feat/detailed-ecosystemic-report`
**Sources synthesized:**
`formula-inventory.md`, `validation-A-ulanowicz-core.md`, `validation-B-robustness-viability.md`,
`validation-CD-roles-cycling.md`, `validation-EF-network-stats.md`, `validation-G-oasis-composite.md`,
`validation-IH-refvalues-bands.md`.

**Classification legend (exactly one per defect):**
- **PAPER-BACKED FIX** — a peer-reviewed paper / canonical network-science reference unambiguously
  gives the correct form. Safe to fix under the CLAUDE.md "no formula change without peer-reviewed
  support" rule. Citation + equation quoted.
- **PROPRIETARY-DESIGN DECISION** — OASIS composite logic with no governing paper. Needs a product
  decision, not a literature fix.
- **SIZE-NORMALIZATION CONSTANT** — a constant whose *job* is to gauge/scale across network sizes.
  Per project guidance (`oasis-composite-size-normalization`): these are **not arbitrary bugs to
  strip**. Many Ulanowicz/network quantities scale with system size (TST, effective numbers, roles,
  betweenness, clustering ∝ n), so normalization is *necessary* to make a 0–100 score comparable
  across a 5-node org and a 40-node ecosystem. α = A/C and robustness R = −α·ln α are already
  size-invariant (so SUSTAINABLE is size-robust); the size sensitivity lives in OPEN / INTELLIGENT /
  SYMBIOTIC. Judgement is whether the constant gauges size *correctly*: a FIXED divisor (e.g.
  roles/10) implicitly assumes a size and mis-gauges very small/large networks → recommend
  SIZE-RELATIVE normalization (relative to n, effective nodes, or a theoretical max for that n). The
  existence of the constant is **not** flagged as an error.
- **DOC/PRESENTATION FIX** — code is correct; only documentation text or a report band/label is
  wrong or inconsistent.
- **DATA-PROVENANCE** — a stored value that cannot be sourced to its cited paper.

---

## Part 1 — Headline verdict

**The core Ulanowicz information-theoretic mathematics is CORRECT and paper-faithful.** Every one of
the 11 core measures (U1 TST, U2 AMI, U3 Ascendency A, U4 Development Capacity C, U5 Reserve/Overhead
Φ, U6 relative ascendency α, U7 Shannon flow diversity H, U8 conditional entropy, U9 structural
information, U10 Φ/C, U11 identity check) matches the Ulanowicz-2009 equation it claims
(A→Eq.12, C→Eq.11, Φ→Eq.13/14, α→A/C), with correct marginal-sum conventions (row = output = Tᵢ.,
col = input = T.ⱼ — no swap), correct `0·log0` skipping, correct TST/zero guards, and the loop and
vectorized implementations agree to machine precision. The fundamental identity C = A + Φ holds
exactly, and A = TST × AMI is confirmed. The Zorach-Ulanowicz roles family is likewise correct except
one inverted quantity, and the standard Gini/PageRank/eigenvector/assortativity/modularity metrics
are correct. **No headline number produced by the core engine is wrong.** The defects are concentrated
in the **derived / threshold / composite / presentation** layers: an incorrect α-optimality target,
one inverted effective-connectivity formula, two mislabeled cycling/trophic metrics, several
weight-as-distance and directed-vs-undirected network-stat issues, the composite roll-up's missing
viability veto, undocumented normalization caps, one unsourced benchmark value, and a set of
self-contradicting report bands and one wrong appendix formula.

**Counts.** Of the 99 inventoried quantities, the large majority are **OK / paper-faithful**
(all of A except a units-labeling note; Z1/Z2/Z4/Z5/Z6/Z8; D8/D9; the correct network metrics
N1/N2/N3/N6-directed/N7-eigenvector/N7-pagerank/N8/N12/N18; S1/S2/S3/S4/S6; the correct robustness
pieces R3/R4/R7/R9; O11 and all sub-weights summing to 1.0; every stored published value's internal
identities). **Distinct defects flagged: 27** (one row each in Part 2). By severity:
**CRITICAL = 3, MAJOR = 14, MINOR = 10.**

---

## Part 2 — Consolidated findings table

One row per distinct defect across all six reports.

| ID | Formula / quantity | file:line | Severity | What's wrong | Correct form | CLASSIFICATION | Business impact |
|----|--------------------|-----------|----------|--------------|--------------|----------------|-----------------|
| **F1** | Roll-up floor / veto (overall band) | `oasis_calculator.py:695-698, 713-718` | **CRITICAL** | Overall status bands a flat weighted mean (≥60 HEALTHY) independently of per-dimension status; four strong dims mask a collapsed one. `(100,100,100,100,0)→80→HEALTHY` while SUSTAINABLE is CRITICAL. No floor/veto anywhere. | Add a viability veto: overall cannot be HEALTHY if any dimension (esp. SUSTAINABLE / Window-of-Viability) is CRITICAL. No governing paper for *which* rule. | **PROPRIETARY-DESIGN DECISION** | Root cause of the "HEALTHY vs Non-Viable" self-contradiction. A non-viable org can be labeled HEALTHY — #1 credibility risk. |
| **F2** | α-optimality target = 0.37 | `oasis_calculator.py:623-626` (O10) | **CRITICAL** | Uses 0.37 (peak of the −α·ln α robustness *proxy*) as the *operating* optimum α. Paper explicitly rejects 1/e and fixes the propitious α at **0.4596**. | Target α_opt = **0.4596**. Ulanowicz-2009 §6: *"the geometric center of the window (c=1.25, n=3.25)… translate into α = 0.4596… most propitious value of β = 1.288"*; and *"There is no more reason to force the balance… to occur at (1/e)."* | **PAPER-BACKED FIX** | Mis-scores α-optimality → mis-scores SUSTAINABLE dimension and every regen/distance-to-optimum figure. Biggest paper-backed scientific correction. |
| **F3** | Regenerative capacity center = 0.37 + α-vs-efficiency mixup | `ulanowicz_calculator.py:877-887` (R8) | **CRITICAL** | (a) `regen = R·(1−|α−0.37|)` uses 0.37 not 0.4596. (b) `current_ratio = calculate_network_efficiency()` (=α=A/C but semantically labeled "efficiency") is fed to an α-distance term — variable-confusion risk. | (a) Use 0.4596 (Ulanowicz-2009 §6, as F2). (b) Feed the same α value with correct labeling. The R·(1−|Δ|) *blend* itself is proprietary. | **PAPER-BACKED FIX** (the 0.37→0.4596 constant) | Mis-scores regenerative capacity → feeds SUSTAINABLE; distorts the regen narrative. |
| **F4** | Distance-to-optimum uses 1/e | `report_intelligence.py:70` (R10) | **MAJOR** | `|α − 0.3679|` used as "distance to optimum." If meant as sustainability distance it must key off 0.4596; 1/e is only the peak of the −α·ln α proxy. | Distance to **0.4596** for a sustainability target (Ulanowicz-2009 §6). Keep 1/e only if explicitly relabeled "distance to R-proxy peak." | **PAPER-BACKED FIX** | Mis-states how far a system is from the sustainable optimum in the risk/narrative layer. |
| **F5** | Robustness R = −α·ln α mislabeled as Eq-17 robustness | `ulanowicz_calculator.py:548-549`; `vectorized_metrics.py:445-448,480-483` (R1) | **MAJOR** | Code computes the *shape* of Eq-15 fitness (k=1, natural log), which is a legitimate dimensionless proxy peaking at 1/e — but it is **not** the paper's Eq-17 R = T··×F (β=1.288 kernel). The math is defensible; the *label* is wrong. | Do NOT change the math. Relabel as "relative fitness / robustness proxy (−α·ln α)", distinct from Eq-17 R = T···F. Whether to adopt Eq-17 is a judgment call. | **DOC/PRESENTATION FIX** (label); adopting Eq-17 = design decision | Terminology risk in methodology text; number itself is a valid proxy. |
| **F6** | 1/e used as α-target vs as proxy ceiling | `report_intelligence.py:15`; `oasis_calculator.py:609` (R5) | **MAJOR** | 1/e = 0.3679 is correct as the max of the −α·ln α proxy (so `norm_robustness = R/(1/e)` at O11 is VALID), but wrong wherever it is used as *the optimal α operating point*. | Keep 1/e ONLY where it normalizes the R1 proxy (O11 correct). Use 0.4596 for α-target uses (Ulanowicz-2009 §6). | **PAPER-BACKED FIX** (distinction) | Same root as F2/F4 — three "optimal α" constants coexist; clarifying which each formula uses removes ambiguity. |
| **F7** | Effective connectivity inverted (reports N/F not F/N) | `ulanowicz_calculator.py:1084-1086`; `vectorized_metrics.py:388-396` (Z3) | **MAJOR** | Literal exp(½Σw·ln(Tij²/(TiTj))) yields C_code = N/F < 1 (e.g. 0.27 where F/N = 3.64) — the reciprocal of connectivity. Z7 consistency check silently substitutes F/N, masking it. | Effective connectivity **C = F/N** (Zorach-Ulanowicz 2003, identity block p.72: `C ≡ F/N`, `R ≡ F/C²`). | **PAPER-BACKED FIX** | Any reported "effective connectivity" is inverted; feeds roles/complexity narrative (AUTONOMOUS/INTELLIGENT context). |
| **F8** | Finn Cycling Index — short-cycle proxy mislabeled | `ulanowicz_calculator.py:719-729` (D1) | **MAJOR** | Counts only self-loops + 2-cycles; misses all cycles length ≥3; returns 0 for a pure 4-node ring (true cycling 100%). Docstring calls it "Finn Cycling Index." | Relabel as "short-cycle proxy"; defer to a corrected full Finn (F9). True FCI = TSTc/TST via Leontief inverse (Finn 1976; Ulanowicz 2004 §5). | **PAPER-BACKED FIX** (relabel + defer) | Underestimates cycling → mis-scores AUTONOMOUS (FCI is 0.35 of O2). |
| **F9** | Finn Cycling Index (Leontief) — non-standard normalization | `ecosystem_flow_calculator.py:140-144` (D2) | **MAJOR** | Normalizes by scalar TST not column throughflow Tⱼ (so [I−G]⁻¹≈I, cycling crushed); sums off-diagonal S not diagonal; no throughput weighting. ≈0.3–0.6× canonical FCI. | Column-normalize `g_ij = T_ij/T_j`; `S = [I−G]⁻¹`; `TSTc = Σ_i ((s_ii−1)/s_ii)·T_i`; **FCI = TSTc/TST** (Finn 1976; Ulanowicz 2004 §5). | **PAPER-BACKED FIX** | Systematic ~2× FCI underestimate → mis-scores AUTONOMOUS and ecosystem-health FCI bands. |
| **F10** | Trophic depth = unweighted shortest path | `ulanowicz_calculator.py:628` (D5) | **MAJOR** | Uses `nx.average_shortest_path_length` (topological hops), ignoring flow magnitudes; cannot reproduce fractional effective levels (paper's 2.5 example). | Effective trophic level = column-sums of `[S]=[I−G]⁻¹` (Levine 1980; Ulanowicz 2004 §4); depth = max/mean of these. | **PAPER-BACKED FIX** | Mis-states trophic structure in the ecosystem narrative. |
| **F11** | "Lindeman efficiency" mislabeled | `ecosystem_flow_calculator.py:194-196` (D7) | **MAJOR** | `1 − respiration/(TST+imports)` is a system-wide energy-retention ratio, not Lindeman between-level transfer efficiency (the ~10% rule). | True transfer efficiency from Lindeman spine `[L]` (Lindeman 1942; Ulanowicz 2004 §4); or rename to "respiratory retention ratio." | **PAPER-BACKED FIX** (relabel or replace) | Mislabeled ecosystem metric; presentation credibility. |
| **F12** | Freeman centralization denominator | `ulanowicz_calculator.py:956-963` (N4) | **MAJOR** | Denominator `(n−1)(n−2)` is the *undirected* star max applied to *directed* in/out degree; directed max of Σ(d*−dᵢ) is `(n−1)²`. Can exceed 1. | Directed normalizer `(n−1)²` (Freeman 1979; undirected `(n−1)(n−2)` only for normalized-degree undirected graphs). | **PAPER-BACKED FIX** | Over-states/under-normalizes centralization; a directed-network stat. |
| **F13** | Betweenness / closeness treat flow as distance | `network_analyzer.py:86,103` (N7) | **MAJOR** | `weight='weight'` makes shortest paths *minimize* flow → high-flow (strong) ties treated as long/far — inverted for strong-tie networks. | Invert to cost/distance `d = 1/flow` (Brandes 2001; weighted betweenness/closeness use distance). | **PAPER-BACKED FIX** | Betweenness feeds O1 → **mis-scores OPEN dimension** (avgBetweenness is 0.25 of OPEN). |
| **F14** | Small-world random baseline `<k>` corrupted | `network_analyzer.py:230-231` (N11) | **MAJOR** | `nx.average_degree_connectivity(G).get(1,2)` returns avg-neighbour-degree of degree-1 nodes, not mean degree; corrupts `Lr = ln(n)/ln<k>`, so σ (N9) and ω (N10) and `is_small_world` are unreliable. | `<k> = 2m/n` (Fronczak et al. 2004: `Lr ≈ ln(n)/ln⟨k⟩`, `⟨k⟩=2m/n`). | **PAPER-BACKED FIX** | Small-world verdict (σ, ω) unreliable — the single most impactful network-stat bug. |
| **F15** | ω second term uses random not lattice clustering | `network_analyzer.py:244` (N10) | **MAJOR** | Uses `C_random`; Telford's ω = `Lr/L − C/C_latt` needs **lattice** clustering. | Second term uses lattice clustering `C_latt` (Telford / Bassett et al. 2011). | **PAPER-BACKED FIX** | ω small-world coefficient definitionally wrong. |
| **F16** | Rich-club unnormalized + arbitrary k | `network_analyzer.py:314-320` (N13) | **MAJOR** | `normalized=False`; unnormalized φ(k) is monotone and not interpretable; `k=90th percentile` arbitrary; computed on `to_undirected()`. | `normalized=True` (ratio to degree-preserving randomization; Colizza et al. 2006). k choice = design. | **PAPER-BACKED FIX** (normalized=True) | Rich-club claim not interpretable as stated. |
| **F17** | Path redundancy — arbitrary cutoff + biased sampling | `network_analyzer.py:421-427` (N16) | **MAJOR** | `cutoff=3` arbitrary; only first `min(10,n)` nodes sampled → biased, not whole-graph. Non-standard "path redundancy." | If edge-independent paths intended, use Menger / `node_connectivity` (canonical). Otherwise proprietary — do not force to a std def. | **PROPRIETARY-DESIGN DECISION** (proxy) | Redundancy figure is a biased proxy; low downstream weight. |
| **F18** | Flow-diversity utilization % — mixed log base | `publication_report.py:266-267` (S5) | **MAJOR** | `fd/log2(n²)·100`: `fd` is in **nats** (engine ln) but denominator is **bits** (log2) → utilization understated by factor ln2≈0.693. | Match bases: `np.log(n**2)` (nats) to match `fd`, or convert fd to bits (Shannon; base consistency). | **PAPER-BACKED FIX** | Understates a reported utilization percentage. |
| **F19** | α / network-efficiency verdict bands contradict | `publication_report.py:645-651,668-680`; `report_intelligence.py:110-158`; `main.py:166-172`; `pdf_generator.py:400` (H5/H8/H9) | **MAJOR** | Same α (e.g. 0.65) is "Very High (good) efficiency" in one section, "Over-constrained/brittle (HIGH risk)" in another; breakpoints differ (0.2/0.4/0.6 vs 0.2/0.35/0.45/0.6). | Align `_categorize_efficiency` labels to the Window-of-Viability model so α>0.6 is not "Very High (good)"; unify breakpoints. | **DOC/PRESENTATION FIX** | Root cause of the "self-contradicting report" gap (upper α tail). |
| **F20** | Robustness "high" threshold 0.20 vs 0.25 | `publication_report.py:283-288`; `pdf_generator.py:398`; `latex_report_generator.py:265-266`; `main.py:174-179` (H4) | **MAJOR** | R=0.22 is "strong/High" in ReportLab/PDF path but "below high-resilience threshold" in LaTeX/CLI — cross-file disagreement on same metric. | Pick one "high" threshold across all report generators (the 0.15 lower rung is already consistent). | **DOC/PRESENTATION FIX** | Contributes to the self-contradicting-report gap. |
| **F21** | Network Efficiency appendix formula wrong | `publication_report.py:~432` (H2 / Issue 4) | **MAJOR** | Appendix prints `Network Efficiency = A/(C·log2 n)`; engine computes `network_efficiency = A/C = α` (alias `vectorized_metrics.py:508`). The `log2 n` divisor is a stray conflation with the redundancy H_max normalizer. | Fix appendix text to `Network Efficiency: α = A/C` (engine is authoritative; every other doc surface agrees). Do NOT change engine. | **DOC/PRESENTATION FIX** | Printed methodology contradicts the number shown — analyst-reproducibility / credibility gap. |
| **F22** | florida_bay α = 0.367 unsourceable | `services/published_metrics_db.py:179-186` (P4) | **MAJOR** | Cited Heymans 2002 paper is about Everglades graminoid/cypress (reports α≈0.52/0.34), not Florida Bay; stored "seagrass/marine" description mismatches; 0.367 suspiciously equals 1/e used elsewhere. | Replace with paper's actual graminoid (0.52) or cypress (0.34) figure + corrected label, OR source a genuine Florida Bay α (e.g. Ulanowicz et al. 1998, not in corpus). Human source-tracing required. | **DATA-PROVENANCE** | Benchmarking-credibility gap: a benchmark anchor with a likely-wrong value and mismatched source. |
| **F23** | O5 SUSTAINABLE docstring weights ≠ code | `oasis_calculator.py:599-600 vs 633-638` | **MINOR** | Docstring 0.30/0.25/0.20/0.25; code 0.30/**0.20**/0.20/**0.30** (both sum to 1.0). Auditor reading the docstring gets the wrong model. | Sync docstring to executed weights (or vice-versa per product intent). | **DOC/PRESENTATION FIX** | Model-transparency risk; no scoring change. |
| **F24** | Per-dimension normalization caps (0.5–0.8) | `oasis_calculator.py:99-104,341,414,492,564,641` (O6) | **MINOR** (mechanism amplifier of F1) | Caps OPEN 0.6 / AUT 0.5 / SYM 0.7 / INT 0.6 / SUS 0.8 are undocumented and cause saturation (3 dims pinned at 100), which amplifies F1's masking. Caps *per se* are necessary to gauge size — not a bug. | Keep the concept; make them **size-relative** (theoretical max of each convex combination for that n, or corpus P95) and document the basis. OPEN/INT/SYM caps carry the size sensitivity; SUS is size-invariant. | **SIZE-NORMALIZATION CONSTANT** | Saturation is the mechanism behind F1's masking; re-baselining changes OPEN/INT/SYM scores. |
| **F25** | Sub-metric divisors (roles/10, rolesPerNode/2, regen/0.3, autocat·10) | `oasis_calculator.py:538,548,619,188` (O12/D3) | **MINOR** | Fixed divisors implicitly assume a network size (10 roles as ceiling etc.); mis-gauge very small/large networks. `fitness/0.4` computed but unused. | Make SIZE-RELATIVE: normalize roles / roles-per-node relative to n or effective nodes (roles scale with size); replace fixed `·10`/`/10`/`/2` with an n-relative theoretical max. Fath gives no index, so the *blend* is proprietary. | **SIZE-NORMALIZATION CONSTANT** (divisors) / **PROPRIETARY-DESIGN DECISION** (the autocat blend + `·10`) | Mis-gauges INTELLIGENT/AUTONOMOUS across sizes; a 5-node org and a 40-node net are scored on the same fixed ceiling. |
| **F26** | Autocatalytic index magic constants | `ulanowicz_calculator.py:815-818` (D3) | **MINOR** | `0.5·count + 0.5·min(1, cycle_ratio·10)`; `expected_cycles = n(n−1)/2` and `·10` have no theoretical basis; any net >10% cycle-flow saturates. | Fath 2019 §3.8 prescribes no index. Report count + cycle_flow_ratio raw, or make the normalizer size-relative. | **PROPRIETARY-DESIGN DECISION** | Saturation distorts AUTONOMOUS autocat sub-term. |
| **F27** | Density definition inconsistency (n² vs n(n−1)) | `precompute_pipeline.py:117 vs 118`; N2′ | **MINOR** | `network_density = m/n²` coexists with connectance `m/(n(n−1))`; two "density" definitions. | Pick one denominator; if no self-loops use `n(n−1)` (directed connectance, May 1972). | **PAPER-BACKED FIX** | Minor internal inconsistency; low downstream impact. |

**Additional MINOR items noted in source reports (rolled into the above / no separate row needed):**
katz α=0.1 fixed not λmax-adaptive (N7), in/out degree CoV concatenation (N5), directed-vs-undirected
clustering inconsistency (N6), num_simulations=10 low (N14), percolation-threshold labeling (N15),
reciprocity variable naming (N17), direct-only mutualism omitting indirect utility (D6), cycle-overlap
double-count (D4), Z7 masking Z3, silent input substitutions inflating scores (O2 FCI/reciprocity, O3
modularity defaults), the ln-vs-log2 base convention across modules (A units note), and the crystal_river
assumed log-base. These are documented in the raw reports; none is CRITICAL or a headline-changer.

---

## Part 3 — The must-fix list, prioritized (CRITICAL + MAJOR, ranked by severity × business impact)

| Rank | Finding | One-line fix | Classification | Changes a HEADLINE number/verdict? | Business-revision gap it root-causes |
|------|---------|--------------|----------------|-------------------------------------|--------------------------------------|
| 1 | **F1 — roll-up floor** | Add a viability veto: overall ≠ HEALTHY if any dim (esp. SUSTAINABLE) is CRITICAL | PROPRIETARY-DESIGN DECISION | **YES — changes the overall verdict** (Non-Viable no longer prints HEALTHY) | **"HEALTHY vs Non-Viable" contradiction** — this is the direct root cause. |
| 2 | **F2 — α-optimality target 0.37→0.4596** | Set α_opt = 0.4596 in O10 α-optimality | PAPER-BACKED FIX | **YES — changes α-optimality → SUSTAINABLE score** | Mis-scored α-optimality/regen/distance. Biggest paper-backed scientific correction (paper explicitly rejects 1/e, gives 0.4596). |
| 3 | **F3 — regen center 0.37→0.4596 + α/efficiency var** | Use 0.4596 in regen; feed correctly-labeled α | PAPER-BACKED FIX (constant) | **YES — changes regenerative-capacity → SUSTAINABLE score** | Same regen/α-optimality mis-scoring as F2. |
| 4 | **F13 — betweenness as distance** | Invert weight to `1/flow` for betweenness/closeness | PAPER-BACKED FIX | **YES — changes betweenness → OPEN dimension score** | Betweenness feeds O1 → **mis-scored OPEN dimension**. |
| 5 | **F9 — full FCI (Leontief)** | Column-normalize G, diagonal-based TSTc, FCI=TSTc/TST | PAPER-BACKED FIX | **YES — changes FCI → AUTONOMOUS score & FCI bands** | **FCI underestimate → mis-scored AUTONOMOUS/roles.** |
| 6 | **F7 — effective connectivity inversion** | Set eff. connectivity = F/N | PAPER-BACKED FIX | **YES — changes reported connectivity (Z3)** | **Z3 connectivity inversion → mis-scored AUTONOMOUS/roles context.** |
| 7 | **F8 — short-cycle FCI mislabel** | Relabel D1 "short-cycle proxy"; defer to F9 | PAPER-BACKED FIX (relabel) | No (label) — but corrects a metric shown as "FCI" | Same AUTONOMOUS/cycling gap as F9. |
| 8 | **F14 — small-world `<k>=2m/n`** | Replace corrupted `<k>` with `2m/n` | PAPER-BACKED FIX | **YES — changes σ, ω, is_small_world** | Unreliable small-world verdict. |
| 9 | **F22 — florida_bay provenance** | Replace 0.367 with sourced graminoid 0.52 / cypress 0.34 (human-traced) | DATA-PROVENANCE | **YES — changes a benchmark anchor** | **Benchmarking-credibility gap.** |
| 10 | **F19 — contradictory α/efficiency bands** | Align efficiency labels to the viability model; unify breakpoints | DOC/PRESENTATION FIX | No number; **changes the verdict *text*** | **"Self-contradicting report" gap** (α upper tail). |
| 11 | **F21 — Network-Efficiency appendix** | Fix appendix text to `α = A/C` (do not touch engine) | DOC/PRESENTATION FIX | No (text only) | **"Self-contradicting report" gap** (doc vs engine). |
| 12 | **F20 — robustness band 0.20 vs 0.25** | Unify the "high robustness" threshold across generators | DOC/PRESENTATION FIX | No number; changes verdict text | Self-contradicting-report gap. |
| 13 | **F4 — distance-to-optimum 1/e→0.4596** | Key distance-to-optimum off 0.4596 | PAPER-BACKED FIX | **YES — changes reported distance** | Risk/narrative α-distance mis-stated. |
| 14 | **F10 — flow-weighted trophic depth** | Use Levine column-sums of [S] | PAPER-BACKED FIX | **YES — changes trophic depth** | Ecosystem-narrative accuracy. |
| 15 | **F11 — Lindeman efficiency relabel/replace** | Use [L]-based transfer efficiency or rename | PAPER-BACKED FIX | Depends (relabel = no; replace = yes) | Ecosystem-metric credibility. |
| 16 | **F12 — Freeman denominator `(n−1)²`** | Directed normalizer `(n−1)²` | PAPER-BACKED FIX | **YES — changes centralization** | Directed-stat accuracy. |
| 17 | **F15 — ω lattice clustering** | Second term uses `C_latt` | PAPER-BACKED FIX | **YES — changes ω** | Small-world accuracy. |
| 18 | **F16 — rich-club normalized=True** | Set `normalized=True` | PAPER-BACKED FIX | **YES — changes rich-club** | Interpretability of rich-club claim. |
| 19 | **F18 — S5 utilization log base** | Use `np.log(n²)` (nats) to match fd | PAPER-BACKED FIX | **YES — changes utilization %** | Reported percentage accuracy. |
| 20 | **F5 — robustness proxy label** | Relabel −α·ln α as fitness proxy, not Eq-17 R | DOC/PRESENTATION FIX | No | Methodology terminology. |
| 21 | **F6 — 1/e as proxy ceiling only** | Restrict 1/e to O11 normalization; α-target=0.4596 | PAPER-BACKED FIX (distinction) | No new number (clarifies F2/F4) | Removes the "three optimal-α constants" ambiguity. |
| 22 | **F17 — path redundancy proxy** | Use `node_connectivity` or flag as proprietary proxy | PROPRIETARY-DESIGN DECISION | Possibly | Low-weight redundancy figure. |

---

## Part 4 — The two tracks, explicitly separated

### Track 1 — Paper-backed / canonical corrections (SAFE to implement under CLAUDE.md)

Each is unambiguously specified by a peer-reviewed paper or canonical network-science reference. These
satisfy the "no formula change without peer-reviewed support" rule.

1. **F2 α-optimality target → 0.4596** — Ulanowicz, Goerner, Lietaer & Gomez (2009), *Ecological
   Complexity* 6:27–36, §6: *"the geometric center of the window (c=1.25, n=3.25)… translate into
   α = 0.4596… most propitious value of β = 1.288"*; and §5: *"There is no more reason to force the
   balance… to occur at (1/e)."*
2. **F3 regenerative-capacity center → 0.4596** — same citation (Ulanowicz-2009 §6). (The R·(1−|Δ|)
   *blend shape* is proprietary; only the constant is paper-backed.)
3. **F4 distance-to-optimum → 0.4596** — same citation (Ulanowicz-2009 §6) for the sustainability target.
4. **F6 1/e restricted to proxy normalization** — Ulanowicz-2009 §5 (1/e is the peak of the −α·ln α
   Eq-15 shape only, and is explicitly rejected as the operating optimum). Keep 1/e at O11
   `R/(1/e)` (that normalization is VALID); use 0.4596 elsewhere.
5. **F7 effective connectivity = F/N** — Zorach & Ulanowicz (2003), *Complexity* 8(3):68–76, identity
   block p.72: `C ≡ F/N`, `R ≡ F/C²`.
6. **F8 relabel D1 as short-cycle proxy; F9 full Finn FCI** — Finn (1976), *J. Theor. Biol.*
   56:363–380; Ulanowicz (2004), *Comp. Biol. Chem.* 28:321–339 §5: column-normalized `g_ij=T_ij/T_j`,
   `S=[I−G]⁻¹`, `TSTc = Σ_i ((s_ii−1)/s_ii)·T_i`, **FCI = TSTc/TST**.
7. **F10 flow-weighted effective trophic level / depth** — Levine (1980); Ulanowicz (2004) §4:
   effective trophic level = column-sums of `[S]=[I−G]⁻¹`.
8. **F11 Lindeman transfer efficiency (or relabel)** — Lindeman (1942), *Ecology* 23:399–418;
   Ulanowicz (2004) §4 (Lindeman spine `[L]`, ratio of successive `Σ(L_m)` rows).
9. **F12 Freeman directed normalization `(n−1)²`** — Freeman (1979); undirected `(n−1)(n−2)` is only
   for normalized-degree undirected graphs.
10. **F13 betweenness/closeness weight inversion `d=1/flow`** — Brandes (2001): weighted
    betweenness/closeness treat weight as *distance*; strong ties must be inverted.
11. **F14 small-world `<k>=2m/n`** — Fronczak et al. (2004): `Lr ≈ ln(n)/ln⟨k⟩`, `⟨k⟩ = 2m/n`.
12. **F15 ω lattice clustering in 2nd term** — Telford / Bassett et al. (2011):
    `ω = L_rand/L − C/C_latt`.
13. **F16 rich-club `normalized=True`** — Colizza et al. (2006): rich-club must be the ratio to a
    degree-preserving randomization.
14. **F18 S5 log-base consistency** — use `log(n²)` in the *same base* as `fd` (nats); Shannon base
    convention.
15. **F27 density denominator consistency** — directed connectance `L/(N(N−1))` (May 1972); pick one
    denominator (drop `m/n²` if self-loops disallowed).
16. **DOC/BAND consistency (F5, F19, F20, F21, F23)** — code is correct; fix the presentation:
    relabel the −α·ln α proxy (F5); align efficiency labels to the viability window + unify breakpoints
    (F19); unify robustness "high" threshold (F20); fix the appendix `A/(C·log2 n)` → `α = A/C` (F21,
    engine authoritative, corroborated by `docs_registry.py:432` and the report's own `:420`); sync the
    O5 docstring to executed weights (F23).

**Note on F5/adopting Eq-17:** relabeling the proxy is Track 1 (doc). *Replacing* the engine's
`−α·ln α` with the paper's Eq-17 `R = T··×F(β=1.288)` is a Track-2 design decision (it changes the
canonical robustness metric), so it is NOT auto-approved here.

### Track 2 — Proprietary design decisions (need the user's product call, NOT a literature fix)

These have **no governing paper**; a literature search cannot resolve them.

1. **F1 — the roll-up floor / veto rule.** Which policy: any-CRITICAL caps overall at WARNING vs a
   SUSTAINABLE-only viability veto (Option A), geometric/harmonic mean roll-up (Option B), or a
   multiplicative SUSTAINABLE gate (Option C). Report G recommends **Option A** now (smallest change,
   fixes "Non-Viable = HEALTHY", explainable) with Option B considered later if the business will
   re-baseline. **Product decision required.**
2. **F24 — size-relative redesign of the OPEN / INT / SYM caps (and the SUS cap).** Per the
   size-normalization reframe: do **not** delete the caps — they are necessary to compose a 0–100
   score across different network sizes. Decide the *basis*: theoretical max of each convex combination
   for that n, a corpus P95, or an n-relative gauge. OPEN/INT/SYM carry the size sensitivity (α-based
   SUS is size-invariant). **Product decision on the normalization basis + re-baseline.**
3. **F25/F26 — size-relative sub-metric divisors and the autocatalysis blend.** Whether roles/10,
   rolesPerNode/2, regen/0.3 become n-relative (roles scale with size, so a fixed ceiling mis-gauges
   small/large nets); and whether the autocat `0.5·count + 0.5·min(1, ratio·10)` blend (Fath gives no
   index) is kept, re-weighted, or replaced by raw count + cycle_flow_ratio. **Product decision.**
4. **Whether to keep the [0.2, 0.6] α-window heuristic** vs the paper's (c,n)/0.4596 formulation. The
   band is *not* verbatim in Ulanowicz-2009 (which defines the window on the (c,n) axes and gives a
   single optimal α=0.4596); [0.2,0.6] is a secondary-literature approximation. It is not contradicted
   by the paper (0.4596 sits ~65% up the band). **Product decision:** retain as documented heuristic,
   or move to the paper's (c,n)/0.4596 formulation.
5. **Magic-number tuning generally** — the O8 overall bands (60/40), O9 per-dim thresholds (15
   values), O13 recommendation triggers (50/30/40/25), and the equal 20% dimension weights. Internally
   consistent but unsourced; **product decision** on whether to empirically re-derive from a reference
   corpus.
6. **F17 path-redundancy proxy** — keep as a proprietary proxy, or switch to canonical
   `node_connectivity` (Menger). **Product decision** on which quantity is intended.
7. **Adopting Ulanowicz Eq-17 robustness** (`R = T··×F`, β=1.288) vs keeping the dimensionless
   `−α·ln α` proxy — a metric-definition choice (see F5 note). **Product decision.**

---

## Part 5 — Regression-safety note (what a fix pass must re-baseline)

**Fixes that WILL CHANGE existing outputs** (so historical/benchmark scores must be re-baselined and
tests re-run):

- **α-optimality & regenerative capacity** (F2, F3, F4, F6): every α-optimality score, regen-capacity
  value, and distance-to-optimum shifts (target 0.37 → 0.4596). This flows into the **SUSTAINABLE**
  dimension score and thus the **overall OASIS score**.
- **OASIS dimension scores**: **SUSTAINABLE** (via F2/F3), **OPEN** (via F13 betweenness feeding O1),
  **AUTONOMOUS** (via F9 FCI + F7 connectivity context feeding O2), **INTELLIGENT** (via F25 role
  divisors if made size-relative). If F24 size-relative caps are adopted (Track 2), OPEN/INT/SYM
  scores re-scale further.
- **Overall verdict** (F1 veto): systems previously labeled HEALTHY with a CRITICAL dimension will
  flip to WARNING/CRITICAL — the headline verdict changes for exactly the class the fix targets.
- **FCI** (F8/F9): finn_cycling_index roughly doubles toward canonical values; ecosystem-health FCI
  bands (H12: 0.1/0.2/0.5) re-trigger.
- **Effective connectivity** (F7): reported connectivity inverts (N/F → F/N) — order-of-magnitude change.
- **Small-world** (F14/F15): σ, ω, and `is_small_world` all change; the small-world verdict may flip.
- **Freeman centralization** (F12), **rich-club** (F16), **trophic depth** (F10), **flow-diversity
  utilization %** (F18): all change value.
- **Benchmark anchor** (F22): florida_bay α changes (0.367 → sourced value) — any benchmarking or
  published-value validation keyed to it must be updated.

**Doc/label-only fixes that do NOT change any computed number** (safe, but the *rendered verdict text*
changes): F5 (proxy label), F19 (efficiency labels/breakpoints), F20 (robustness threshold text), F21
(appendix formula text), F23 (docstring weights).

**Tests to re-run after the fix pass:**
- **Published-value validation** in `services/` — `published_metrics_db.py` +
  `scientific_validation_agent.py` (cone_spring, crystal_river, prawns_alligator identities; the
  log2↔ln conversion path P7; invariants P8). F22 (florida_bay) requires updating the stored anchor
  before this passes.
- **Any existing unit tests** covering `oasis_calculator`, `ulanowicz_calculator`, `vectorized_metrics`,
  `ecosystem_flow_calculator`, and `network_analyzer` — re-baseline expected values for every metric
  listed above.
- **Cross-check** loop vs vectorized parity remains intact after F7 (Z3) is changed in both
  `ulanowicz_calculator.py` and `vectorized_metrics.py`.

---

*Validation-only synthesis. No source code was modified. No commit was made — the controller commits
and reviews.*
