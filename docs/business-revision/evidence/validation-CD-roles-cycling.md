# Validation C & D — Roles / Effective-Complexity & Cycling / Trophic / Fath

**Scope:** formula-inventory families C (roles / effective numbers, Zorach & Ulanowicz 2003) and
D (cycling / trophic / Fath-2019 principles).
**Method:** code read + line-by-line comparison against the primary PDFs in `_papers/`, plus
numerical identity tests on random flow matrices (natural-log / `exp` transform confirmed for all
entropy quantities — `math.log` / `np.log` throughout; no base mismatch anywhere in these families).
**Mode:** validation only — no source modified.

## Primary sources

- **Zorach, A.C. & Ulanowicz, R.E. (2003)** "Quantifying the Complexity of Flow Networks:
  How many roles are there?" *Complexity* 8(3):68–76.
  `_papers/Quantifying the Complexity of Flow Networks- How many roles are there?.pdf`
- **Ulanowicz, R.E. (2004)** "Quantitative methods for ecological network analysis"
  *Computational Biology and Chemistry* 28:321–339.
  `_papers/Quntitative methods for ecological network analysis.pdf`
- **Finn, J.T. (1976)** *J. Theor. Biol.* 56:363–380 (Finn Cycling Index; cited by both papers).
- **Levine, S. (1980)** effective trophic level as column-sums of the Leontief structure matrix
  (cited in Ulanowicz 2004 §4).
- **Lindeman, R.L. (1942)** "The trophic-dynamic aspect of ecology" *Ecology* 23:399–418.
- **Fath, B.D. et al. (2019)** "Measuring regenerative economics: 10 principles…" *Global Transitions* 1:15–27.
  `_papers/Measuring regenerative economics_ 10 principles...Fath.pdf`

## Base-consistency (all of family C&D)

Flow diversity `H` and AMI are computed with **natural log** (`math.log` / `np.log`) and returned in
**nats** (`ulanowicz_calculator.py:490, 203`; docstrings say "in nats"). The effective-number
transform is `exp(·)` everywhere (`exp(H)`, `exp(AMI)`, `exp(½·Σ…)`). **This is internally
consistent** — nats ⇒ `exp`, not `2^`. Vectorized helpers (`vectorized_metrics.py`) reproduce the
loop formulas exactly (verified by inspection; identical `np.log` + `exp`). **No log-base defect in C or D.**

---

## Family C — Roles / effective numbers (Zorach & Ulanowicz 2003)

Paper equations (product form on p.72–73 decode to weighted-sum-of-logs inside `exp`):

| Paper symbol | Canonical form |
|---|---|
| F (eff. flows) | `∏ (Tij/T··)^(−Tij/T··)` = `exp(−Σ (Tij/T··)·ln(Tij/T··))` = `exp(H)` |
| N (eff. nodes) | `exp(½·Σ (Tij/T··)·ln(T··²/(Ti·T·j)))` |
| C (eff. connectivity, p.72 "e^κ/2") | `exp(½·Σ (Tij/T··)·ln(Tij²/(Ti·T·j)))` |
| R (eff. roles) | `∏ (Tij·T··/(Ti·T·j))^(Tij/T··)` = `exp(AMI)`; and `log R = AMI` |
| Identities (p.72) | `C ≡ F/N`, `R ≡ N²/F ≡ F/C² ≡ N/C` |

| ID | Code vs paper | Base | Severity | Correct form + citation | Paper-backed fix? |
|----|---------------|------|----------|-------------------------|-------------------|
| **Z1** `F=exp(H)` (`ulanowicz_calculator.py:1001-1002`) | Exact match, incl. the sign inside the exponent | nats+exp OK | **OK** | Zorach-Ulanowicz 2003, F eq. p.72 | n/a |
| **Z2** `N=exp(½·Σ w·ln(T··²/(Ti·Tj)))` (`:1041-1043`) | Exact match incl. ½ factor and `T··²` numerator | nats+exp OK | **OK** | Z-U 2003 N eq. p.72 ("note the 1/2 in the exponent") | n/a |
| **Z3** `C=exp(½·Σ w·ln(Tij²/(Ti·Tj)))` (`:1084-1086`) | Formula transcribed correctly **but the resulting quantity is the RECIPROCAL of the intended connectivity.** Numerically `ln C_code = ln N − ln F`, i.e. **C_code = N/F**, whereas the paper defines connectivity as **C = F/N** (flows per node, must be ≥ 1). For real matrices C_code < 1 (e.g. 0.27 where F/N = 3.64). | nats+exp OK | **MAJOR** | Paper: `C = F/N`; and `R = F/C²`. The reported metric should be `exp(lnF − lnN) = F/N`, not `exp(½Σw·ln(Tij²/…))`. The literal Ulanowicz-[18] "effective connectivity" formula copied here yields the inverse; the paper's own identity block (p.72) makes clear C must equal F/N. | **Yes** — set eff. connectivity = F/N per Z-U 2003 identity `C = F/N` (p.72). Do NOT change without confirming the paper convention; documented here. |
| **Z4** `R=exp(AMI)` (`:1113-1114`) | Exact match; `log R = AMI` identity confirmed | nats+exp OK | **OK** | Z-U 2003 R eq. + "taking the logarithm of R yields [AMI]" p.73 | n/a |
| **Z5** `functional_diversity=log(R)=AMI` (`:1166`) | Exact; equals AMI by construction | nats+exp OK | **OK** | Z-U 2003 p.73 | n/a |
| **Z6** `roles_per_node=R/N`, `specialization=R/n_actual` (`:1164-1165`) | Derived ratios, dimensionally fine; not a named paper quantity but logically sound | OK | **OK** | Derived from Z-U metrics | n/a |
| **Z7** consistency check (`:1145-1156`) | `verification1 = |R − N²/F|` — **true algebraic identity, passes to 1e-16.** BUT verification2/3 (`R=F/C²`, `R=N/C`) **silently substitute `derived_c = F/N`** (`:1150`) instead of the Z3 value. So the "consistency" check passes only because it discards the (inverted) Z3 output. It masks the Z3 defect rather than catching it. | OK | **MINOR** (self-consistent but misleading) | The check is mathematically valid *given* C=F/N. It confirms Z3 SHOULD be F/N — reinforcing the Z3 fix. Comment on `:1140` ("we use C = F/N for consistency") is an implicit admission that Z3's own value is not used. | Aligns with Z3 fix |
| **Z8** Effective Link Density (`:587-597`) | Proprietary: `(active_links/n²)·(AMI/ln(n²))`. Not from Z-U 2003; a custom blend of connectance × normalized AMI. Dimensionally a fraction in [0,1]. No arbitrary magic constants; logically defensible as "density weighted by organization." | OK | **OK (proprietary)** | No peer source claimed; label as proprietary. Sound but should not be presented as a Zorach-Ulanowicz metric. | No change; flag as proprietary in docs |

### Numerical identity evidence (random 3–6 node matrices, seeds 1/3/9)

- `R = N²/F` — holds to **≤ 4.4e-16** in every trial. ✅ (this is the identity Z7 verification1 uses)
- `R = F/C²`, `R = N/C`, `C = F/N` using the **Z3-coded C** — **fail by large margins** (e.g. F/C² = 98.5 vs R = 1.28). ✅ confirms Z3 is inverted.
- `ln C_code = ln N − ln F` exactly ⇒ **C_code = N/F = 1/(F/N)**. The intended `F/N` recovers all three identities to 1e-16.

**Roles-family verdict:** Z1, Z2, Z4, Z5, Z6 are **correct per Zorach-Ulanowicz** and base-consistent.
**Z3 is inverted (reports N/F instead of F/N)** — MAJOR. Z7 is a valid identity but hides Z3 by
recomputing F/N internally.

---

## Family D — Cycling / Trophic / Fath

| ID | Code vs paper | Base | Severity | Correct form + citation | Paper-backed fix? |
|----|---------------|------|----------|-------------------------|-------------------|
| **D1** FCI approx: self-loops + ½·Σ min(Tij,Tji) over TST (`ulanowicz_calculator.py:719-729`) | Counts ONLY self-loops and 2-cycles; **misses every cycle of length ≥ 3.** Numerically underestimates true FCI by ~2× on dense nets and **returns 0 for a pure 4-node ring whose true cycling = 100%.** Docstring calls it "Finn Cycling Index" — it is not. | OK (ratio) | **MAJOR** | True FCI = TSTc/TST via Leontief inverse (Finn 1976; Ulanowicz 2004 §5). D1 is a lower-bound proxy valid only when cycling is dominated by self/2-cycles. | Not a fix to D1 itself; **relabel as "short-cycle proxy"** and defer to D2 (once D2 is corrected). |
| **D2** FCI full (Leontief) `fci=(Σleontief − n)/Σleontief` (`ecosystem_flow_calculator.py:140-144`) | **Does NOT implement the standard Finn method** despite docstring. Two defects: (1) `flow_norm = T/tst` normalizes by the scalar total throughput, not by column throughflow `T_j` — the correct `G` is column-stochastic `g_ij = T_ij/T_j`. With `T/tst` the entries are tiny so `[I−G]⁻¹ ≈ I` and cycling is crushed. (2) `Σleontief − n` sums **all off-diagonal** S entries (through-flow along all paths), but Finn cycling uses only the **diagonal** via `(s_ii−1)/s_ii`. (3) No throughput weighting of compartments. Result ≈ 0.3–0.6× the canonical FCI in every test. | OK (ratio) | **MAJOR** | Canonical: build column-normalized `G` (`g_ij=T_ij/T_j`), `S=[I−G]⁻¹`, `TSTc = Σ_i ((s_ii−1)/s_ii)·T_i`, **FCI = TSTc/TST** (Finn 1976; Ulanowicz 2004 §5, "each diagonal element multiplied by throughput of that taxon, summed"). | **Yes** — replace with the column-normalized Leontief + diagonal-based TSTc form. Fully paper-backed. |
| **D3** Autocatalytic Index `0.5·count_factor + 0.5·min(1, cycle_flow_ratio·10)` (`ulanowicz_calculator.py:815-818`) | **Concept faithful** to Fath 2019 Principle 9 ("number of autocatalytic cycles… length > 1"): counts simple cycles (len ≤ 6) via Johnson's algorithm. **But the composite has arbitrary magic numbers:** `expected_cycles = n(n−1)/2` normalizer has no theoretical basis, and the **`·10`** multiplier makes any net with >10% cycle-flow saturate to 1.0. | OK | **MINOR** | Fath 2019 §3.8 only asserts "number of autocatalytic cycles is one indicator" — it prescribes no index. The count and cycle_flow_ratio are legitimate; the **`·10` and `n(n−1)/2` weights are unjustified.** | No paper-backed fix (Fath gives no formula). Flag magic `·10`; document blend as proprietary; consider reporting count + cycle_flow_ratio raw. |
| **D4** cycle_flow_ratio = cycle_flow/TST (`ulanowicz_calculator.py:811`) | `cycle_flow` = Σ over detected simple cycles of the **min edge flow** in each cycle. Reasonable "bottleneck" measure of flow committed to cycling, but **double-counts** flow shared by overlapping cycles (a link in k cycles contributes k times) so ratio can exceed the true cycled fraction. | OK (ratio) | **MINOR** | No single canonical definition; true cycled flow is TSTc (see D2). D4 is a heuristic. | Flag as heuristic; the rigorous cycled-flow fraction is FCI (D2 corrected). |
| **D5** Trophic depth = `nx.average_shortest_path_length` (unweighted) (`ulanowicz_calculator.py:628`) | Uses **unweighted topological hops**, ignoring flow magnitudes. The canonical effective trophic level is **flow-weighted** (column-sums of Leontief `[S]`, Levine 1980). Paper's own worked example: comp. 4 gets 60%/30%/10% at levels 2/3/4 ⇒ effective level **2.5** — a shortest-path metric returns the min chain length and cannot reproduce this. | OK (levels) | **MAJOR** | Effective trophic level = column-sums of `[S]=[I−G]⁻¹` (Levine 1980; Ulanowicz 2004 §4); trophic depth = max/mean of these effective levels. | **Yes** — flow-weighted Levine/Lindeman-spine approach is paper-backed. Current shortest-path is a weak topological proxy; relabel or replace. |
| **D6** Mutualism ratio & weighted (`oasis_calculator.py:229,247`) | **Direct** bidirectional flow only: `mutual_pairs` (both directions >0) / connected pairs; `weighted = Σ min(Tij,Tji)/Σ max(Tij,Tji)`. Fath [44] defines mutualism over the **direct + indirect** integral utility matrix (`U=(I−D)⁻¹`) and its sign structure (+/+ mutualist, etc.). Code omits indirect effects. | OK (ratio) | **MINOR** | Fath 2019 §3.7 / Fath [44] Network-mutualism: sign of integral utility matrix incl. indirect relations. Concept (reciprocity ⇒ mutual benefit) is faithful; operationalization is a **direct-only** simplification. No magic numbers. | Optional upgrade to integral utility (paper-backed) if indirect mutualism is required; direct proxy is defensible for a first-order measure. |
| **D7** Lindeman efficiency = `1 − respiration/(TST + imports)` (`ecosystem_flow_calculator.py:194-196`) | This is a **system-wide energy-retention ratio**, NOT Lindeman between-trophic-level transfer efficiency. Lindeman (1942) efficiency = productivity passed level_n→level_{n+1} (`λ_{n+1}/λ_n`, the "~10% rule"), obtained from the Lindeman spine `[L]`. The coded quantity is dimensionally sound and bounded [0,1] but is a **dissipation metric mislabeled as Lindeman efficiency.** | OK (ratio) | **MAJOR (mislabel)** | Lindeman 1942; Ulanowicz 2004 §4 (Lindeman transformation matrix `[L]`, ratio of successive `Σ(L_m)` rows). | **Yes** — true transfer efficiency from `[L]`. If keeping the current metric, rename to "respiratory retention ratio"; do not call it Lindeman efficiency. |
| **D8** Extended TST = internal + imports + exports + respiration (`ecosystem_flow_calculator.py:100`) | Standard boundary-inclusive TST. Correct. | OK (flow units) | **OK** | Ulanowicz 2004 §2 (TST incl. boundary exchanges) | n/a |
| **D9** import/export/respiration ratios over TST_extended (`ecosystem_flow_calculator.py:217-219`) | Simple bounded fractions of extended TST; dimensionally consistent, no formula issues. | OK (ratio) | **OK** | Standard bookkeeping ratios | n/a |

### Numerical evidence (D1/D2)

- **D2 vs canonical Finn** (5 random nets w/ boundary flows): code D2 = 0.10–0.20 where canonical FCI = 0.17–0.47 → **ratio 0.31–0.62** (systematic ~2× underestimate).
- **D1 pure 4-node ring** (A→B→C→D→A, 100% cycling): **D1 = 0.0** (misses the length-4 cycle). On dense random nets canonical ≈ 1.0 while D1 ≈ 0.32–0.50.

---

## Severity roll-up

| Severity | IDs |
|----------|-----|
| **CRITICAL** | — (none) |
| **MAJOR** | **Z3** (effective connectivity inverted: reports N/F not F/N); **D1** (FCI short-cycle proxy → 0 on pure rings, mislabeled as Finn); **D2** (non-standard Leontief normalization + off-diagonal cycling → ~2× underestimate, mislabeled as standard Finn); **D5** (unweighted shortest-path ≠ flow-weighted effective trophic level); **D7** (respiration ratio mislabeled as Lindeman efficiency) |
| **MINOR** | **Z7** (valid identity but masks Z3), **D3** (magic `·10` + ad-hoc `n(n−1)/2` normalizer), **D4** (cycle-overlap double-count heuristic), **D6** (direct-only mutualism, omits indirect) |
| **OK** | Z1, Z2, Z4, Z5, Z6, Z8(proprietary), D8, D9; base/log consistency across all |

**Papers referenced for fixes:** Zorach & Ulanowicz 2003 (Z3); Finn 1976 + Ulanowicz 2004 §5 (D1, D2);
Levine 1980 + Ulanowicz 2004 §4 (D5); Lindeman 1942 + Ulanowicz 2004 §4 (D7); Fath 2019 §§3.7–3.8 (D3, D6).
No source was modified. Per CLAUDE.md, all proposed corrections are cited to peer-reviewed sources above;
none should be applied without confirming the paper convention documented here.
