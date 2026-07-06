# OASIS Codebase — Exhaustive Formula Inventory

**Purpose:** Gate a rigorous scientific-validation pass. Every computed scientific/mathematical
quantity, threshold constant, verdict band, weighting/aggregation, and statistical formula found
across `src/` is inventoried below. **Inventory only — no fixes proposed, no source modified.**

**Working dir:** `/Users/massimomistretta/Claude_Projects/Adaptive_Organization`
**Branch:** `feat/detailed-ecosystemic-report`
**Date:** 2026-07

## Reference papers in `_papers/`

| Short key | File | Validates |
|---|---|---|
| Ulanowicz-2009 | `Quantifying Sustainability Resilience Efficiency.pdf` | TST, C, A, Φ, α, robustness R, fitness Eq.16, window of vitality |
| Zorach-Ulanowicz-2003 | `Quantifying the Complexity of Flow Networks- How many roles are there?.pdf` | Effective flows/nodes/connectivity, number of roles R=exp(AMI) |
| Fath-2019 | `Measuring regenerative economics_ 10 principles and measures undergirding systemic economic health.pdf` | 10 principles; OASIS→principle mapping; autocatalysis; mutualism |
| Ulanowicz-central-theory | `Some steps toward a central theory of ecosystem dynamics.pdf` | Ascendency theory, window of vitality |
| Ulanowicz-dual | `Dual Nature of Ecosystem Dynamics.pdf` | Order/flexibility balance |
| Ulanowicz-process-ecology | `Process_Ecology_A_Transactional_Worldview (1).pdf` | Conceptual foundation |
| ENA-escape-machine | `Ecological network analysys escape from the machine.PDF` | ENA methods |
| Heymans | `Heymans.pdf` | Florida Bay reference values (α=0.367) |
| SFlorida-graminoid | `Network Analysis of Trophic Dynamics in South Florida Ecosystems...Graminoid...pdf` | Reference ecosystem values |
| ENA-quant-methods | `Quntitative methods for ecological network analysis.pdf` | Finn cycling, Lindeman, ENA metrics |
| SupplyChain-complexity | `Towards a use of network analysis- quantifying the complexity of Supply Chain Networks .pdf` | Roles/complexity applied to non-ecological networks |

The papers cover the **Ulanowicz IT core**, **Zorach roles/complexity**, and **Fath 10-principles**.
There is **no paper** dedicated to the OASIS 5-dimension composite, its weights, its normalization
caps, or its HEALTHY/WARNING/CRITICAL bands — those are **proprietary** and must be validated by
internal design logic, not against literature.

---

## A. Core Ulanowicz information-theoretic measures  (validate vs Ulanowicz-2009)

Two implementations exist for most: the loop-based reference in `ulanowicz_calculator.py` and the
numpy `vectorized_metrics.py`. Both must be validated and shown to agree.

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| U1 | TST (Total System Throughput) | `ulanowicz_calculator.py:108,161`; `vectorized_metrics.py:41` | `np.sum(self.flow_matrix)` | Ulanowicz peer-reviewed | Ulanowicz-2009 | **HIGH** |
| U2 | AMI (Average Mutual Information) | `ulanowicz_calculator.py:202-205`; `vectorized_metrics.py:115-133` | `ami_sum += flow_ij*log((flow_ij*tst)/(output_i*input_j))`; `/tst` | Ulanowicz peer-reviewed | Ulanowicz-2009 | **HIGH** |
| U3 | Ascendency A | `ulanowicz_calculator.py:246-249`; `vectorized_metrics.py:166-177` | `ascendency_sum += flow_ij*log((flow_ij*tst)/(output_i*input_j))` (NOT ÷tst) | Ulanowicz peer-reviewed | Ulanowicz-2009 Eq.12 | **HIGH** |
| U4 | Development Capacity C | `ulanowicz_calculator.py:284-286`; `vectorized_metrics.py:210-213` | `capacity_sum += flow_ij*log(flow_ij/tst)`; return `-capacity_sum` | Ulanowicz peer-reviewed | Ulanowicz-2009 Eq.11 | **HIGH** |
| U5 | Reserve/Overhead Φ | `ulanowicz_calculator.py:301-304,361`; `vectorized_metrics.py:238-241` | `development_capacity - ascendency` (Φ = C − A) | Ulanowicz peer-reviewed | Ulanowicz-2009 Eq.13/14 | **HIGH** |
| U6 | Relative Ascendency α = A/C | `ulanowicz_calculator.py:320-323`; `vectorized_metrics.py:269-275` | `ascendency / development_capacity` | Ulanowicz peer-reviewed | Ulanowicz-2009 | **HIGH** |
| U7 | Flow Diversity H (Shannon) | `ulanowicz_calculator.py:489-492`; `vectorized_metrics.py:71-80` | `p_ij = flow_ij/tst; diversity_sum += p_ij*log(p_ij)`; return `-sum` | Ulanowicz/Shannon | Ulanowicz-2009 | MED |
| U8 | Conditional Entropy Hc = H − AMI | `ulanowicz_calculator.py:671-678` | `flow_diversity - ami`; `max(0, .)` | Ulanowicz peer-reviewed | Ulanowicz-2009 | MED |
| U9 | Structural Information SI = log(n²) − H | `ulanowicz_calculator.py:507-509` | `math.log(n_nodes**2) - flow_diversity` | Ulanowicz/derived | Ulanowicz-2009 | MED |
| U10 | Overhead ratio Φ/C (redundancy) | `ulanowicz_calculator.py:422,692-695`; `vectorized_metrics.py:506-507` | `overhead / development_capacity` | Ulanowicz peer-reviewed | Ulanowicz-2009 | MED |
| U11 | Fundamental-relationship check C = A + Φ | `ulanowicz_calculator.py:339-350` | `relative_error < 0.001` | Validation/tolerance | Ulanowicz-2009 | MED |

**Verify fundamental identity holds in both modes.** Note U3 is A (un-normalized sum) while U2 is
AMI (A/TST); the report layer's stated `A = TST × AMI` (see F-block) is the same identity — confirm.

## B. Robustness & Window of Viability  (validate vs Ulanowicz-2009)  — **suspected-issue cluster**

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| R1 | Robustness R = −α·ln(α) | `ulanowicz_calculator.py:548-549`; `vectorized_metrics.py:445-448,480-483` | `-a_c_ratio*math.log(a_c_ratio)`; guard `0<α<1` | Ulanowicz peer-reviewed | Ulanowicz-2009 | **HIGH** |
| R2 | Window-of-Viability bounds (0.2·C, 0.6·C) | `ulanowicz_calculator.py:379-380` | `lower=0.2*development_capacity`; `upper=0.6*development_capacity` | Threshold/constant | Ulanowicz-2009 | **HIGH** |
| R3 | `is_viable` test (bounds vs **A**, not α) | `ulanowicz_calculator.py:428` | `lower_bound <= ascendency <= upper_bound` | Threshold/constant | Ulanowicz-2009 | **HIGH** |
| R4 | α viability band [0.2, 0.6] (dimensionless) | `report_intelligence.py:13-14`; `oasis_calculator.py:920,928`; many report files | `VIABILITY_LOWER=0.2; VIABILITY_UPPER=0.6` | Threshold/constant | Ulanowicz-2009 | **HIGH** |
| R5 | Robustness optimum 1/e ≈ 0.3679 | `report_intelligence.py:15`; `oasis_calculator.py:609` | `ROBUSTNESS_OPTIMUM = 0.367879441`; `1/math.e` | Threshold/constant | Ulanowicz-2009 | **HIGH** |
| R6 | Robustness optimum quoted as 0.37 | `oasis_calculator.py:623,880`; `ulanowicz_calculator.py:880`; report files (many) | `optimal_alpha = 0.37`; `optimal_ratio = 0.37` | Threshold/constant | Ulanowicz-2009 | **HIGH** |
| R7 | Fitness for Evolution (Eq.16), β=1.288, opt α≈0.4596 | `ulanowicz_calculator.py:855-861`; `oasis_calculator.py:282-288` | `-e*alpha_beta*log(alpha_beta)`, `alpha_beta=alpha**beta` | Ulanowicz peer-reviewed | Ulanowicz-2009 Eq.16 | MED |
| R8 | Regenerative Capacity = R·(1−\|α−0.37\|) | `ulanowicz_calculator.py:877-887` | `robustness*(1 - abs(current_ratio-0.37))` | OASIS/derived | proprietary blend — validate by logic | MED |
| R9 | Robustness theoretical max 0.531 = log2(e)/e | report text: `publication_report.py:125`; `latex_report_generator.py:249` | `0 <= R <= log2(e)/e (~0.531)`; point `(0.37,0.531)` | Threshold/constant | Ulanowicz-2009 | LOW |
| R10 | Distance-to-optimum | `report_intelligence.py:70`; report files | `abs(alpha - ROBUSTNESS_OPTIMUM)`; `abs(alpha-0.37)` | Threshold/constant | Ulanowicz-2009 | LOW |

## C. Zorach–Ulanowicz roles / effective-complexity family  (validate vs Zorach-Ulanowicz-2003)

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| Z1 | Effective # flows F = exp(H) | `ulanowicz_calculator.py:1001-1002`; `vectorized_metrics.py:295-296` | `np.exp(flow_diversity)` | Ulanowicz peer-reviewed | Zorach-Ulanowicz-2003 | MED |
| Z2 | Effective # nodes N = exp(½·Σ w·ln(T²/(Ti·T·j))) | `ulanowicz_calculator.py:1041-1043`; `vectorized_metrics.py:333-345` | `np.exp(0.5*sum_term)` | Ulanowicz peer-reviewed | Zorach-Ulanowicz-2003 | MED |
| Z3 | Effective connectivity C = exp(½·Σ w·ln(Tij²/(Ti·T·j))) | `ulanowicz_calculator.py:1084-1086`; `vectorized_metrics.py:388-396` | `np.exp(0.5*sum_term)` | Ulanowicz peer-reviewed | Zorach-Ulanowicz-2003 | MED |
| Z4 | Number of roles R = exp(AMI) | `ulanowicz_calculator.py:1113-1114`; `vectorized_metrics.py:417-418` | `np.exp(ami)` | Ulanowicz peer-reviewed | Zorach-Ulanowicz-2003 | MED |
| Z5 | Functional diversity = log(R) = AMI | `ulanowicz_calculator.py:1166` | `np.log(num_roles)` | Ulanowicz peer-reviewed | Zorach-Ulanowicz-2003 | LOW |
| Z6 | roles/node, specialization R/N | `ulanowicz_calculator.py:1164-1165` | `num_roles/eff_nodes`; `num_roles/n_nodes` | derived | Zorach-Ulanowicz-2003 | LOW |
| Z7 | Roles consistency check (R=N²/F etc.) | `ulanowicz_calculator.py:1145-1156` | `abs(num_roles - eff_nodes**2/eff_flows)` | Validation | Zorach-Ulanowicz-2003 | LOW |
| Z8 | Effective Link Density (custom) | `ulanowicz_calculator.py:587-597` | `(active_links/max_links)*(ami/max_ami)` | OASIS/derived | proprietary — validate by logic | LOW |

## D. Cycling / trophic / Fath-principle measures  (validate vs Fath-2019, ENA-quant-methods)

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| D1 | Finn Cycling Index (approx: self-loops + 2-cycles) | `ulanowicz_calculator.py:719-729` | `diag + Σmin(F,Fᵀ)/2`; `min(cycling_flow/tst,1)` | Fath/ENA (approximation) | ENA-quant-methods | MED |
| D2 | Finn Cycling Index (Leontief, full) | `ecosystem_flow_calculator.py:140-144` | `leontief=inv(I-flow_norm)`; `fci=(Σleontief-n)/Σleontief` | Fath/ENA peer-reviewed | ENA-quant-methods | MED |
| D3 | Autocatalytic Index (proprietary blend) | `ulanowicz_calculator.py:815-818`; `oasis_calculator.py:185-188` | `0.5*count_factor + 0.5*min(1, cycle_flow_ratio*10)` | OASIS proprietary composite | Fath-2019 (concept) | MED |
| D4 | Cycle flow ratio | `ulanowicz_calculator.py:811`; `oasis_calculator.py:181` | `cycle_flow / tst` | Fath/derived | Fath-2019 | LOW |
| D5 | Trophic depth (avg shortest path, unweighted) | `ulanowicz_calculator.py:628` | `nx.average_shortest_path_length(G)` | Network-science standard | ENA-quant-methods | MED |
| D6 | Mutualism ratio / weighted mutualism | `oasis_calculator.py:229,247` | `mutual_pairs/total_connected`; `weighted_mutual/weighted_total` | Fath/derived | Fath-2019 (P8) | MED |
| D7 | Lindeman trophic efficiency | `ecosystem_flow_calculator.py:194-196` | `1-(total_respiration/(tst+Σimports))` | Fath/ENA | ENA-quant-methods | LOW |
| D8 | Extended TST (imports/exports/respiration) | `ecosystem_flow_calculator.py:100` | `internal_tst+imports+exports+respiration` | Fath/ENA | ENA-quant-methods | LOW |
| D9 | Import dependency / export / respiration ratios | `ecosystem_flow_calculator.py:217-219` | `Σimports/tst_ext` etc. | Fath/derived | ENA-quant-methods | LOW |

## E. Network-science standard metrics  (validate vs standard network science)

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| N1 | Density | `ulanowicz_calculator.py:914`; `network_analyzer.py:489` | `nx.density(G)` | Network-science standard | standard (networkx) | MED |
| N2 | Connectance m/(n(n−1)) | `ulanowicz_calculator.py:915`; `database/precompute_pipeline.py:118` | `m/(n*(n-1))` | Network-science standard | standard | MED |
| N3 | Link density m/n | `ulanowicz_calculator.py:916`; `precompute_pipeline.py:119` | `m/n` | Network-science standard | standard | LOW |
| N4 | Degree centralization (Freeman) | `ulanowicz_calculator.py:956-963` | `sum_diff/((n-1)*(n-2))` | Network-science standard | standard | LOW |
| N5 | Degree heterogeneity (CoV of degrees) | `ulanowicz_calculator.py:968` | `np.std(all_degrees)/np.mean(all_degrees)` | Network-science standard | standard | LOW |
| N6 | Clustering coefficient | `ulanowicz_calculator.py:943`; `network_analyzer.py:209` | `nx.average_clustering(G)` | Network-science standard | standard | LOW |
| N7 | Centralities (betweenness, eigenvector, closeness, pagerank α=0.85, katz α=0.1) | `network_analyzer.py:86-123` | `nx.betweenness/eigenvector/closeness/pagerank/katz_centrality` | Network-science standard | standard | MED |
| N8 | Modularity (Louvain seed=42, label-prop, greedy) | `network_analyzer.py:145-183` | `community.modularity(G, communities, weight='weight')` | Network-science standard | standard | MED |
| N9 | Small-world σ = (C/Cr)/(L/Lr) | `network_analyzer.py:235-237` | `C_ratio/L_ratio` | Network-science standard | standard | LOW |
| N10 | Small-world ω | `network_analyzer.py:244` | `(Lr/L)-(C/Cr)` | Network-science standard | standard | LOW |
| N11 | Random-graph baselines (ER) | `network_analyzer.py:227,230-231` | `p=2m/(n(n-1))`; `Lr=log(n)/log(<k>)` | Network-science standard | standard | LOW |
| N12 | Degree assortativity (total/in/out) | `network_analyzer.py:275-287` | `nx.degree_assortativity_coefficient(G, weight)` | Network-science standard | standard | LOW |
| N13 | Rich-club coefficient (k = 90th pctile) | `network_analyzer.py:314-320` | `nx.rich_club_coefficient(G, normalized=False)` | Network-science standard | standard | LOW |
| N14 | Robustness: random-failure & targeted-attack (mean GCC/original) | `network_analyzer.py:379,409` | `np.mean(gcc_sizes)/original_gcc_size` | Network-science standard | standard | LOW |
| N15 | Percolation threshold 1/<k> | `network_analyzer.py:412-413` | `1/avg_degree` | Network-science standard | standard | LOW |
| N16 | Path redundancy (# simple paths, cutoff 3) | `network_analyzer.py:421-427` | `np.mean(len(paths))` | Network-science standard | standard | LOW |
| N17 | Flow reciprocity | `network_analyzer.py:472`; `oasis_calculator.py` (via mutualism) | `reciprocal_flows/total_edges` | Network-science standard | standard | LOW |
| N18 | Throughput efficiency | `network_analyzer.py:459-460` | `total_flow/(n(n-1)*max_flow)` | OASIS/derived | proprietary — validate by logic | LOW |

## F. Statistical / distribution measures  (validate vs standard statistics)

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| S1 | Gini coefficient (flows) | `oasis_calculator.py:463`; `network_analyzer.py:446`; `publication_report.py:690` | `(2*Σ(index*sorted))/(n*Σsorted) - (n+1)/n` | Network-science standard | standard | MED |
| S2 | Flow CoV (std/mean) | `network_analyzer.py:453`; `publication_report.py:154`; `pdf_generator.py:824` | `np.std(flows)/np.mean(flows)` | Network-science standard | standard | LOW |
| S3 | Flow heterogeneity | `network_analyzer.py:453` | `np.std(flows)/np.mean(flows)` | Network-science standard | standard | LOW |
| S4 | Shannon flow diversity (fallback) | `database/precompute_pipeline.py:159` | `-np.sum(p_nonzero*np.log(p_nonzero))` | Shannon standard | Ulanowicz-2009 | LOW |
| S5 | Flow-diversity utilization % | `publication_report.py:266-267` | `fd/log2(n²)*100` | derived | Ulanowicz-2009 | LOW |
| S6 | A/Φ ratio | `publication_report.py:300` | `ascendency/overhead` | derived | Ulanowicz-2009 | LOW |

## G. OASIS composite (PROPRIETARY — validate by internal logic, NOT literature)  — **suspected-issue cluster**

| ID | Quantity | Source file:line | Verbatim expression | Category | Paper | Prio |
|---|---|---|---|---|---|---|
| O1 | OPEN dimension raw score | `oasis_calculator.py:333-338` | `0.25*conn + 0.30*normFD + 0.25*avgBetween + 0.20*clustering` | OASIS proprietary composite | proprietary — no paper | MED |
| O2 | AUTONOMOUS raw score | `oasis_calculator.py:406-411` | `0.35*FCI + 0.25*recip + 0.25*normAMI + 0.15*autocat` | OASIS proprietary composite | proprietary — no paper | MED |
| O3 | SYMBIOTIC raw score | `oasis_calculator.py:484-489` | `0.30*(1−gini) + 0.25*modularity + 0.25*nodeRatio + 0.20*mutualism` | OASIS proprietary composite | proprietary — no paper | MED |
| O4 | INTELLIGENT raw score | `oasis_calculator.py:556-561` | `0.35*roles + 0.25*divers + 0.20*rolesPerNode + 0.20*condEntropy` | OASIS proprietary composite | proprietary — no paper | MED |
| O5 | SUSTAINABLE raw score | `oasis_calculator.py:633-638` | `0.30*normRob + 0.20*inWindow + 0.20*normRegen + 0.30*alphaOpt` | OASIS proprietary composite | proprietary — no paper | **HIGH** |
| O6 | Normalize-to-100 (per-dim min/max caps) | `oasis_calculator.py:99-104,341,414,492,564,641` | `normalized*100`, caps: OPEN 0.6 / AUT 0.5 / SYM 0.7 / INT 0.6 / SUS 0.8 | OASIS proprietary composite | proprietary — no paper | **HIGH** |
| O7 | Overall = Σ dim·weight (default 20% each) | `oasis_calculator.py:41-47,695-698` | `sum(scores[dim]*self.weights[dim])` | OASIS proprietary composite | proprietary — no paper | **HIGH** |
| O8 | Overall band HEALTHY≥60 / WARNING≥40 / CRITICAL | `oasis_calculator.py:713-718` | `if overall>=60 ... elif overall>=40 ...` | Threshold/constant (proprietary) | proprietary — no paper | **HIGH** |
| O9 | Per-dimension HEALTH_THRESHOLDS (asymmetric bands) | `oasis_calculator.py:50-56,701-708` | e.g. sustainable healthy (60,95) warning (40,60) critical (0,40) | Threshold/constant (proprietary) | proprietary — no paper | **HIGH** |
| O10 | α-optimality (distance from 0.37) | `oasis_calculator.py:624-626` | `max(0, 1-(abs(alpha-0.37)/0.37))` | OASIS proprietary composite | proprietary — no paper | MED |
| O11 | norm_robustness = R/(1/e) | `oasis_calculator.py:609-610` | `robustness/(1/math.e)` | derived | Ulanowicz-2009 | MED |
| O12 | Sub-metric normalization constants | `oasis_calculator.py:538,548,619,630` | `roles/10`, `rolesPerNode/2`, `regen/0.3`, `fitness/0.4` | Threshold/constant (proprietary) | proprietary — no paper | MED |
| O13 | Recommendation triggers (α<0.2 / α>0.6 CRITICAL; gini>0.5; roles<3) | `oasis_calculator.py:896,907,920,928` | `if alpha<0.2 ... elif alpha>0.6 ...` | Threshold/constant | proprietary — no paper | MED |

## H. Report-layer verdict bands & benchmark thresholds  (mostly proprietary presentation logic)

These recompute or re-band already-computed values; validate for **self-consistency** with the engine.

| ID | Quantity | Source file:line | Verbatim expression | Category | Prio |
|---|---|---|---|---|---|
| H1 | Benchmark "high-performing org" α band [0.30,0.45] | `pdf_generator.py:408,750`; `publication_report.py:321`; `latex_report_generator.py:276` | `0.30 <= alpha <= 0.45` | Threshold/constant (proprietary) | MED |
| H2 | Report "Network Efficiency = A/(C·log2 n)" (text def) | `publication_report.py:~432 (Appendix)` | `A/(C*log2(n))` | Threshold/derived | **HIGH** (see Issue 4) |
| H3 | Report "Regenerative Capacity = (Φ/C)·(1−\|α−0.37\|)" (text def) | `publication_report.py:~434` | `(Φ/C)*(1-abs(alpha-0.37))` | derived | MED |
| H4 | Robustness verbal bands (0.15/0.20/0.25) | `publication_report.py:283-288`; `pdf_generator.py:398`; `latex_report_generator.py:265` | `rob>0.20 ... rob>0.15` | Threshold/constant | LOW |
| H5 | Efficiency verbal bands (0.2/0.4/0.6) | `publication_report.py:642-652`; `latex:373-383`; `main.py:166,169` | `<0.2 Low ... <0.6 High` | Threshold/constant | LOW |
| H6 | Gini inequality bands (0.3/0.6) | `publication_report.py:235`; `pdf_generator.py:852-854` | `gini>0.6 high / >0.3 moderate` | Threshold/constant | LOW |
| H7 | Redundancy bands (0.3/0.6) | `publication_report.py:707-713`; `pdf_generator.py:700` | `>0.6 High / >0.3 Moderate` | Threshold/constant | LOW |
| H8 | α position bands [<0.2,<0.35,<0.45,<0.6] | `publication_report.py:668-680` | interpret-position bands | Threshold/constant | LOW |
| H9 | Risk fragility bands (α vs [0.2,0.6], 0.05 edge warning) | `report_intelligence.py:110-158` | `alpha<0.2 under / >0.6 over`; `<0.05` edge | Threshold/constant | MED |
| H10 | ESG crosswalk (qualitative lookup, not scored) | `report_intelligence.py:214-262` | `_ESG_CROSSWALK` dict | Non-scored lookup | LOW |
| H11 | Action roadmap severity→horizon bucketing | `report_intelligence.py:195-209` | `{'CRITICAL':'immediate',...}` | Non-scored lookup | LOW |
| H12 | ecosystem health bands (respiration 0.3/0.6/0.7, FCI 0.1/0.2/0.5, import 0.2/0.5) | `ecosystem_flow_calculator.py:237-261` | `respiration_ratio>0.7` etc. | Threshold/constant | LOW |
| H13 | main.py CLI assessment bands (eff 0.2/0.6, rob 0.1/0.25, regen 0.1/0.2) | `main.py:166-186` | `if efficiency<0.2 ... robustness>0.25` | Threshold/constant | LOW |

## I. Published reference values & validation tolerances  (validate the stored NUMBERS vs source papers)

Stored literal ecosystem metric values used as benchmark anchors (`services/published_metrics_db.py`).
These are **claimed measurements** — validation = confirm each number matches its cited paper.

| ID | Network | Key stored values | Source:line | Prio |
|---|---|---|---|---|
| P1 | cone_spring_original (Ulanowicz&Norden 1990, log2) | TST 42016, C 135000, A 68191, Φ 66809, α **0.505**, AMI 1.623, H 3.213 | `published_metrics_db.py:67-96` | MED |
| P2 | cone_spring_eutrophicated (Ulanowicz 2009) | α **0.529**; note embeds "optimal 0.460" | `published_metrics_db.py:120-121` | MED |
| P3 | crystal_river_creek (Ulanowicz 1986, log2, tol 0.10) | TST 97916, C 204355, A 112891, Φ 91464, α **0.552** | `published_metrics_db.py:144-163` | MED |
| P4 | florida_bay (Heymans 2002, tol 0.10) | α **0.367** | `published_metrics_db.py:186` | MED |
| P5 | prawns_alligator_{original,efficient,adapted} | TST 102.6/121.8/99.7; A 53.9/100.3/44.5; Φ 121.3/0.0/68.2 | `published_metrics_db.py:211-290` | MED |
| P6 | Validation tolerances | default 0.05; crystal/florida 0.10; fundamental 0.001; WARNING at 2× tolerance | `published_metrics_db.py:44,388`; `scientific_validation_agent.py:199-202` | MED |
| P7 | log2↔ln conversion for validation | `x/ln2`, `ln2=math.log(2)` | `scientific_validation_agent.py:160-169` | MED |
| P8 | Validation invariants (0≤α≤1, A≤C, TST>0, Φ≥0, 0≤FCI≤1) | ranges | `published_metrics_db.py:393-414`; `scientific_validation_agent.py:242-302` | MED |
| P9 | EXAMPLE_METRICS embedded published values | ascendency 68191/53.9; C 135000; α 0.505/0.529/0.552 | `services/new_metric_checklist.py:459-500` | LOW |

---

# The FOUR suspected issues — exact code found

## Issue 1 — OASIS roll-up: 3 dims at 100 can outvote a CRITICAL dim
**This is a proprietary DESIGN-logic question, not a peer-reviewed formula.**

Aggregation is a flat weighted mean with equal 20% weights:
```
oasis_calculator.py:41-47   DEFAULT_WEIGHTS = {'open':0.20,'autonomous':0.20,'symbiotic':0.20,'intelligent':0.20,'sustainable':0.20}
oasis_calculator.py:695-698 overall = sum(scores[dim] * self.weights[dim] for dim in scores)
```
Overall band (independent of any single dimension's status):
```
oasis_calculator.py:713-718
    if overall >= 60:   overall_status = 'HEALTHY'
    elif overall >= 40: overall_status = 'WARNING'
    else:               overall_status = 'CRITICAL'
```
Per-dimension status is computed separately (`get_status`, L701-708) and does **not** gate the overall
band. **Confirmed design flaw surface:** e.g. OPEN/AUT/SYM/INT = 100 and SUSTAINABLE = 0 →
overall = 0.20·(100·4) = 80 → `HEALTHY`, even though SUSTAINABLE is `CRITICAL`. There is **no
"floor" / no "worst-dimension caps the verdict" rule** anywhere. The per-dimension CRITICAL only
surfaces in narrative/risk items (`report_intelligence.build_risk_view` L160-168), never in the headline band.
→ **Validate by internal logic (proprietary); recommend a floor/veto rule in the fix pass.**

## Issue 2 — Window-of-Viability bounds in capacity units vs α as a 0–1 ratio  (SCALE MISMATCH)
The engine computes the bounds as **absolute capacity units** and tests them against **A** (also
capacity units) — internally consistent:
```
ulanowicz_calculator.py:378-382
    development_capacity = self.calculate_development_capacity()
    lower_bound = 0.2 * development_capacity     # capacity units (flow-nats)
    upper_bound = 0.6 * development_capacity
ulanowicz_calculator.py:428   'is_viable': lower_bound <= ascendency <= upper_bound   # A vs 0.2C..0.6C  (consistent)
```
But **everywhere downstream** the "viability band" is compared against **α = A/C**, a 0–1 ratio, using
the *same numbers 0.2 and 0.6* as if they were on the α scale:
```
report_intelligence.py:13-14,53-58,68   VIABILITY_LOWER=0.2; VIABILITY_UPPER=0.6; alpha < VIABILITY_LOWER ...
oasis_calculator.py:920,928             if alpha < 0.2 ... elif alpha > 0.6
publication_report.py / pdf_generator / oasis_pdf_report / latex   "0.2 < alpha < 0.6"
```
Because A ≤ 0.2C ⇔ α ≤ 0.2, the α-band [0.2,0.6] test and the A-vs-[0.2C,0.6C] test are in fact
**mathematically equivalent** (dividing both sides by C). So the two representations agree *when α is
used consistently* — **but** `is_viable` (the flag the SUSTAINABLE dimension reads at
`oasis_calculator.py:613`) is the **A-based** version, while the report narratives independently
re-derive the α-based version. They should always agree; **the risk is any place that mixes an
absolute bound with a ratio.** No place was found comparing `0.2*C` directly against `α` (that would
be the true bug); the exposure is that the bounds are stored/exported as capacity-unit numbers
(`viability_lower_bound`/`viability_upper_bound`, L426-427) and could be misused as α-scale elsewhere.
→ **HIGH priority to validate the two paths never diverge and that exported bounds are labeled by unit.**

## Issue 3 — The [0.2,0.6] band and robustness optimum (1/e vs 0.37 vs 0.4596): consistency
- **α viability band [0.2, 0.6]** literal in: `report_intelligence.py:13-14`; `oasis_calculator.py:920,928`;
  `ulanowicz_calculator.py:379-380`; and as text in every report generator.
- **Robustness optimum** appears as THREE different numbers:
  - `1/e = 0.367879441` — `report_intelligence.py:15`; `oasis_calculator.py:609` (`1/math.e`)  → the **true** maximizer of R = −α·ln(α).
  - `0.37` (rounded) — `oasis_calculator.py:623,880`; `ulanowicz_calculator.py:880`; `pdf_generator.py:782`; `publication_report.py:118,180`; `latex_report_generator.py:249`; glossaries.  Used as the **target** in α-optimality (O10) and Regenerative Capacity (R8).
  - `0.4596` — appears **only in docstrings/comments** as the maximizer of the *fitness* function Eq.16 (β=1.288), NOT robustness: `ulanowicz_calculator.py:836,853`; `oasis_calculator.py:266,281`. The literal `0.460` is embedded as a note in `published_metrics_db.py:121`.
- **Inconsistency to flag:** the codebase uses `0.37` as the scoring target for α-optimality (O10) and
  regenerative capacity (R8), but the "true" robustness optimum is `1/e ≈ 0.3679`, and Ulanowicz's
  *window-of-vitality geometric center / fitness optimum* is `0.4596`. Three distinct constants for
  "optimal α" coexist. → **HIGH: validate which target each formula should use per Ulanowicz-2009.**

## Issue 4 — "Network Efficiency" vs α: are they the same expression?
- In the **engine**, Network Efficiency is **literally α = A/C**:
```
ulanowicz_calculator.py:567-570   calculate_network_efficiency(): return ascendency/development_capacity  (== α)
vectorized_metrics.py:508         'network_efficiency': relative_ascendency,   # explicit alias
```
- But the **publication report Appendix** defines it **differently**, with an extra log(n) factor:
```
publication_report.py (Appendix A.1, ~L432)   Network Efficiency = A / (C · log2(n))
```
→ **CONFIRMED discrepancy:** the app computes `network_efficiency = α = A/C`, while the printed
methodology claims `A/(C·log2 n)`. These are **not** the same expression. HIGH priority — either the
code or the documented formula is wrong; also note `_assess_efficiency` and `main.py` bands treat
"efficiency" as α (0.2/0.6 bands), reinforcing that the engine value is α, so the Appendix text is the outlier.

---

# Validation plan — 6 families for parallel validation

1. **Core Ulanowicz information measures** (Group A: U1–U11) — validate every expression and the
   loop-vs-vectorized agreement against Ulanowicz-2009 Eqs. 11–14. HIGH: U1–U6.
2. **Robustness & Window of Viability** (Group B: R1–R10, + Issues 2 & 3) — validate R=−α·ln(α),
   the 0.2/0.6 bounds, unit consistency (α vs capacity units), and the 1/e vs 0.37 vs 0.4596 constants.
   HIGH: R1–R6.
3. **Zorach roles & effective-complexity + cycling/trophic/Fath** (Groups C & D: Z1–Z8, D1–D9) —
   validate exp(H)/exp(AMI) family vs Zorach-Ulanowicz-2003; Finn cycling (two impls D1/D2) and
   Lindeman/mutualism/autocatalysis vs Fath-2019 & ENA-quant-methods.
4. **Network-science standard metrics + statistics** (Groups E & F: N1–N18, S1–S6) — validate against
   standard definitions (Gini, modularity, centralities, small-world, assortativity, rich-club,
   percolation, CoV). Flag proprietary ones (N18 throughput efficiency, Z8 ELD).
5. **OASIS composite (PROPRIETARY)** (Group G + H benchmark bands: O1–O13, H1) — validate by
   **internal design logic only** (no literature): weights, normalization caps, band thresholds, and
   the **roll-up floor problem (Issue 1)**, the α-optimality target (Issue 3), and the
   Network-Efficiency definition mismatch (Issue 4 / H2). HIGH: O5–O9, H2.
6. **Published reference values, tolerances & report verdict bands** (Groups I & H: P1–P9, H2–H13) —
   confirm every stored ecosystem number matches its cited paper (log2 vs natural base!), the
   validation tolerances/invariants, and that report-layer re-bandings are self-consistent with the engine.

---

## Summary counts

- **Total formulas / quantities inventoried:** 99
  (A:11, B:10, C:8, D:9, E:18, F:6, G:13, H:13, I:9, + engine-vs-report duplicates counted once)
- **By category:**
  - Ulanowicz peer-reviewed: ~24 (Groups A, B core, C, parts of D)
  - Fath peer-reviewed / ENA: ~9 (Group D)
  - Network-science standard: ~24 (Groups E, F)
  - OASIS proprietary composite: ~15 (Group G + N18, Z8, D3, R8)
  - Threshold/constant: ~27 (Group B bounds, H bands, I tolerances, O8/O9/O12/O13)
- **HIGH-priority formulas:** U1 (TST), U2 (AMI), U3 (A), U4 (C), U5 (Φ), U6 (α), R1 (Robustness),
  R2 (WoV bounds 0.2C/0.6C), R3 (is_viable), R4 (α band [0.2,0.6]), R5 (1/e optimum), R6 (0.37 optimum),
  O5 (SUSTAINABLE score), O6 (normalization caps), O7 (overall weighted mean), O8 (overall band),
  O9 (per-dim thresholds), H2 (Network Efficiency def mismatch).
