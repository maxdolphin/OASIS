# Validation — Families E (Network-Science) & F (Statistical) Formulas

**Scope:** N1–N18 (network-science standard) and S1–S6 (statistics) from `formula-inventory.md`.
**Method:** Compare code to canonical definitions (Newman *Networks* 2nd ed.; Freeman 1978/79 centralization;
Humphries & Gurney 2008 σ; Telford et al. 2011 ω; standard sorted-Gini). Numeric cross-checks run in Python.
**Rule:** validation only — no source modified. Flow networks are **DIRECTED** (`nx.DiGraph`, `network_analyzer.py:51-65`).

**Files:** `network_analyzer.py`, `ulanowicz_calculator.py`, `database/precompute_pipeline.py`,
`publication_report.py`, `pdf_generator.py`, `oasis_calculator.py`.

---

## Headline findings

1. **Gini (S1) — 3 implementations AGREE and match the canonical sorted-Gini.** Verified byte-identical
   formula in all three sites and numerically equal to the mean-absolute-difference Gini to 1e-16
   (`oasis_calculator.py:463`, `network_analyzer.py:446`, `publication_report.py:690`). No off-by-one:
   indices run `1..n` ascending on a `np.sort`-ascending array; the `(n+1)/n` term is correct. **OK.**

2. **Small-world random baseline (N11) is COMPUTED WRONG → propagates to σ (N9) and ω (N10).**
   `network_analyzer.py:231` uses `nx.average_degree_connectivity(G).get(1, 2)` as `<k>`. That function
   returns a dict **keyed by node degree** whose values are the *average neighbour degree*; `.get(1,2)`
   pulls the avg-neighbour-degree of degree-1 nodes (default 2). This is **not the mean degree `<k>`**.
   The intended `Lr = ln(n)/ln(<k>)` is therefore corrupted, and both σ and ω (and `is_small_world`)
   are unreliable. **MAJOR.**

3. **ω (N10) uses the random-graph clustering, not a lattice.** Telford's ω = `Lr/L − C/C_latt` uses the
   **lattice** clustering in the second term; the code uses `C_random` (`network_analyzer.py:244`). The
   σ form (Humphries) is correct; ω is a definitional deviation. **MINOR–MAJOR.**

4. **Directed graph run through undirected formulas.** Clustering, assortativity, rich-club, small-world,
   and the ER baseline are all computed on `G.to_undirected()` or with undirected normalizations while the
   real network is directed. Some are defensible (community detection conventionally undirected), others
   silently discard directionality that matters for flow networks. Flagged per-row below.

5. **Freeman centralization (N4) denominator `(n−1)(n−2)` is the UNDIRECTED max applied to in/out
   degree of a DIRECTED graph.** For a directed graph the theoretical maximum of Σ(d_max − d_i) for the
   in- or out-degree is `(n−1)²`, not `(n−1)(n−2)`. Using the undirected star normalization on directed
   in/out degrees under-normalizes and can push the coefficient above 1. **MAJOR.**

---

## E. Network-science metrics

| ID | Matches canonical def? | Directed-graph correctness | Magic-number flags | Severity | Correct form / citation | Fix backed by std def? |
|---|---|---|---|---|---|---|
| N1 Density `nx.density(G)` (`ulanowicz_calculator.py:914`) | Yes | Correct — `nx.density` uses `m/(n(n−1))` for a DiGraph automatically | — | **OK** | Newman §6.10; networkx density | n/a |
| N2 Connectance `m/(n(n−1))` (`ulanowicz_calculator.py:915`, `precompute_pipeline.py:118`) | Yes | Correct for directed (no self-loops) — equals directed density | — | **OK** | directed connectance C = L/(N(N−1)) (May 1972) | n/a |
| N2′ `network_density = m/n²` (`precompute_pipeline.py:117`) | Divergent | Uses `n²` (includes self-loop slots) as denominator, unlike N1/N2 which use `n(n−1)`. Two different "density" definitions coexist. | — | **MINOR** | If self-loops disallowed, use `n(n−1)`; label the `n²` variant explicitly | Yes — pick one denominator consistently |
| N3 Link density `m/n` (`ulanowicz_calculator.py:916`, `precompute_pipeline.py:119`) | Yes | Correct (edges per node; direction-agnostic count) | — | **OK** | standard link density L/N | n/a |
| N4 Degree centralization Freeman `sum_diff/((n−1)(n−2))` (`ulanowicz_calculator.py:956-963`) | **No** | **Denominator is the UNDIRECTED star max**, applied separately to in- and out-degree of a directed graph. Directed max of Σ(d*−dᵢ) is `(n−1)²`. Can exceed 1. | — | **MAJOR** | Freeman (1979): undirected max `(n−1)(n−2)` is for degree centrality normalized to [0,1] *on undirected graphs*. For raw directed in/out-degree, normalizer is `(n−1)²`. | Yes — directed normalizer is `(n−1)²`; or convert to normalized degree centrality first then use `(n−1)(n−2)` per Freeman |
| N5 Degree heterogeneity CoV `std/mean` of degrees (`ulanowicz_calculator.py:968`) | Yes (as CoV) | Concatenates in- and out-degree lists (`:966`) — mixes two distributions; defensible but non-standard | Guard `mean>0` present ✓ | **MINOR** | CoV = σ/μ (standard). For directed nets report in/out CoV separately | Optional |
| N6 Clustering `nx.average_clustering(G)` (`ulanowicz_calculator.py:943`); `nx.average_clustering(G_und, weight)` (`network_analyzer.py:209`) | Partly | `ulanowicz` passes the **DiGraph** → networkx computes the *directed* clustering (Fagiolo) — OK. `network_analyzer` first does `to_undirected()` → discards direction. Two different clustering definitions for the "same" metric. | — | **MINOR** | Fagiolo (2007) directed clustering vs Watts–Strogatz undirected. Choose one; document. | Yes — both are valid; inconsistency is the issue |
| N7 Betweenness `weight='weight'` (`network_analyzer.py:86`) | **Partial** | Uses raw **flow as distance**: shortest paths minimize Σweight, so high-flow edges are treated as *long* — inverted. Betweenness weight should be a **cost/distance**, i.e. `1/flow`. | — | **MAJOR** | Brandes (2001): weighted betweenness treats weight as distance. Strong ties must be inverted (`d=1/w`). | Yes — invert weights for strong-tie networks |
| N7 Closeness `distance='weight'` (`network_analyzer.py:103`) | Partial | Same weight-as-distance inversion problem: high flow → far. | — | **MAJOR** | closeness uses distance; invert flow to cost | Yes |
| N7 Eigenvector `weight='weight'` (`network_analyzer.py:94`) | Yes | Correct — eigenvector uses weight as strength (higher = more influence). Directed DiGraph OK; may need left/right choice. | max_iter=1000 (reasonable) | **OK** | Newman §7.2 | n/a |
| N7 PageRank α=0.85 (`network_analyzer.py:111`) | Yes | Correct — 0.85 is Brin–Page canonical damping; weight as strength is correct. | α=0.85 (standard) | **OK** | Brin & Page (1998) | n/a |
| N7 Katz α=0.1 (`network_analyzer.py:119`) | Yes | Convergence needs α < 1/λmax. Numeric check: sparse graphs λmax≈2–5 → 1/λmax≈0.2–0.4, so 0.1 OK; **dense/large graphs (λmax>10) diverge**. `try/except` falls back to degree centrality → graceful but silent. | α=0.1 (**flag: fixed, not λmax-adaptive**) | **MINOR** | Newman §7.3: require α < 1/λmax(A). Prefer α = f·(1/λmax), f≈0.85. | Yes — adaptive α is the standard-safe form |
| N8 Modularity Louvain(seed=42)/label-prop/greedy, `weight='weight'` (`network_analyzer.py:145-183`) | Yes | Communities computed on `to_undirected()` (line 141) — conventional. `weight='weight'` passed correctly to both `louvain_communities` and `modularity`. seed=42 → reproducible. | seed=42 (reproducibility, OK) | **OK** | Newman & Girvan (2004); Blondel et al. (2008). Directed modularity (Leicht–Newman 2008) exists but undirected is an accepted convention. | n/a |
| N9 Small-world σ = (C/Cr)/(L/Lr) (`network_analyzer.py:235-237`) | Form Yes / inputs No | Undirected (OK for σ convention) **but Lr is corrupted (see N11)**, so σ is unreliable. | `is_small_world = σ>1` threshold (standard) | **MAJOR** (via N11) | Humphries & Gurney (2008): σ=(C/Cr)/(L/Lr). Form correct; baseline broken. | Fix is in N11 |
| N10 Small-world ω = Lr/L − C/Cr (`network_analyzer.py:244`) | **No** | Uses **C_random** in 2nd term; Telford's ω uses **C_lattice**. Also inherits broken Lr (N11). | — | **MAJOR** | Telford, Bassett et al. (2011): ω = Lrand/L − C/Clatt. Second term needs lattice clustering. | Yes — 2nd term must use lattice C |
| N11 ER baselines: `p=2m/(n(n−1))` (`:227`); `Lr=log(n)/log(<k>)` (`:230-231`) | **No (Lr)** | `p` correct for undirected. **`<k>` is wrong**: `nx.average_degree_connectivity(G).get(1,2)` returns avg-neighbour-degree of degree-1 nodes, not mean degree. `Cr=p` OK. | default `<k>`→2 fallback (arbitrary) | **MAJOR** | Fronczak et al. (2004): Lr ≈ ln(n)/ln(⟨k⟩), ⟨k⟩=2m/n. Replace with `2*m/n`. | Yes — `<k>=2m/n` is the standard mean degree |
| N12 Assortativity total/in/out, `weight='weight'` (`network_analyzer.py:275-287`) | Yes | Correct — `degree_assortativity_coefficient` with `x='in',y='in'` / `x='out',y='out'` is the proper **directed** assortativity (Newman 2003; Foster et al. 2010). Best-handled directed metric in the file. | — | **OK** | Newman (2002/2003); directed variants Foster (2010) | n/a |
| N13 Rich-club, k=90th pctile, `normalized=False` (`network_analyzer.py:314-320`) | Partial | Computed on `to_undirected()` — discards direction. **Unnormalized φ(k) is not interpretable** (monotonic in k for most graphs); the ratio to a randomized null is what signals rich-club-ness. | **k=90th percentile (ARBITRARY)**; `normalized=False` | **MAJOR** | Colizza et al. (2006): use `normalized=True` (ratio to degree-preserving randomization). Unnormalized value alone cannot indicate a rich-club effect. | Yes — `normalized=True` is the standard |
| N14 Attack robustness `mean(gcc_sizes)/original` (`network_analyzer.py:379,409`) | Approx | Uses `weakly_connected_components` (right choice for directed GCC). Random-failure averaged over `num_simulations=10` (low). "Area under curve" approximated by `mean(gcc_sizes)` (unnormalized by removal fraction — crude but monotone). | num_simulations=10 (low, flag) | **MINOR** | Schneider et al. (2011) R-index = (1/N)Σ s(Q). Current is a proxy. Directed handling OK. | Optional (proxy acceptable if labeled) |
| N15 Percolation `1/<k>` with `<k>=2m/n` (`network_analyzer.py:412-413`) | Partial | `avg_degree=2m/n` treats graph as undirected. For an ER/undirected net, critical threshold f_c = 1 − 1/⟨k⟩ (giant-component); `1/⟨k⟩` is the **Molloy–Reed / bond-percolation** point, a different quantity. Directed percolation uses ⟨k_in·k_out⟩. | — | **MINOR** | Molloy & Reed (1995); Cohen et al. (2000) f_c=1−1/(κ−1). Current `1/⟨k⟩` is the ER giant-emergence threshold, defensible if labeled as such. | Depends on intended quantity — label it |
| N16 Path redundancy, `cutoff=3`, first ≤10×10 nodes (`network_analyzer.py:421-427`) | Ad hoc | Directed `all_simple_paths` (OK). But **cutoff=3 arbitrary** and **only nodes 0–9 sampled** (`min(10,n)`) → biased, not a whole-graph measure. | **cutoff=3 (ARBITRARY); 10×10 node cap (sampling bias)** | **MAJOR** | No canonical "path redundancy"; if edge-independent paths are meant, use Menger/`node_connectivity`. Current is a non-standard proxy on a biased sample. | Proprietary proxy — flag, don't "fix" to a std def |
| N17 Reciprocity `reciprocal/total_edges` (`network_analyzer.py:463-472`) | Partial | Counts unordered pairs with flow both ways ÷ pairs with any flow. This is the **pair-based reciprocity r = L↔/(L↔+L→)**, a valid directed measure — but *not* Garlaschelli–Loffredo ρ. Denominator counts *undirected pairs* not *directed edges* despite variable name `total_edges`. | — | **MINOR** | Garlaschelli & Loffredo (2004) ρ corrects for density; the simple ratio is the classic reciprocity. Rename var; acceptable metric. | Optional |
| N18 Throughput efficiency `total_flow/(n(n−1)·max_flow)` (`network_analyzer.py:459-460`) | **Proprietary** | Not a standard metric — normalizes TST by a hypothetical fully-connected max-flow network. Directed `n(n−1)` denominator is at least dimensionally consistent. | — | **OK (proprietary)** | No literature; validate by internal logic only | n/a |

---

## F. Statistical / distribution measures

| ID | Matches canonical def? | Directed / data correctness | Magic-number flags | Severity | Correct form / citation | Fix backed by std def? |
|---|---|---|---|---|---|---|
| **S1 Gini** — 3 impls (`oasis_calculator.py:463`, `network_analyzer.py:446`, `publication_report.py:690`; consumed by `pdf_generator.py:818`) | **Yes — all 3 identical & canonical** | Operates on `flows>0` sorted ascending — **correct**. Non-negativity guaranteed by `>0` filter ✓. Index `1..n` ascending, `(n+1)/n` term correct. Single-value (`n≤1`) → 0 guard present in `oasis`/`publication`; `network_analyzer` guards `len(flows)>0` but not `>1` (n=1 still yields 0 by formula: `2·1·x/(1·x) − 2/1 = 0`). | — | **OK** | Sorted-Gini: G = (2·Σ i·xᵢ)/(n·Σx) − (n+1)/n, x ascending (Sen 1973; Damgaard & Weiner 2000). **Numerically verified == MAD-Gini to 1e-16** across 6 test cases. | n/a — all three agree with each other and the canonical def |
| S2 Flow CoV `std/mean` (`publication_report.py:154`) | Yes | On active flows | Guard `mean>0` present ✓ | **OK** | CoV = σ/μ | n/a |
| S2′ CoV `np.std(flows)/np.mean(flows)` (`pdf_generator.py:824`) | Yes | On `flows>0` | **No zero-mean guard** (line 824) — but `flows>0` non-empty implies mean>0, so safe in practice | **OK** | CoV = σ/μ | n/a |
| S3 Flow heterogeneity `std/mean` (`network_analyzer.py:453`) | Yes | Same as CoV | Guard `len(flows)>0` ✓ (mean>0 implied) | **OK** | identical to CoV | n/a |
| S4 Shannon fallback `−Σ p·ln p` (`precompute_pipeline.py:159`) | Yes | `p=flow/TST` over nonzero entries; natural-log units (nats) — consistent with Ulanowicz engine base | Guard `len(p_nonzero)>0` ✓ | **OK** | Shannon (1948) H=−Σp ln p | n/a |
| S5 Flow-diversity utilization `fd/log2(n²)·100` (`publication_report.py:266-267`) | **Mixed-base risk** | Denominator `np.log2(n²)` is in **bits**, but `fd` (flow_diversity) is computed in **nats** (ln) by the Ulanowicz engine. Ratio mixes bases → utilization % understated by factor ln2≈0.693. Guard `h_max>0` ✓. | — | **MAJOR** | H_max for n² cells = log(n²) **in the same base as H**. Use `np.log(n**2)` (nats) to match `fd`, or convert fd to bits. | Yes — base must match (log₂ vs ln) |
| S6 A/Φ ratio `ascendency/overhead` (`publication_report.py:300`) | Yes | Guard `overhead>0` ✓ | — | **OK** | derived ratio; dimensionally consistent (both flow-nats) | n/a |

---

## Answers to the specific scrutiny items

- **Gini (S1) cross-implementation:** all THREE are the identical sorted-Gini and **agree with the canonical
  MAD Gini to floating-point precision** (verified: `[1,2,3,4,5]`→0.2667, random(50)→0.3119, etc.). No
  off-by-one, correct ascending sort, correct `(n+1)/n`. **OK across the board.**

- **Freeman centralization (N4):** denominator `(n−1)(n−2)` is the **undirected** normalizer applied to
  **directed** in/out degree. Correct directed normalizer is `(n−1)²`. **MAJOR — under-normalized, can exceed 1.**

- **Small-world σ (N9) / ω (N10):**
  - σ form `(C/Cr)/(L/Lr)` is the correct **Humphries** definition.
  - ω uses `C_random` where **Telford** requires `C_lattice` — deviation.
  - **Both inherit a broken `Lr`**: N11 computes `<k>` via `average_degree_connectivity().get(1,2)`, which is
    **not the mean degree**. This is the single most impactful network-science bug found. **MAJOR.**

- **Directed vs undirected on directed flows:** clustering (`network_analyzer` path), small-world, rich-club,
  ER baseline all run on `to_undirected()`; N4 uses undirected normalization on directed degrees; N7
  betweenness/closeness treat flow as distance (inverted). Assortativity (N12) is the correctly-directed one.

- **Magic numbers flagged:** pagerank α=0.85 (**standard, OK**); katz α=0.1 (**fixed, not λmax-adaptive —
  diverges on dense graphs, silent fallback**); rich-club k=90th percentile (**arbitrary**); path-redundancy
  cutoff=3 + 10×10 node sampling (**arbitrary + biased**); robustness num_simulations=10 (**low**);
  ER `<k>` default→2 (arbitrary fallback masking the N11 bug).

- **Statistical guards:** Gini non-negativity ensured by `flows>0`, ascending sort ✓. CoV mean>0 guarded
  in 2/3 sites; the third is safe because inputs are strictly positive. S5 has a **log-base mismatch** (nats
  vs bits).

---

## Severity roll-up

- **MAJOR:** N4 (Freeman denominator), N7 betweenness+closeness (weight-as-distance inversion),
  N9/N10/N11 (small-world `<k>` baseline; ω lattice term), N13 (unnormalized rich-club + arbitrary k),
  N16 (arbitrary cutoff + biased sampling), S5 (log-base mismatch).
- **MINOR:** N2′ (n² vs n(n−1) density inconsistency), N5 (in/out concat), N6 (directed vs undirected
  clustering inconsistency), N7 katz fixed-α, N14 (crude AUC, 10 sims), N15 (percolation-threshold labeling),
  N17 (var naming / non-ρ reciprocity).
- **OK:** N1, N2, N3, N6(ulanowicz path — directed Fagiolo), N7 eigenvector+pagerank, N8 modularity,
  N12 assortativity, N18 (proprietary), **S1 Gini (all 3)**, S2, S3, S4, S6.

*Validation only — no source code modified.*
