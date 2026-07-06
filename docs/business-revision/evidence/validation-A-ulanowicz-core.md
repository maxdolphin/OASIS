# Validation A — Core Ulanowicz Information-Theoretic Formulas

**Scope:** Family A (U1–U11) from `docs/business-revision/evidence/formula-inventory.md`.
**Primary source:** Ulanowicz, Goerner, Lietaer & Gomez (2009), *"Quantifying sustainability:
Resilience, efficiency and the return of information theory"*, Ecological Complexity 6, 27–36.
DOI: 10.1016/j.ecocom.2008.10.005. (`_papers/Quantifying Sustainability Resilience Efficiency.pdf`)
**Cross-check:** Ulanowicz, *"Some steps toward a central theory of ecosystem dynamics"*
(`_papers/Some steps toward a central theory of ecosystem dynamics.pdf`).
**Code validated:** `src/ulanowicz_calculator.py` (loop reference) and `src/vectorized_metrics.py` (numpy).
**Method:** Read code at the cited lines, transcribed the paper's equations verbatim, compared term
by term, and ran a numeric spot-check (loop vs vectorized vs independent hand computation) plus edge
cases. **No source code was modified.**

---

## Paper equations (verbatim, transcribed from the PDF)

Marginal / total conventions (Ulanowicz-2009, p.29, footnote 2 and text):
- A dot replacing an index means summation over that index.
- `T_i.` (= Σ_j T_ij) = "everything leaving i" → **row sum = output throughput**.
- `T_.j` (= Σ_i T_ij) = "everything entering j" → **column sum = input throughput**.
- `T..` = Σ_{i,j} T_ij = **Total System Throughput (TST)**.

Estimators (Eq. 9): `p_ij ≈ T_ij/T..`, `p_i. ≈ T_i./T..`, `p_.j ≈ T_.j/T..`.

Scaled measures (all scaled by `k = T..`, p.29):

- **Eq. (7) / (5) AMI (X):**  `X = k · Σ_ij p_ij · log( p_ij / (p_i.·p_.j) )`
  → with `k=T..` and the estimators, `X = Σ_ij T_ij · log( T_ij·T.. / (T_i.·T_.j) ) / T..`.
- **Eq. (11) Development Capacity (C):**  `C = T..·H = −Σ_ij T_ij · log( T_ij / T.. )`.
- **Eq. (12) Ascendency (A):**  `A = T..·X = Σ_ij T_ij · log( T_ij·T.. / (T_i.·T_.j) )`.
- **Eq. (13) Reserve/Overhead (Φ):**  `Φ = T..·C_cond = −Σ_ij T_ij · log( T_ij² / (T_i.·T_.j) )`.
- **Eq. (14) Fundamental identity:**  `C = A + Φ`.
- **Eq. (3) Diversity H (per-unit):**  `H = −k Σ p_i log(p_i)`; flow form `H = −Σ (T_ij/T..)·log(T_ij/T..)`.
- **Eq. (8):**  `H = X + C_cond` (per-unit) ⇒ conditional entropy `= H − AMI`.
- **Log base (p.29):** "the only dimensions that H, X and C carry are those of the base of the
  logarithm… if the base is 2, the variables are all measured in **bits**." The paper reports all
  ecosystem numbers in **bits (log2)**.

---

## Per-formula results

| ID | Quantity | Code matches paper? | Log-base note | loop = vectorized? | Severity | Paper citation (eq.) | Recommended fix (paper-backed?) |
|----|----------|--------------------|---------------|--------------------|----------|----------------------|---------------------------------|
| U1 | TST = Σ Tij | **YES** — `np.sum(flow_matrix)` = T.. | base-invariant (pure sum) | YES (both `np.sum`) | **OK** | Eq. 9 / p.29 "T.." | — |
| U2 | AMI = Σ(Tij·T/(Ti·Tj))·log(...)/T | **YES** — `Σ Tij·log(Tij·tst/(out_i·in_j)) / tst` | **`math.log`/`np.log` = natural (nats)**; paper Eq.5/7 in bits. Value is base-dependent | YES | **MINOR** (base) | Eq. 5, Eq. 7 (X) | Optional: document units = nats, or ÷ln2 for bits comparisons. Base is a convention — **needs-judgment**, both bases valid |
| U3 | A = Σ Tij·log(Tij·T/(Ti·Tj)) (no ÷T) | **YES** — same as U2 without `/tst` | natural log; paper Eq.12 in bits | YES | **MINOR** (base) | **Eq. 12** | As U2 — **needs-judgment** |
| U4 | C = −Σ Tij·log(Tij/T) | **YES** — `−Σ Tij·log(Tij/tst)` | natural log; paper Eq.11 in bits | YES | **MINOR** (base) | **Eq. 11** | As U2 — **needs-judgment** |
| U5 | Φ = C − A | **YES** — `dev_capacity − ascendency` | inherits base of C,A (consistent) | YES | **OK** | Eq. 13 / 14 | — (see note below on direct Eq.13 form) |
| U6 | α = A/C | **YES** — `ascendency/dev_capacity`, guarded `C>0` | **base-invariant** (ratio) | YES | **OK** | p.29–30 (A/C) | — |
| U7 | H = −Σ(Tij/T)·log(Tij/T) | **YES** — `p=Tij/tst; −Σ p·log(p)` | natural log; paper Eq.3 in bits | YES | **MINOR** (base) | Eq. 3 | As U2 — **needs-judgment** |
| U8 | Hc = H − AMI | **YES** — `flow_diversity − ami`, `max(0,·)` | both natural (consistent) | loop-only (no vec fn); consistent | **OK** | Eq. 8 (H = X + C) | `max(0,·)` clamp is defensive only; H ≥ AMI is guaranteed by Eq. 6 |
| U9 | SI = log(n²) − H | code = `math.log(n²) − H` | **BOTH natural** — internally consistent | loop-only | **OK*** | derived (not a named paper eq.) | *Not from Ulanowicz-2009; if compared to any published log2 figure, mixing would break — but here both terms are natural, so OK |
| U10 | Φ/C (overhead ratio) | **YES** — `overhead/dev_capacity` | base-invariant (ratio) | YES | **OK** | Eq. 13/14 (Φ), ratio derived | — |
| U11 | identity check C = A + Φ | **YES** — `relative_error < 0.001` | base-invariant | YES (Φ defined as C−A ⇒ exact) | **OK** | **Eq. 14** | Tolerance is sound; because Φ := C−A the identity is exact (diff = 0), so the check can never fail — trivially true but harmless |

### Marginal-sum check (common-bug audit)
- `output_throughput = np.sum(flow_matrix, axis=1)` → **row sum = T_i.** ✅ correct (leaving i).
- `input_throughput = np.sum(flow_matrix, axis=0)` → **col sum = T_.j** ✅ correct (entering j).
- In U2/U3 the denominator is `output_i * input_j` = `T_i. · T_.j`, matching Eqs. 5/12 exactly.
  **No input/output swap, no wrong total.** The vectorized path uses `np.outer(row_sums, col_sums)`
  = `T_i.·T_.j` in the identical `[i,j]` positions. ✅

### Log-base summary
The code uses `math.log` (loop) and `np.log` (vectorized) — **both natural log (nats)**. Ulanowicz-2009
reports all magnitudes in **bits (log2)**. Consequences:
- **Base-invariant (unaffected):** U1 (TST), U6 (α = A/C), U10 (Φ/C), U11 (identity), and R1 robustness
  (a ratio-of-logs form). These match the paper regardless of base.
- **Base-dependent (magnitude differs by factor ln2 ≈ 0.6931):** U2 AMI, U3 A, U4 C, U5 Φ, U7 H, U8 Hc.
  A natural-log A is `A_bits · ln2`. This is **not an error** (the paper explicitly says the base is a
  free convention, p.29) but it **matters for any direct comparison to published bit-valued figures**
  (e.g. the stored reference values in group I / `published_metrics_db.py`, which are log2). The
  validation layer already converts via `x/ln2` (inventory P7) — confirm that path is used wherever
  code values are checked against the log2 reference numbers.
- **U9 SI:** `log(n²)` and `H` are BOTH natural in the code, so the subtraction is internally
  consistent. The only risk (flagged in the task) would be mixing `log2(n²)` with a natural-log H, or
  vice-versa — **that does not occur here.** SI itself is a derived quantity, not a named Ulanowicz-2009
  equation; treat as OK for internal use but note it is not paper-anchored.

### Loop vs vectorized agreement
Verified **identical to machine precision** for every metric (see spot-check below). The expressions are
term-for-term the same; the vectorized version merely replaces the nested loop with `np.outer` +
masking. The calculator's auto-vectorized path (`use_vectorized=True`) returns the same values as the
loop path.

### Edge / guard handling
- `Tij = 0` terms are **skipped** in every sum (loop: `if flow_ij > 0`; vectorized: `np.where(mask,…,0)`),
  which is the correct entropy convention `0·log0 ≡ 0`. ✅
- `TST = 0` short-circuits to 0 in AMI/A/C/H (both paths). ✅
- Empty row/col (a node with no in- or out-flow): those `Tij` are 0 and skipped; verified numerically. ✅
- Single node / all-zeros: TST=0 → all metrics 0, α=0 (guarded `C>0`), `SI = log(1) − 0 = 0`. No
  divide-by-zero, no `log(0)`. ✅

---

## Numeric spot-check

Test matrix (4×4 directed, includes zeros to exercise the `log(0)` guard; a strict 3×4 is impossible
because the calculator requires a square matrix — a 4×4 with a rank-deficient/zero node covers the same
guards):

```
F = [[0, 5, 2, 0],
     [0, 0, 3, 4],
     [1, 0, 0, 6],
     [2, 0, 0, 0]]
TST = 23.0    Ti (row/out) = [7,7,7,2]    Tj (col/in) = [3,5,5,10]
```

| Metric | Loop | Vectorized | Independent hand | Match |
|--------|------|-----------|------------------|-------|
| TST | 23.0 | 23.0 | 23.0 | ✅ |
| AMI | 0.776576 | 0.776576 | 0.776576 | ✅ |
| A (ascendency) | 17.861242 | 17.861242 | 17.861242 | ✅ |
| C (dev. capacity) | 41.705018 | 41.705018 | 41.705018 | ✅ |
| Φ (reserve) | 23.843776 | 23.843776 | — (C−A) | ✅ |
| α = A/C | 0.428276 | 0.428276 | — | ✅ |
| H (flow diversity) | 1.813262 | 1.813262 | 1.813262 | ✅ |
| Hc = H − AMI | 1.036686 | — | 1.036686 | ✅ |
| SI = log(16) − H | 0.959327 | — | log16=2.772589 | ✅ |
| Φ/C (overhead ratio) | 0.571724 | — | — | ✅ |

**Identity (U11):** `C = 41.705018`, `A + Φ = 41.705018`, `|diff| = 0.00e+00` → holds exactly
(because Φ is *defined* as C − A, the identity is algebraically exact, not merely within tolerance).
**`AMI·TST = A`:** `0.776576 × 23 = 17.861242 = A` → confirms the report-layer identity `A = TST × AMI`
(inventory note on F-block) is correct: U3 (A) is exactly U2 (AMI) × TST.

Edge cases: single-node zero matrix → all 0, α=0, SI=0; 3×3 all-zeros → all 0; node with no
in/out-flow → A and C computed correctly from the two live edges (loop 4.780357 = vectorized 4.780357).

---

## Summary of findings

**Formulas OK/correct vs paper: 11 of 11.** Every U1–U11 expression matches the Ulanowicz-2009 equation
it claims (U2→Eq.5/7, U3→Eq.12, U4→Eq.11, U5→Eq.13/14, U11→Eq.14), with correct marginal-sum
conventions (row=output=T_i., col=input=T_.j — **no swap**), correct denominator positions, correct
`0·log0` skipping, and correct TST/zero guards.

**CRITICAL issues: NONE.** No headline number is wrong.

**MAJOR issues: NONE.** The only cross-cutting item is the **log base**:

- **Base note (MINOR, needs-judgment, not an error):** the engine computes A, C, AMI, Φ, H, Hc in
  **natural log (nats)** while Ulanowicz-2009 reports them in **bits (log2)**. Per the paper (p.29) the
  base is an explicit free convention, so this is **not a formula error** and does **not** affect the
  base-invariant headline metrics α, Φ/C, robustness, or the C=A+Φ identity. It **only** matters when a
  nat-valued magnitude is compared directly to a published log2 figure. Recommendation (documentation,
  not a formula change): label the units of A/C/AMI/Φ/H as *nats*, and ensure the ×(1/ln2) conversion
  is applied wherever these are validated against the stored log2 reference values (group I). This is a
  **labeling/comparison** concern, not a correctness defect — **not paper-mandated to change the base**.

**Loop vs vectorized:** **agree exactly** (machine precision) across all metrics and via the
calculator's auto-vectorized path.

**Minor notes (no action required):**
- U8 `max(0, H−AMI)`: defensive clamp; H ≥ AMI is guaranteed by Eq. 6, so the clamp never fires.
- U9 SI (`log(n²)−H`) is a derived OASIS quantity, not a named Ulanowicz-2009 equation; internally
  base-consistent (both natural), fine for internal use, but should not be presented as a paper metric.
- U5/U11: Φ is implemented as C−A (Eq. 14) rather than the direct Eq. 13 sum; the two are mathematically
  identical and C−A is the more numerically stable choice. The identity check (U11) is therefore always
  exactly satisfied — sound but tautological.

**File:** `docs/business-revision/evidence/validation-A-ulanowicz-core.md`
