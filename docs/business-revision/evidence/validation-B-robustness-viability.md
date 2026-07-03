# Validation B — Robustness & Window-of-Viability formulas

**Scope:** Family B (Robustness, Window of Viability, Fitness-for-Evolution, optimum
constants). Validation only — **no source code was modified.**

**Primary source:** Ulanowicz, R.E., Goerner, S.J., Lietaer, B., Gomez, R. (2009).
*Quantifying sustainability: Resilience, efficiency and the return of information
theory.* Ecological Complexity 6, 27–36. (`_papers/Quantifying Sustainability
Resilience Efficiency.pdf`) — hereafter **U2009**.
**Supporting:** Ulanowicz (2009) *Some steps toward a central theory of ecosystem
dynamics* (`_papers/Some steps toward a central theory of ecosystem dynamics.pdf`).

All page/line references below are to the extracted text of U2009. The relevant
equations were read verbatim from §5 "The survival of the most robust" and §6
"Vectors to sustainability".

---

## 0. What the paper actually says (verbatim anchors)

The paper builds robustness in three explicit steps:

- **Eq (15) — "fitness for evolution":**
  > "we choose the Boltzmann formulation, –k·log(α) ... the product of α and ~α ...
  > F = −k·α·log(α)"
  > "It is 0 for α = 1 and approaches the limit of 0 as α → 0. One can normalize this
  > function by choosing k = e·log(e) ... such that 1 > F > 0."
  > "**F is still constrained to peak at α = (1/e).** There is no more reason to force
  > the balance between A and F to occur at [A/(A+F)] = (1/e) than it was to mandate
  > that it happen when A = F."

- **Eq (16) — generalized/normalized fitness (with adjustable exponent β):**
  > "F = −k·α^β·log(α^β). This function can be normalized by choosing k = e/log(e),
  > so that F_max = 1 at **α = e^(−1/β)**, where β can be any positive real number."
  >
  > **F = −[e/log(e)]·α^β·log(α^β)  … (16)**

- **Eq (17) — ROBUSTNESS itself:**
  > "the robustness, R, of the system becomes **R = T·· · F   (17)**"
  (T·· = total system throughput; F = the dimensionless Eq-16 fitness fraction.)

- **The optimum (§6, verbatim):**
  > "We therefore choose the geometric center of the window (c = 1.25 and n = 3.25)
  > as the best possible configuration for sustainability ... **These values translate
  > into α = 0.4596, from which we calculate a most propitious value of β = 1.288.**"
  > "When α < 0.4596, the system likely requires more coherence ... Conversely, when
  > α > 0.4596, the system might be over-developed."

- **The "window of vitality" is defined on the (c, n) axes**, not on α:
  > "they plotted the networks ... on the transformed axes c and n ... the empirical
  > networks all cluster within a rectangle bounded roughly in the vertical direction
  > by c = 1 and c ≈ 3.01 and horizontally by n = 2 and n ≈ 4.5."
  c = effective link density; n = effective number of roles/trophic levels.
  **The paper gives NO explicit "α ∈ [0.2, 0.6]" window.** It gives a single
  optimal α = 0.4596 (the window *center*).

Numerical checks (run, source untouched):
- d/dα[−α·ln α] = 0 → α = 1/e = 0.367879… ✓
- e^(−1/1.288) = 0.45996 ≈ 0.4596 (paper) ✓
- R(0.37) = 0.367873 vs R(1/e) = 0.367879 — 0.37 is a valid rounding of 1/e *for the
  Eq-15 maximizer*. ✓
- max of −α·log₂(α) = log₂(e)/e = 0.530738 ✓ (matches R9)
- A ∈ [0.2C, 0.6C] ⇔ α = A/C ∈ [0.2, 0.6] — verified True over 10⁶ random draws. ✓

---

## 1. Per-formula table

| ID | Location | Code | Paper form | Severity | Correct form / constant (citation) | Paper-backed fix? |
|----|----------|------|-----------|----------|-----------------------------------|-------------------|
| **R1** | `ulanowicz_calculator.py:548-549`; `vectorized_metrics.py:445-448,480-483` | `R = −α·ln(α)`, α = A/C | Eq (17): **R = T···F**, F = Eq-16 (dimensionless, β-adjustable). Eq (15) −k·α·log α is the *un-adjusted, un-scaled* "fitness for evolution", not "robustness". | **MAJOR (semantic)** | The code's `R = −α·ln α` = the *shape* of Eq (15) with k=1 (unnormalized, natural-log). It is a legitimate, widely-used **relative/dimensionless robustness proxy** (the F-fraction shape) and is internally consistent, but it is **not** the paper's Eq-17 robustness (which is scaled by T·· and uses the β=1.288 Eq-16 kernel). Labeling matters: this is "relative fitness/robustness shape", peaking at 1/e — NOT R=T···F. | **Needs-judgment.** Do not "fix" the math; the dimensionless proxy is defensible. Recommend a comment/label correction only (no formula change without deciding whether the product proxy or the paper's β-kernel is the intended metric). |
| **R2** | `ulanowicz_calculator.py:379-380` | lower = 0.2·C, upper = 0.6·C (capacity units) | Window defined on (c,n) axes; center → α=0.4596. No explicit 0.2/0.6 α-bounds in U2009. | **MINOR** | 0.2/0.6 are a **secondary-literature heuristic** approximation of the α-window, not verbatim U2009. They straddle the paper's α=0.4596 optimum asymmetrically (0.4596 sits at 0.65 of the way up the band), which is broadly consistent with the empirical scatter. Algebra `0.2C..0.6C` vs α is correct (see R3). | **Needs-judgment.** Bounds are approximate but not contradicted by the paper. Keep, but cite as heuristic (Ulanowicz's popularizations) rather than "Eq. X of U2009". |
| **R3** | `ulanowicz_calculator.py:428` | `is_viable = lower ≤ A ≤ upper`, bounds = 0.2C/0.6C | A ∈ [0.2C, 0.6C] ⇔ α ∈ [0.2,0.6] | **OK** | Comparing **A (capacity units) to 0.2C/0.6C (capacity units)** is dimensionally consistent and algebraically identical to α∈[0.2,0.6]. Verified. | N/A — correct. |
| **R4** | `report_intelligence.py:13-14`; `oasis_calculator.py:920,928` | α band [0.2, 0.6], dimensionless α compared to 0.2/0.6 | same heuristic band on α | **OK** | Here 0.2/0.6 are compared to **dimensionless α** — correct scale. Consistent with R3's engine path (both encode α∈[0.2,0.6]). | N/A — consistent with R2/R3. |
| **R5** | `report_intelligence.py:15`; `oasis_calculator.py:609` | robustness optimum = 1/e ≈ 0.3679 | Eq (15) peaks at α = 1/e; **BUT paper rejects 1/e as the sustainability optimum** (uses 0.4596). | **MAJOR** | 1/e = 0.3679 is correct **only** as the maximizer of the un-adjusted Eq-15 shape (which the code's R1 uses). As the *normalization ceiling* for that specific proxy (`max_robustness = 1/e`, oasis:609) it is **internally correct**. But if presented as "the sustainability/window optimum for α", it is **wrong** — the paper's optimum is 0.4596. | **PAPER-BACKED distinction.** 1/e is OK as the max of `−α·ln α`; it is NOT the paper's α-optimum. Keep 1/e only where it normalizes the R1 proxy; do NOT use it as the α-target. |
| **R6** | `oasis_calculator.py:623,880`; report files | "optimal α ≈ 0.37" used as the α-target (alpha_optimality, regen) | Paper's α-optimum = **0.4596** | **CRITICAL** | Using **0.37 as the target value of α** (distance-to-optimum, alpha_optimality score, regen center) is **scientifically incorrect** per U2009 §6. The paper explicitly says the propitious α is 0.4596 and explicitly argues *against* forcing the balance at 1/e. 0.37 is the max of the *robustness proxy R(α)*, which is a different quantity from *the optimal operating α*. Conflating "α that maximizes the −α ln α proxy" with "the sustainable-optimum α" is the core Issue-3 bug. | **PAPER-BACKED FIX:** the α-optimality target and regen center should be **α_opt = 0.4596** (U2009: "These values translate into α = 0.4596 ... most propitious"), not 0.37. |
| **R7** | `ulanowicz_calculator.py:855-861`; `oasis_calculator.py:282-288` | `F = −e·α^β·ln(α^β)`, β=1.288, opt α=e^(−1/β)≈0.4596 | Eq (16) exactly (with log(e)=1 in nat-log form ⇒ k=e/log(e)=e). | **OK** | Matches Eq (16) verbatim; β=1.288 and optimum 0.4596 both cited directly from U2009. Natural-log simplification (log(e)=1 ⇒ k=e) is algebraically correct. | N/A — **PAPER-BACKED and correct.** This is the *only* place the paper's true optimum (0.4596) is honored. |
| **R8** | `ulanowicz_calculator.py:877-887` | `regen = R·(1 − |α − 0.37|)` (uses `network_efficiency` for the ratio, and 0.37) | proprietary blend, no paper source | **CRITICAL (two defects)** | (a) Uses **0.37 instead of 0.4596** as the optimum → same Issue-3 error as R6. (b) `current_ratio = calculate_network_efficiency()` (line 881) is compared to an **α-optimum** — network efficiency is NOT α (A/C); this mixes two different ratios into the distance term. | Blend itself is proprietary (needs-judgment), but **the 0.37 constant is a PAPER-BACKED fix → 0.4596**, and the α-vs-efficiency mismatch (881 vs 884) is a genuine variable-confusion bug worth flagging. |
| **R9** | `publication_report.py:125` | `0 ≤ R ≤ log2(e)/e ≈ 0.531` | max of −α·log₂(α) = log₂(e)/e | **OK** | 0.530738 confirmed numerically. Consistent with a **base-2** robustness proxy. Note: R1/R5 use **natural log** (max = 1/e = 0.368); publication_report claims **base-2** (max 0.531). The stated max is correct for base-2 but **inconsistent with the natural-log engine** (R1). | **Needs-judgment:** the ceiling is mathematically right for its base but the codebase mixes ln and log₂ across modules — pick one base. |
| **R10** | `report_intelligence.py:70` | `distance_to_optimum = |α − 0.3679|` (ROBUSTNESS_OPTIMUM = 1/e) | should be distance to the **operating** optimum α=0.4596 | **MAJOR** | If this "distance to optimum" is meant as distance to the *sustainable* α, it must use 0.4596, not 1/e. As "distance to the robustness-proxy peak" it's fine but is then a different, easily-misread quantity. | **PAPER-BACKED:** for a sustainability target use 0.4596; keep 1/e only if explicitly labeled "distance to R-proxy peak". |

---

## 2. Resolving 1/e (0.3679) vs 0.37 vs 0.4596 — the paper's own words

There are **three distinct quantities**; the codebase conflates them. Untangled:

1. **α = 1/e ≈ 0.3679 (and its rounding 0.37).** This is the maximizer of the
   *un-adjusted, β=1* fitness shape **F = −k·α·log(α) (Eq 15)** — i.e. exactly the
   `−α·ln(α)` used by R1. The paper introduces this, then **explicitly rejects it as
   the sustainability optimum**:
   > "F is still constrained to peak at α = (1/e). **There is no more reason to force
   > the balance ... to occur at (1/e)** than it was to mandate that it happen when
   > A = F."
   ⇒ **1/e is ONLY the peak of the raw −α·ln α curve.** It is scientifically correct
   for R5's normalization ceiling of that specific proxy and for R9 (in base-2:
   log₂(e)/e). It is **NOT** the optimal operating α.

2. **α = 0.4596.** This is the paper's actual **"most propitious" / optimal α**,
   derived from the geometric center of the empirical window of vitality:
   > "the geometric center of the window (c = 1.25 and n = 3.25) ... translate into
   > **α = 0.4596**, from which we calculate a most propitious value of β = 1.288."
   ⇒ **0.4596 is the scientifically-correct target for "how organized should the
   system be" (α-optimality, distance-to-optimum, regenerative-capacity center).**

3. **β = 1.288.** The shape parameter that *moves* the fitness maximum from 1/e to
   0.4596 via **α_opt = e^(−1/β)** (Eq 16). Correctly implemented in R7.

**Verdict on each consumer:**

| Consumer | Constant used | Correct? |
|----------|--------------|----------|
| R5 `max_robustness = 1/e` (normalizes the −α·ln α proxy) | 1/e | **OK** — it is the true max of *that* proxy. |
| R9 ceiling `log2(e)/e ≈ 0.531` | 1/e in base-2 | **OK** — max of base-2 proxy. |
| **R6 alpha_optimality target = 0.37** | 0.37 | **WRONG → 0.4596.** MAJOR/CRITICAL: this is the sustainability α-target, which the paper fixes at 0.4596. |
| **R8 regen center = 0.37** | 0.37 | **WRONG → 0.4596.** Same error. |
| **R10 distance_to_optimum uses 1/e** | 1/e | **WRONG if it means sustainability distance → 0.4596.** |
| R7 fitness opt = 0.4596 (β=1.288) | 0.4596 | **CORRECT.** |

So: **0.37/1/e is defensible only as the peak of the −α·ln α robustness *proxy*.
Using it as the *optimal α operating point* (R6, R8, R10-as-sustainability) is a
MAJOR-to-CRITICAL scientific error** — the paper's operating optimum is unambiguously
**α = 0.4596**.

---

## 3. Issue 2 — unit consistency (A vs 0.2C..0.6C  vs  α vs 0.2..0.6)

**Engine path** (`ulanowicz_calculator.py`):
- `calculate_window_of_viability()` returns `(0.2·C, 0.6·C)` — **capacity units**.
- `is_viable = lower ≤ A ≤ upper` (line 428) compares **A (capacity units)** to those
  bounds. **Dimensionally consistent.** ✓
- Exported `viability_lower_bound / viability_upper_bound` (426-427) are **capacity
  units** — must never be compared to a dimensionless α downstream.

**Report path** (`report_intelligence.py`, `oasis_calculator.py:920,928`):
- Compares dimensionless **α** to the raw **0.2 / 0.6** constants. **Also consistent**
  (α-scale on both sides). ✓

**Equivalence proven:** A ∈ [0.2C, 0.6C] ⇔ α = A/C ∈ [0.2, 0.6] (verified over 10⁶
random draws). So the two paths are **mathematically equivalent and not mixed**:
the engine compares A↔0.2C/0.6C; the report compares α↔0.2/0.6. **No real unit bug
found in the viability-bound comparison itself.**

**One genuine variable-mismatch to flag (not a unit bug, a wrong-variable bug):**
R8 `calculate_regenerative_capacity` (line 881) sets
`current_ratio = calculate_network_efficiency()` and then compares it to the
**α-optimum 0.37** (line 884). Network efficiency is **not** α = A/C. This mixes a
different ratio into an α-distance term. This is a substantive correctness concern
independent of the 0.37 issue.

**Consumer check of exported capacity-unit bounds:** searched — the exported
`viability_lower_bound/upper_bound` are surfaced in reports as raw numbers alongside
A (same units), so no capacity-vs-α cross-comparison was found. If any UI later plots
these bounds on an α (0–1) axis, that WOULD be a bug — worth a guard/label, but not
currently triggered in the validated paths.

---

## 4. Band-correctness — are [0.2, 0.6] paper-backed?

**Not verbatim.** U2009 defines the *window of vitality* on the **(c, n)** axes
(c ∈ [1, 3.01], n ∈ [2, 4.5]) and reports a **single optimal α = 0.4596** (window
center). It does **not** publish "α ∈ [0.2, 0.6]" as the viability band. The 0.2/0.6
figures come from **Ulanowicz's later popularizations / secondary literature** as a
rough α-range for the observed scatter, and are commonly cited but approximate.

- **Consistency check:** 0.4596 lies inside [0.2, 0.6] (about 65% up the band) — the
  band is asymmetric about the optimum, tilted toward the efficient/upper side, which
  is qualitatively consistent with the empirical cloud in Fig. 4. Not contradicted by
  the paper.
- **Recommendation (needs-judgment):** keep [0.2, 0.6] as a documented heuristic, but
  do **not** cite it as "Eq. X / stated bounds of U2009". The paper-backed anchor is
  the **optimum α = 0.4596**, and any "distance to optimum / α-optimality" logic
  should key off 0.4596, not the band edges and not 0.37.

---

## 5. Severity roll-up

- **CRITICAL:** R6 (0.37 as α-target), R8 (0.37 + α-vs-efficiency mismatch).
- **MAJOR:** R1 (proxy mislabeled as Eq-17 robustness), R5 (1/e OK as ceiling, wrong
  if used as α-target), R10 (1/e distance if meant as sustainability distance).
- **MINOR:** R2 (0.2/0.6 heuristic, not verbatim).
- **OK / correct:** R3, R4, R7 (the one true 0.4596 implementation), R9 (base-2 max,
  modulo ln/log₂ base inconsistency across modules).

**Paper-backed fixes (safe to make, cited):**
- R6 / R8 / R10 α-target: **0.37 → 0.4596** (U2009 §6, "These values translate into
  α = 0.4596 ... most propitious").
- Label R1/R5 as the *un-adjusted fitness (−α·ln α) proxy*, distinct from Eq-17
  robustness and from the α=0.4596 operating optimum.

**Needs-judgment (do NOT change formula without a decision):**
- Whether the engine's canonical robustness should become the paper's Eq-17
  R = T···F (β=1.288) or remain the dimensionless −α·ln α proxy.
- ln vs log₂ base unification (R1 uses ln → max 1/e; R9 claims log₂ → max 0.531).
- R2 band [0.2,0.6] retention as heuristic.
- R8 network-efficiency-vs-α variable mismatch.
