# Expert Mathematician — Adversarial Verification of the OASIS Formula-Validation Claims

**Mandate:** Independently derive/verify the mathematics behind claims M1–M6 raised by the prior
validation pass (`validation-SYNTHESIS.md` and the B / C&D / E&F detail files). Default posture:
a formula is NOT changed unless the math **and** the cited paper unambiguously demand it. Verdicts:
CONFIRM / REFUTE / PARTIALLY-CONFIRM / UNCERTAIN.

**Primary paper:** Ulanowicz, Goerner, Lietaer & Gomez (2009), *Quantifying sustainability: resilience,
efficiency and the return of information theory*, Ecological Complexity 6:27–36
(`_papers/Quantifying Sustainability Resilience Efficiency.pdf`) — "U2009".
**Also:** Zorach & Ulanowicz (2003), *Quantifying the Complexity of Flow Networks: How many roles are
there?*, Complexity 8(3):68–76 — "Z-U2003".

All derivations below were run in sympy/numpy. **No source code was modified.** The equations quoted
from the PDFs are transcribed from the extracted text (OCR artifacts like `(cid:2)` are the minus sign
`−` or superscripts; I note where that matters).

---

## Per-claim verdict table

| Claim | Verdict | The math (independently derived) | Paper eq. cited | Justifies a formula change? |
|-------|---------|----------------------------------|-----------------|-----------------------------|
| **M1** α-optimum = 0.4596 | **PARTIALLY-CONFIRM** | (a) d/dα[−α ln α] = −ln α − 1 = 0 ⇒ α = 1/e = 0.36788 ✓. (b) F = −e·α^β·ln(α^β); dF/dα = e·β·α^(β−1)·(−β ln α − 1) = 0 ⇒ **ln α = −1/β ⇒ α_opt = e^(−1/β)**. e^(−1/1.288) = **0.46006** (paper rounds "0.4596"). (c) β is **NOT independent**: U2009 derives α=0.4596 from the (c,n) window center, then *back-solves* β=1.288 from it (−1/ln(0.4596) = 1.2863 ≈ 1.288). So "0.4596 is the optimum" rests on the **empirical window center**, not on pure math. | U2009 Eq.16 `F = −[e/log(e)]·α^β·log(α^β)`, "F_max = 1 at α = e^(−1/β)"; §6 "geometric center of the window (c=1.25, n=3.25)… translate into α = 0.4596, from which we calculate a most propitious value of β = 1.288." | **YES**, for the α-*operating-target* formulas only (see ruling). The optimum of the paper's chosen fitness kernel is unambiguously e^(−1/β)=0.4596, not 1/e. |
| **M2** robustness form / base | **CONFIRM (form) + CONFIRM (base inconsistency)** | Code `R = −α·ln α` (natural log, `ulanowicz_calculator.py:549` `math.log`) is the **shape of Eq.15** (β=1, k=1). Its max is at 1/e, value 1/e = 0.36788 — a *different function* from Eq.17 `R = T··×F` (β-kernel, scaled by throughput). Base check: max(−α·ln α)=0.36788 **at** 1/e; max(−α·log₂ α)=**0.53074** (= log₂(e)/e) also **at** α=1/e — the 0.531 is the *value*, the *location* is still 1/e. Engine uses **ln** (max 0.368); `publication_report.py:125` states the ceiling 0.531 and "base-2 logarithms" — a real cross-module base mismatch. | U2009 Eq.15 `F = −k·α·log α`, Eq.17 `R = T···F`. | **NO formula change** to the engine math (the −α·ln α proxy is a legitimate dimensionless quantity). **YES doc/label fix**: R1 is mislabeled as Eq.17 robustness; R9's 0.531/base-2 text contradicts the ln engine. |
| **M3** window equivalence | **CONFIRM** | A ∈ [0.2C, 0.6C] ⇔ A/C = α ∈ [0.2, 0.6] for C>0 (divide by C). Verified identical over 10⁶ random (A,C) draws (`np.array_equal` = True). The engine compares A↔0.2C/0.6C (capacity units); the report compares α↔0.2/0.6 (dimensionless). **No unit bug in the band logic itself.** | (Algebraic; band [0.2,0.6] is secondary-literature heuristic, not verbatim U2009.) | **NO.** The band-comparison arithmetic is correct on both paths. (Whether [0.2,0.6] is the right band is a separate, non-mathematical question.) |
| **M4** effective connectivity inverted (Z3) | **CONFIRM** | Z-U2003 defines `C ≡ F/N` (flows/node, ≥1) and `R ≡ N/C ≡ N²/F ≡ F/C²`. Code Z3 = exp(½ Σ w·ln(T_ij²/(T_i T_j))). Numerically ln(C_code) = ln N − ln F exactly ⇒ **C_code = N/F = 1/(F/N)** the reciprocal. Random seeds 1/3/9: C_code = 0.295/0.193/0.295 while F/N = 3.39/5.18/3.39; `np.isclose(C_code, N/F)` = True every time. The paper's identities R=F/C² and R=N/C only close when C=F/N (Z7 silently substitutes F/N, masking Z3). | Z-U2003 p.72: "Let C ≡ F/N be the connectivity, measured in flows/node"; "R ≡ N/C ≡ N²/F ≡ F/C²." | **YES.** The reported "effective connectivity" should be **F/N**, not the exp(½Σ…) expression. Paper-unambiguous. |
| **M5** Finn Cycling Index | **CONFIRM (D1 & D2 both wrong) + PARTIAL on "~2×"** | Pure 4-node ring (permutation matrix): I−G is singular (ρ(G)=1). Adding boundary leak ε and taking ε→0: canonical FCI = Σ((s_ii−1)/s_ii·T_i)/TST → **1.0** (ε=0.5→0.20, 0.1→0.68, 0.01→0.96, 1e-4→0.9996). **D1** (self-loops + 2-cycles) = **0.0** on the ring (no self/2-cycles) — structural miss. **D2** (code: (ΣS−n)/ΣS, S=(I−T/TST)⁻¹) = **0.25** on the ring — crushed by scalar-TST normalization. On 5 random nets D2/canonical = 0.22, 0.32, 0.72, 0.83, **1.47** — **mostly an underestimate but NOT a clean 2×** (it over-estimates in one case). | Finn (1976); Ulanowicz (2004) §5: column-normalize g_ij=T_ij/T_j, S=(I−G)⁻¹, TSTc = Σ_i((s_ii−1)/s_ii)·T_i, **FCI = TSTc/TST**. | **YES** for D2 (replace with column-normalized Leontief + diagonal TSTc) and **YES** relabel D1 as a short-cycle proxy. The "~2× underestimate" characterization is **imprecise** — the error is variable in sign and magnitude; the correct statement is "D2 does not implement Finn and is systematically biased (usually low)." |
| **M6** stats / network math | **CONFIRM all three** | **(a) Freeman:** directed out-star, out-degrees [n−1,0,…,0], Σ(d*−d_i) = (n−1)². Code denom (n−1)(n−2) ⇒ C = (n−1)²/((n−1)(n−2)) = (n−1)/(n−2) **> 1** (n=5→1.333, n=10→1.125, n=20→1.056). Correct denom (n−1)² ⇒ exactly 1.0. **(b) ⟨k⟩:** 2m/n vs `average_degree_connectivity(G).get(1,2)` differ on ER(20,40) 4.0 vs 5.0, BA(30,3) 5.4 vs 2, WS(20,4) 4.0 vs 2 — the latter is avg-neighbour-degree of degree-1 nodes (or the default 2), not mean degree. **(c) Gini:** sorted-Gini = MAD-Gini to 1e-16 on [1..5]=0.2667, random(50)=0.3170, [3,1,4,1,5,9,2,6]=0.3669. | Freeman (1979) directed normalizer (n−1)²; Fronczak et al. (2004) Lr≈ln n/ln⟨k⟩, ⟨k⟩=2m/n; Sen (1973)/Damgaard–Weiner (2000) Gini. | **YES** for (a) directed normalizer (n−1)² and (b) ⟨k⟩=2m/n (both canonical, code demonstrably wrong/degenerate). **NO** for (c) — Gini is correct; the claim was that it's *right*, and it is. |

---

## Detailed derivations & where the prior pass OVERREACHED or was IMPRECISE

### M1 — the α-optimum (the headline claim)

**Mathematically established (not disputable):**
1. `−α ln α` is maximized at α = 1/e = 0.367879. (d/dα = −ln α − 1 = 0.)
2. The paper's fitness kernel `F = −[e/log(e)]·α^β·log(α^β)` (Eq.16) is maximized at
   **α = e^(−1/β)**. Derivation: with natural log, F = −e·β·α^β·ln α; dF/dα = −e·β·α^(β−1)(β ln α + 1);
   zero ⇒ ln α = −1/β ⇒ α = e^(−1/β). For β=1.288 this is **0.46006** (paper writes 0.4596; the small
   gap is rounding — β=1.288 is itself a 4-sig-fig round of −1/ln(0.4596)=1.2863).

**What is ASSERTED, not derived (the honest caveat the prior pass understates):**
The number **0.4596 is empirical, not a theorem.** U2009 obtains it from the *geometric center of the
window of vitality* (c=1.25, n=3.25), an **empirically observed cloud** of real ecosystem networks,
then back-computes β=1.288 to place the fitness maximum there. The chain is:

> empirical window center (c,n) → α = 0.4596 → β = 1.288 (via β = −1/ln α_opt).

So the relationship between 0.4596 and 1.288 is **internally circular by construction** — β is *chosen
so that* the max lands at 0.4596. This is not a defect (the paper is explicit: "the value of β fixes
the optimal value of α… There is no a priori reason to assume that the value of β is universal"), but
it means the claim "0.4596 is *the* mathematically-correct maximizer" is **only** correct *conditional
on accepting the paper's empirically-fitted β=1.288*. The prior pass presents 0.4596 as if it were a
hard mathematical constant on the same footing as 1/e; it is not — it is an **empirically-calibrated
target**. That nuance matters for the CLAUDE.md "no change without scientific support" rule: the
support here is "U2009's empirical fit," which the paper itself flags as provisional.

**Is 0.37 a genuine error?** It depends *entirely* on which quantity 0.37 is used for:
- As the **maximizer of the raw −α ln α proxy** (β=1) → 0.37 (≈1/e) is **correct**. Legitimate uses:
  the normalization ceiling `max_robustness = 1/e` (O11), the base-2 ceiling 0.531 (R9). U2009 §5 states
  the un-adjusted F "is still constrained to peak at α = (1/e)."
- As the **operating/sustainability α-target** (α-optimality score, regen center, distance-to-optimum)
  → 0.37 is **wrong**; U2009 §6 fixes that target at 0.4596 and *explicitly rejects 1/e*: "There is no
  more reason to force the balance between A and F to occur at [A/(A+F)]=(1/e)."

**Ruling on M1:** Changing the α-optimality *operating target* to 0.4596 **is justified**, but ONLY
for the formulas that use α-as-a-target: **F2** (`oasis_calculator.py:623-626`, O10 α-optimality),
**F3** (`ulanowicz_calculator.py:877-887`, regen center), **F4** (`report_intelligence.py:70`,
distance-to-optimum). It is **NOT** justified — and would be an *error* to change — for the two places
where 1/e correctly normalizes the −α·ln α proxy: **O11** `R/(1/e)` (`oasis_calculator.py:609`) and
**R9** the log₂(e)/e = 0.531 ceiling (`publication_report.py:125`). The prior pass gets this
distinction right (F6), but the SYNTHESIS headline ("Paper explicitly rejects 1/e", CRITICAL) risks
being read as "1/e is wrong everywhere," which is false.

### M2 — robustness form and base

- CONFIRM the two functions differ: engine `−α ln α` (Eq.15 shape, k=1) ≠ Eq.17 `R = T··×F` (β-kernel,
  throughput-scaled). Calling the engine value "robustness (Eq.17)" is a **labeling error**, not a
  numeric one — the −α ln α proxy is a valid dimensionless quantity.
- Base: engine uses `math.log` = **ln** ⇒ proxy max 1/e = 0.368. `publication_report.py:125` asserts
  max = log₂(e)/e = 0.531 and "base-2 logarithms" — that is correct for base-2 but **inconsistent with
  the ln engine**. R9's *value* 0.531 is right for its stated base; the codebase mixing ln (engine) and
  log₂ (report ceiling) is the real issue.
- **Important correction to a loose statement:** −α·log₂ α does **not** "peak at 0.531." It peaks
  **at α = 1/e** (same location as the ln version); 0.531 is the *height* at that peak. Any wording
  implying the *location* moves with the base is wrong. (The prior B-file states this correctly as
  "max of −α·log₂ α = 0.531"; just flagging that "peaks at 0.531" would be a category error.)

### M3 — window equivalence

Trivially CONFIRM. Dividing the inequality A ∈ [0.2C, 0.6C] by C>0 gives α ∈ [0.2, 0.6]. No unit bug in
the band comparison; engine and report paths are algebraically identical. (Separate, non-mathematical
question — whether [0.2,0.6] is the right heuristic band — is out of scope here and is correctly flagged
as "needs-judgment," since it is not verbatim in U2009.)

### M4 — effective connectivity inversion (Z3)

CONFIRM, paper-unambiguous. Z-U2003 p.72 literally: "Let **C ≡ F/N** be the connectivity, measured in
flows/node" and "**R ≡ N/C ≡ N²/F ≡ F/C²**." Code Z3 computes exp(½Σw·ln(T_ij²/(T_i T_j))), which equals
exp(ln N − ln F) = **N/F**, the reciprocal. Numerically C_code = N/F to machine precision on all seeds;
F/N (the paper quantity) is ≥1 while C_code <1. The three role identities only close with C=F/N (which
is exactly what Z7 silently substitutes). This is a genuine inversion; setting effective connectivity =
F/N is paper-backed. **One caveat:** the exp(½Σ…) expression *is* a real Zorach-Ulanowicz-family
quantity (it is the reciprocal-connectivity / "1/C"); the fix is to report F/N under the label
"connectivity," not to delete the expression as meaningless.

### M5 — Finn Cycling Index

- CONFIRM the canonical FCI → 1.0 for the pure ring (shown via the ε→0 boundary-leak limit; the exact
  permutation ring makes I−G singular, which is *why* a naive implementation fails and *why* boundary
  flows are part of the standard construction).
- CONFIRM D1 = 0 on the ring (misses all cycles ≥3) and D2 = 0.25 on the ring (scalar-TST
  normalization ⇒ (I−G)⁻¹≈I ⇒ cycling crushed).
- **IMPRECISE in the prior pass:** the "systematic ~2× underestimate" for D2. My 5-seed sample gives
  ratios D2/canonical of 0.22, 0.32, 0.72, 0.83, **1.47** — predominantly an underestimate but not a
  constant factor, and it *over*-estimates in at least one case. The defensible claim is "D2 does not
  implement the Finn/Leontief method and is biased (usually low, magnitude data-dependent)," **not** a
  clean 2×. The formula-change recommendation (adopt column-normalized Leontief + diagonal TSTc) stands
  regardless.

### M6 — statistics / network math

- **(a) Freeman:** CONFIRM. A directed out-star yields Σ(d*−d_i) = (n−1)², so with the code's
  (n−1)(n−2) denominator the centralization = (n−1)/(n−2) > 1 (1.333 at n=5). Freeman's directed
  normalizer (n−1)² gives exactly 1.0. Code can and does exceed 1 → paper-backed fix.
- **(b) ⟨k⟩:** CONFIRM. `2m/n` ≠ `average_degree_connectivity(G).get(1,2)` in every tested graph; the
  networkx call returns the average *neighbour* degree of degree-1 nodes (or the default 2), not mean
  degree. This corrupts Lr=ln n/ln⟨k⟩ and thus σ, ω, is_small_world. ⟨k⟩=2m/n is the canonical fix.
- **(c) Gini:** CONFIRM the code is CORRECT — sorted-Gini equals the mean-absolute-difference Gini to
  1e-16 on all test vectors. This claim asserted correctness, and it holds; **no change**.

---

## Crisp ruling requested by the mandate

**Is changing the α-optimality target to 0.4596 mathematically justified, and for which formulas?**

**YES — but scoped and with one honesty caveat.**

- Justified **only** for formulas that use α as an *operating/sustainability target*:
  **F2** (O10 α-optimality, `oasis_calculator.py:623-626`), **F3** (regenerative-capacity center,
  `ulanowicz_calculator.py:877-887`), **F4** (distance-to-optimum, `report_intelligence.py:70`).
  For these, 0.37/1/e is the wrong quantity and U2009 §6 unambiguously specifies 0.4596 (= e^(−1/β),
  β=1.288). The single place that *already* uses 0.4596 (R7, `ulanowicz_calculator.py:855-861`) is
  correct and must not be touched.
- **NOT** justified — changing it would be an error — where 1/e correctly normalizes the −α·ln α proxy:
  **O11** `R/(1/e)` (`oasis_calculator.py:609`) and the **R9** base-2 ceiling 0.531
  (`publication_report.py:125`). These are maxima of the *proxy*, not the operating target.
- **Honesty caveat (where the prior pass overreached):** 0.4596 is **empirically calibrated** (window
  center → back-solved β), not a closed-form theorem like 1/e. U2009 itself calls β provisional
  ("no a priori reason to assume β is universal"). So the correct framing for the CLAUDE.md rule is:
  "adopt U2009's empirically-fitted sustainability target 0.4596 for α-target uses," not "0.4596 is the
  mathematically-forced optimum." The math *forces* e^(−1/β); the *number* 0.4596 rests on the paper's
  empirical fit.

**Net:** M1 PARTIALLY-CONFIRM (target change justified for F2/F3/F4, with the empirical-calibration
caveat), M2 CONFIRM (label/base fix, not engine-math), M3 CONFIRM (no bug), M4 CONFIRM (F/N inversion
real), M5 CONFIRM (D1/D2 both wrong; "~2×" imprecise), M6 CONFIRM (Freeman (n−1)², ⟨k⟩=2m/n, Gini
correct).

*Adversarial verification only. No source code modified. No commit made.*
