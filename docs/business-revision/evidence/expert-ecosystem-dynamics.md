# Adversarial Expert Review — Ecosystem-Dynamics / Ascendency Theory

**Reviewer role:** Theoretical-ecology / Ulanowicz ascendency-theory expert, brought in to
**refute** (not rubber-stamp) the prior formula-validation pass's *interpretive* claims about the
ecology. Verdicts are grounded in direct quotes from the papers in `_papers/`. No source code was
modified.

**Papers read (verbatim quotes below):**
- **U2009** — Ulanowicz, Goerner, Lietaer, Gomez (2009), *Quantifying sustainability: resilience,
  efficiency and the return of information theory*, Ecological Complexity 6:27–36.
  (`_papers/Quantifying Sustainability Resilience Efficiency.pdf`)
- **DUAL** — Ulanowicz (2009), *The dual nature of ecosystem dynamics*, Ecological Modelling
  220:1886–1892. (`_papers/Dual Nature of Ecosystem Dynamics.pdf`) — **same author, same year.**
- **FATH2019** — Fath, Fiscus, Goerner, Berea, Ulanowicz (2019), *Measuring regenerative economics:
  10 principles and measures undergirding systemic economic health*, Global Transitions 1:15–27.
  (`_papers/Measuring regenerative economics...pdf`) — **the paper that explicitly applies this to
  economics/organizations.**
- **ZU2003** — Zorach & Ulanowicz (2003), *Quantifying the complexity of flow networks: How many
  roles are there?*, Complexity 8(3):68–76. (`_papers/Quantifying the Complexity of Flow Networks-
  How many roles are there?.pdf`) — the actual source of the (c, n) window.
- **PROC** — Ulanowicz, *Process Ecology: A Transactional Worldview*.

**Bottom line up front:** The prior pass is **half right and half overstated**. It correctly
identified that U2009 §6 states a propitious α = 0.4596, and correctly flagged the codebase's
conflation of three different "optimal-α" constants. **But its central interpretive claim — that
ecosystem theory "explicitly rejects 1/e as the sustainability optimum" and that 0.4596 is THE
operating optimum — is not what the corpus says taken as a whole.** A second Ulanowicz 2009 paper
(DUAL) and the economics-facing FATH2019 paper both treat **α = 1/e as the natural sustainability
optimum / attractor**, and FATH2019 uses the very `−α·log α` form the prior pass called a
"mislabeled proxy" as *the* robustness measure for economies. The prior pass read one hedged
sentence in U2009 as a doctrine the author himself contradicts elsewhere.

---

## E1 — Is 0.4596 "THE optimum," and does the paper "explicitly reject 1/e"?

**Verdict: PARTIALLY-CONFIRM the arithmetic, REFUTE the strong interpretation.**

### What U2009 actually says (the quote the prior pass relied on — accurate)

U2009 §5, verbatim:

> "One can normalize this function by choosing k = e log(e) … such that 1 > F > 0. This does not
> solve our second problem, however, as F is **still constrained to peak at a = (1/e). There is no
> more reason to force the balance between A and F to occur at [A/(A + F)] = (1/e)** than it was to
> mandate that it happen when A = F. Clearly, the location of the optimum could be the consequence
> of (as yet) unknown dynamical factors, rather than one of mathematical convenience."

And U2009 §5 end:

> "We therefore choose the **geometric center of the window (c = 1.25 and n = 3.25)** as the best
> possible configuration for sustainability under the information currently available. These values
> translate into **a = 0.4596**, from which we calculate a most propitious value of **b = 1.288**."

So the 0.4596 value **is** in U2009, and it **is** derived as a window-center, and U2009 **does**
argue against *hard-wiring* the optimum at 1/e on grounds of mathematical convenience. Numerically
verified: `e^(−1/1.288) = 0.46006 ≈ 0.4596` (the β↔α relation `α_opt = e^(−1/β)` holds). ✓

### Why the prior pass's interpretation is OVERSTATED — three refutations

**(1) U2009 hedges 0.4596 heavily; it is NOT presented as a hard optimum.** The paper explicitly
frames it as provisional and heuristic:

> "Data on existing flow networks of ecosystems **do not appear sufficient to determine a precise
> value for b**."
> "the best possible configuration for sustainability **under the information currently available**."
> "**Should it survive further scrutiny**, this threshold in a provides an extremely useful guide."

And U2009 explicitly says the value is **domain-dependent, not universal**:

> "There is **no apriori reason to assume that the value of b is universal**. There might be one
> value of b most germane to ecosystem networks, **another for economic communities**, and still
> another for networks of genetic switching."

So U2009 itself does **not** claim 0.4596 is THE operating optimum for anything but ecosystems, and
even there only tentatively. The prior pass's language ("scientifically-correct target," "the
paper's operating optimum is unambiguously α = 0.4596") is stronger than the paper's own hedged,
domain-relative framing.

**(2) A SECOND 2009 Ulanowicz paper contradicts the "rejects 1/e" reading.** DUAL (same author,
same year) treats **α = 1/e as THE sustainability optimum and an attractor point**, not as a
rejected artifact:

> "One observes that most systems cluster around the **maximal fitness (a = 1/e)**, with some bias
> towards higher values of a." (DUAL §7)
> "The data … reveals a striking natural tendency for systems to **gravitate towards configurations
> of maximal fitness** … ecosystems tend to gravitate towards configurations that possess maximal
> fitness for evolution." (DUAL §8)
> "At a = (1/e), all flows contribute equally towards sustaining the system in this **propitious
> state** … the system is acting as a coherent whole." (DUAL, after Eq. 5)
> "**Analytical proof that a = [1/e] … is an attractor point for living systems** has yet to be
> provided [but] … whenever the starting value a0 < (1/e), the sequence converges … to a = (1/e)."
> "at the attractor point itself, noise plays the role of an idempotent operator … it indicates
> that **systems in nature could sustain themselves indefinitely at (1/e)** without supplementary
> work. **It appears to be the point of natural sustainability.**" (DUAL §8)

This is the opposite of "the theory explicitly rejects 1/e as the sustainability optimum." In DUAL,
1/e **is** the point of natural sustainability, and — critically for E5/E6 — Ulanowicz says systems
sitting **above** 1/e are the *artificial* ones:

> "It is not that systems cannot exist when a > (1/e) (**as with many artificial systems, e.g.,
> agriculture or economics**), but that additional work is required to maintain metastable
> configurations."

**(3) The window-center arithmetic in U2009 does not close.** U2009 states the window as
c ∈ [1, 3.01], n ∈ [2, 4.5], then calls **c = 1.25, n = 3.25** the "geometric center." But the
midpoint of the c-range is (1 + 3.01)/2 = **2.005** (geometric mean 1.735), not 1.25; only the
n-value (3.25) is the true midpoint of [2, 4.5]. So "c = 1.25" is **not** the center of the stated
c-window by any standard definition (arithmetic or geometric). Either there is a typo/idiosyncratic
construction in U2009, or the (c,n)→α mapping is more involved than a midpoint (ZU2003 defines
effective connectivity as `e^{Φ/2}` and roles via `F/N`, `N²/F`; the transform is non-trivial).
**Either way, 0.4596 is the output of a heuristically-chosen, arithmetically-loose center — not a
first-principles optimum.** This further undercuts treating 0.4596 as a hard scientific constant.

**Net E1:** 0.4596 is real, is in U2009, and is a legitimate *ecosystem heuristic center*. But
(a) U2009 presents it tentatively and domain-relative, not as a universal optimum; (b) the same
author's DUAL paper treats **1/e** as the natural-sustainability attractor; and (c) the window-center
that produces 0.4596 doesn't even sit at the arithmetic center of the quoted window. The prior pass
faithfully quoted the 0.4596 sentence but **overstated it into a doctrine the corpus does not
uniformly support.**

---

## E2 — Does the theory endorse the α ∈ [0.2, 0.6] window?

**Verdict: CONFIRM the prior pass's finding (with a caveat it under-weighted).** The [0.2, 0.6]
α-band is **not** in the primary literature; it is a secondary-literature operationalization.

U2009 and ZU2003 define the window on **(c, n)** axes, not α:

> "they plotted the networks, **not on the axes A vs. F, but rather on the transformed axes
> c = 2^{Φ/2} and n = 2^A** … c measures the effective connectivity of the system in links per node
> … n gauges the effective number of trophic levels." (U2009 §5)
> "the empirical networks all cluster within a rectangle that is bounded roughly in the vertical
> direction by **c = 1 and c ≈ 3.01** and horizontally by **n = 2 and n ≈ 4.5**." (U2009 §5)

ZU2003 confirms the axes are effective connectivity (`e^{Φ/2}` = F/N) and effective number of roles
(`N²/F`), with c ≈ 3.015 the empirical connectivity ceiling — **there is no α-band [0.2, 0.6]
anywhere in either primary source.** The prior pass is correct: U2009 gives a *single* α = 0.4596
(a point, the window center), never an α-interval.

**Caveat the prior pass under-stated:** The [0.2, 0.6] band is not merely "approximate but not
contradicted" — it is a **materially different object** from the paper's construct. The paper's
window is a 2-D rectangle in (effective-connectivity, effective-roles) space; collapsing it to a
1-D α-interval discards the connectivity/roles structure entirely, and two systems with identical α
can sit inside vs. outside the true (c,n) window. So [0.2, 0.6] is not a faithful projection of the
published window; it is a convenience band from popularizations (Lietaer/Goerner trade writing).
Verified: 0.4596 sits ~65% up the [0.2, 0.6] band, and 1/e ~42% up — so the band is at least
*consistent* with either candidate optimum, but it is not derived from the paper.

**Recommendation:** keep [0.2, 0.6] only as an **explicitly labeled heuristic**, never cited as a
U2009 result — and see E6 for why it is likely mis-calibrated for organizations regardless.

---

## E3 — Is R = −α·ln α an acceptable "robustness," or is it reserved for the β-adjusted Eq.16/17?

**Verdict: REFUTE the prior pass's "mislabeled" claim.** `−α·log α` is a **legitimate, published
robustness formula** — including for economics — not a mere "proxy" to be relabeled.

The prior pass (F5/R1) called the code's `R = −α·ln α` a mislabel that is "**not** the paper's Eq-17
robustness." That framing is too strong. **FATH2019 — the peer-reviewed paper that applies this to
economic/organizational networks — literally defines Robustness as `−α·log α`:**

> "The Window Vitality measures a network's degree of organization as **α = A/C**. **Systemic
> Robustness is measured as: Robustness = −α log α.** A healthy economy is presumed to **maximize
> the robustness value**, as is seen in ecosystems." (FATH2019, Appendix A)

This is exactly the code's form (F5/R1: `R = −α·ln α`), and FATH2019 says a healthy economy
**maximizes** it — whose maximum is at **α = 1/e**, not 0.4596. So within the paper that governs the
org/economics application, the code's robustness formula and its 1/e peak are **correct and
paper-backed**, not a mislabel.

The distinction the prior pass drew (Eq-15 `−kα·log α` "fitness for evolution" vs. Eq-17
`R = T··×F` with β=1.288) is real *inside U2009's own derivation*, but the theory does **not**
"reserve" the word robustness for the β-form:
- U2009 Eq (17): `R = T···F` — this is **dimensioned** (scaled by total throughput T··), an
  absolute magnitude, not a 0–1 score.
- The code's `−α·ln α` and FATH2019's `−α·log α` are the **dimensionless** shape (the F-fraction
  with k=1, β=1). This is the appropriate quantity for a **cross-network comparable 0–1 index** —
  which is exactly what OASIS needs. Multiplying by T·· (Eq-17) would make a 5-node org and a
  40-node ecosystem incomparable, defeating the purpose.

**Does the choice change WHERE healthy systems should sit? Yes — and this is the crux.** The β=1.288
kernel peaks at α=0.4596; the k=1 kernel (`−α·ln α`, as in FATH2019) peaks at α=1/e≈0.368. So
"robustness" and "operating optimum" are entangled: if you adopt FATH2019's `−α·log α` as robustness
(which the org-facing paper does), the internally-consistent optimum is **1/e, not 0.4596.** The
prior pass wants to keep `−α·ln α` as the metric **and** move the target to 0.4596 — but those two
choices come from **different kernels** and are **mutually inconsistent**. Maximizing `−α·log α`
gives 1/e; you only get 0.4596 by switching to the β=1.288 Eq-16 kernel. **The prior pass's "fix" of
0.37→0.4596 while leaving R=−α·ln α in place would leave the codebase's robustness peak (1/e) and
its α-target (0.4596) pointing at two different α values — arguably a worse internal contradiction
than the one it set out to fix.**

---

## E4 — Is "average shortest path length" a valid proxy for trophic depth?

**Verdict: CONFIRM the prior pass (this one is right).** Topological shortest-path is **not** a valid
ecological trophic-level measure; the theory requires the **flow-weighted** effective trophic level.

The papers repeatedly cite the Lindeman/Levine flow-network lineage for trophic structure, never a
graph-topological shortest path:

> "Almost 70 years ago Raymond **Lindeman (1942)** … attempted to describe quantitatively the
> trophic processes … A rich literature … has ensued (e.g., Hannon, 1973; Finn, 1976; **Levine,
> 1980**; Fath and Patten, 1999; Ulanowicz, 2004b)." (DUAL §6)
> Reference list, DUAL: "**Levine, S., 1980. Several measures of trophic structure applicable to
> complex food** [webs]"; "**Lindeman, R.L., 1942. The trophic-dynamic aspect of ecology.**"

Ecologically, the effective trophic level is defined by the **flow-weighted** average number of
transfers a quantum of medium makes (Levine 1980 apportionment; column-sums of the Leontief-style
`[I−G]⁻¹`), which yields **fractional** levels (a consumer eating 50% plants / 50% herbivores sits at
level 2.5). Unweighted `average_shortest_path_length` counts topological hops, ignores flow
magnitude, cannot produce fractional levels, and conflates "distance between any two nodes" with
"trophic position relative to the primary-producer base." **How wrong is it? Substantially and
directionally biased:** a heavily side-branched or cyclic web can have a short average path but deep
effective trophic structure, and vice versa. This is a genuine defect; the prior pass's
Levine-1980/`[I−G]⁻¹` recommendation is theoretically sound.

---

## E5 — THE BIG ONE: org = ecosystem transferability of the numeric optima

**Verdict: PARTIALLY-CONFIRM the concept, but REFUTE any claim that the SAME numeric window/optimum
transfers to organizations.** The theory transfers the **qualitative** efficiency-vs-resilience
tradeoff; it explicitly does **NOT** assert the same numeric optimum for economies/organizations —
and the corpus repeatedly says economic networks sit *elsewhere*.

**FATH2019 does apply the Window of Vitality and robustness to economics** — but with heavy,
explicit caveats that the ecological *numbers* may not carry over:

> "Some applications of network principles to human systems reveal the need for **modification and
> further study** to understand **how they must be applied differently to socio-economic networks**.
> For example, using REP #6 and the robustness index, **economic networks appear less efficient
> (more redundant) than ecosystems**. **We continue to work to understand what explains this**
> relative to a universally-observed pattern in ecological networks." (FATH2019 §4)

> "One hypothesis is that networks in which exchange between components is crucial to 'survival' will
> exhibit the optimal balance seen in natural ecosystems, while **networks of optional, less critical
> exchange may not.**" (FATH2019 §4)

> "One study of U.S. interstate food trade found the REP #6 measure of robustness **near the curve
> peak**. However, the robustness index calculated for nitrogen flow in the U.S. beef supply network
> **plotted to the right of the peak**. **Work remains to explain when and why networks plot in the
> three regions** of the robustness, Window of Vitality, curve." (FATH2019 §4)

So the authors who *invented* the economic application report that **real economic/supply networks
plot at DIFFERENT points** than the ecological optimum, and that this is an **open research
question**, not a settled calibration. FATH2019 defines the machinery (α = A/C, Robustness =
−α log α, "maximize robustness") but does **not** publish a validated numeric window or a validated
numeric optimum *for organizations* — and certainly not 0.4596.

**And U2009 itself pre-empts the transfer:**
> "There is no apriori reason to assume that the value of b is universal. There might be one value of
> b most germane to ecosystem networks, **another for economic communities**, and still another for
> networks of genetic switching." (U2009 §5)

**And DUAL positions economics ABOVE the ecological optimum by construction:**
> "It is not that systems cannot exist when a > (1/e) (**as with many artificial systems, e.g.,
> agriculture or economics**), but that additional work is required to maintain metastable
> configurations." (DUAL §8)

**Expert judgment on E5:** The **qualitative** claim transfers — every source supports "too much
efficiency → brittleness, too much redundancy → stagnation, health lies in between." The **specific
numeric constants do NOT transfer as established science.** The ecological window (c∈[1,3.01],
n∈[2,4.5], α≈0.4596 or the 1/e attractor) was fit to **48 trophic ecosystem flow networks**
(ZU2003). Applying those exact numbers to departments/emails/documents is **not** something
Ulanowicz or Fath claim; Fath explicitly flags it as unresolved and empirically *different*. This
means a "near-universal fail" of org samples against the ecological window is **weak evidence about
the orgs and strong evidence about a calibration mismatch** (see E6).

---

## E6 — Calibration implication: is the org "fail" real, a mis-set window, or a units artifact?

**Verdict: predominantly (b) a mis-set/mis-transferred window, with a real (c) units/scale-sensitivity
risk — NOT (a) genuine universal dysfunction.**

The reported pattern (org samples at α ≈ 0.07–0.10 reading "unsustainable"; only a literal wetland at
α ≈ 0.58 "passing") is, from an ecosystem-dynamics standpoint, a **red flag on the measurement, not a
finding about the orgs**, for three theory-grounded reasons:

1. **α ≈ 0.07–0.10 is off the bottom of even the ecological window.** ZU2003's window has a lower
   edge (c=1: "the networks being considered are all fully connected"). Organizational flow networks
   built from emails/documents are typically **large, sparse, and diffuse** (high effective
   connectivity, many weak parallel ties → low A/C). An α near 0.07–0.10 means the network is almost
   all reserve/overhead — which the theory reads as "extremely under-organized." That the *entire*
   org corpus lands there, while only a literal ecosystem passes, is the classic signature of a
   **threshold imported from the wrong domain**, exactly the mismatch FATH2019 flags ("economic
   networks appear less efficient / more redundant than ecosystems").

2. **The theory predicts economies sit on the OTHER side (α > 1/e), not far below.** DUAL says
   artificial systems (agriculture, economics) tend to α **> 1/e** (over-organized, over-efficient).
   Observed org α ≈ 0.07–0.10 sits **far below** 1/e — the opposite direction. This inconsistency
   with the theory's own qualitative prediction strongly implies the org α is being computed on a
   network representation (granularity, flow units, inclusion of countless weak edges) that is **not
   commensurable** with the trophic flow networks the window was calibrated on. That is a
   **scale/units/representation artifact** (option c) feeding a **mis-set window** (option b).

3. **α = A/C is scale-invariant, but WHAT α you get depends entirely on the network you build.**
   The number itself isn't unit-dependent, but org-network construction choices (edge threshold,
   directed vs. undirected, self-loops, how "flow" is quantified from email/doc counts) move α
   enormously. A wetland flow network is a curated, throughput-weighted trophic model; an
   email/document graph is not. Comparing them on one fixed α-threshold is not comparing like with
   like.

**Conclusion E6:** The near-universal org "fail" is **most consistent with an ecological window
mis-transferred to organizations (b)**, aggravated by network-construction/scale effects (c). It is
**not** credible, on this theory, as evidence that essentially all real organizations are genuinely
dysfunctional (a). Ulanowicz and Fath both explicitly leave the organizational calibration open;
treating the ecological window as a pass/fail gate for orgs manufactures a failing signal.

---

## Closing judgment (the two questions asked)

### (i) Is the 0.4596 correction theoretically sound, and for what?

**Only narrowly, and NOT as a blanket "fix."**
- **Sound** as: the *ecosystem-specific* propitious α that U2009 §6 explicitly derives (0.4596,
  β=1.288). If a formula in the code is *specifically* implementing "U2009's β=1.288 window-center
  optimum for ecosystems," then 0.4596 is the right constant (the code already does this correctly at
  R7 / `ulanowicz_calculator.py:855-861`, `oasis_calculator.py:282-288`).
- **NOT sound** as a universal drop-in replacement wherever the codebase uses 1/e/0.37, for two
  reasons the prior pass missed:
  1. **Internal inconsistency.** The code's robustness kernel is `−α·ln α` (FATH2019's own economic
     robustness formula), whose maximum is **1/e**. Setting the α-*target* to 0.4596 while the
     robustness *peak* stays at 1/e makes "the optimum" and "the robustness maximum" disagree — you
     cannot mix the k=1 kernel (peak 1/e) with the β=1.288 target (0.4596) and stay coherent. To
     legitimately target 0.4596 you must **also** switch robustness to the Eq-16 β=1.288 kernel
     everywhere (a bigger, breaking change, and one FATH2019 does *not* endorse for economies).
  2. **Domain mismatch.** For the **organizational** application, the governing paper (FATH2019)
     uses `−α·log α` and says "maximize robustness" (⇒ **1/e**), and explicitly reports that economic
     networks do **not** match the ecological optimum. There is no peer-reviewed org optimum of
     0.4596.

  **Recommendation:** Do **not** globally replace 1/e/0.37 with 0.4596. Instead: (a) keep 0.4596
  only in the explicitly-ecosystem β=1.288 path; (b) resolve the 1/e-vs-0.4596 mixing by picking
  **one** robustness kernel and letting the optimum follow from it (k=1 ⇒ 1/e, per FATH2019; or
  β=1.288 ⇒ 0.4596, per U2009 ecosystems) — this is a **product/scientific decision, not an
  unambiguous paper-backed fix**; (c) for organizations, treat the α-target as an **open, to-be-
  calibrated parameter**, not a fixed ecological constant. The prior pass's classification of
  0.37→0.4596 as a clean "PAPER-BACKED FIX" that "changes headline numbers" is **not defensible** —
  the papers do not speak with one voice, and the org-facing paper points at 1/e.

### (ii) Is the [0.2, 0.6] org window scientifically defensible, or should it be re-derived/caveated?

**Not defensible as-is for organizations. Re-derive or heavily caveat.**
- It is **not in the primary literature** (U2009/ZU2003 give a 2-D (c,n) rectangle and a single
  α-point, never an α-interval) — the prior pass got this right.
- It is a lossy 1-D collapse of a 2-D (connectivity, roles) window — the prior pass under-stated
  this.
- Even the *ecological* window was fit to 48 trophic ecosystems; **FATH2019 explicitly says economic
  networks plot elsewhere and that the org calibration is an open question.** So applying [0.2, 0.6]
  (or the true (c,n) window) as a **pass/fail gate for organizations has no peer-reviewed basis.**
- The observed near-universal org "fail" (E6) is the predicted symptom of using an ecological window
  on non-ecological networks — a calibration artifact, not a finding.

  **Recommendation:** For organizations, **either** (a) re-derive an org-specific window/optimum
  empirically from a corpus of organizational flow networks (the research FATH2019 itself calls for),
  **or** (b) demote the window to a clearly-labeled *ecological reference band* that is reported
  descriptively ("here is where ecosystems sit") rather than used as a viability verdict for orgs.
  Keeping [0.2, 0.6] as a hard org viability gate is scientifically unsupported.

---

## Summary table of verdicts

| Claim | Prior-pass position | Adversarial verdict | Basis |
|-------|--------------------|--------------------|-------|
| **E1** — 0.4596 is THE optimum; paper "explicitly rejects 1/e" | Strong: 0.4596 correct, 1/e wrong as α-target | **PARTIALLY-CONFIRM / overstated** | U2009 hedges 0.4596 as provisional & domain-relative; DUAL (same author, 2009) calls **1/e** "the point of natural sustainability" & an attractor; window-center arithmetic (c=1.25) doesn't match stated window |
| **E2** — [0.2, 0.6] not verbatim in U2009 | [0.2,0.6] is secondary-lit heuristic; keep but caveat | **CONFIRM** (prior pass under-stated the 2-D→1-D information loss) | U2009/ZU2003 define window on (c,n) axes; no α-interval in primary sources |
| **E3** — R=−α·ln α is a mislabeled proxy | "Not the paper's robustness"; relabel | **REFUTE** | FATH2019 App. A: "Systemic Robustness is measured as: Robustness = −α log α … maximize" — the org paper uses this exact form; peak = 1/e |
| **E4** — shortest-path ≠ trophic depth | Use Levine-1980 flow-weighted `[I−G]⁻¹` | **CONFIRM** | DUAL cites Lindeman 1942 / Levine 1980 lineage; effective trophic level is flow-weighted & fractional |
| **E5** — org=ecosystem numeric transfer | (implicit) window/optimum applies to orgs | **REFUTE for numbers, CONFIRM for the qualitative tradeoff** | FATH2019: economic nets "less efficient than ecosystems," plot in different regions, "work remains"; U2009: β "not universal … another for economic communities"; DUAL: economics sits at α>1/e |
| **E6** — near-universal org fail | (treated as a real scoring input) | **Mostly (b) mis-set window + (c) scale artifact; NOT (a) genuine universal dysfunction** | Org α≈0.07–0.10 is off the bottom of even the ecological window & on the wrong side of DUAL's α>1/e prediction for artificial systems |

*Adversarial review only. No source code modified. Not committed.*
