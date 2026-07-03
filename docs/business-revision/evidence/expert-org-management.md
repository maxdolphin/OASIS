# Expert Review — Modern Org-Management & Org-Design Lens on OASIS

**Reviewer role:** Organizational-design / org-health expert (complexity-based org design, organizational
network analysis, McKinsey OHI, adaptive/Teal self-management, Galbraith Star, Team-of-Teams, systems
thinking), engaged to **translate the ecology into defensible modern-management terms** and to make the
**Track-2 product/calibration decisions** credible to a C-suite. This document drives code changes; it
does **not** modify source and does **not** re-litigate the science (the ecology verdicts in
`expert-ecosystem-dynamics.md` stand — I build on them, I do not override them).

**Inputs I relied on:** `OASIS-formula-errors-report.md` (E-1 roll-up veto, E-2 org-calibration,
E-24/E-25 size normalization), `evidence/expert-ecosystem-dynamics.md` (the key finding: ecological
viability optima are **not** established to transfer to organizations; Fath 2019 puts org/economic
networks in a different region of the curve; the "every org unsustainable" pattern is a
mis-calibrated-window artifact), `evidence/validation-G-oasis-composite.md` (composite structure,
sub-weights, caps, bands), and `src/oasis_calculator.py` (the five dimensions, the 0.2/0.6 window, the
HEALTHY/WARNING/CRITICAL bands at 60/40).

---

## Bottom line up front (for the controller)

1. **Org viability calibration:** Do **not** ship a pass/fail gate built on the ecological window.
   Reframe SUSTAINABLE from a **verdict** to a **position-on-a-gradient with a direction-of-travel arrow**
   ("you are over-diffuse / under-structured — move toward more structure"), anchored on a **benchmark-
   relative percentile vs. size-matched peers**, with the ecological window shown only as a **descriptive
   reference band, clearly labeled "ecosystem-derived, indicative."** It is **not acceptable** to call
   almost every real company "unsustainable"; that is a calibration artifact, and the fix is honest
   reframing (gradient + peer percentile), **not** faking ecological validity.

2. **The roll-up veto (E-1):** A viability floor makes **management sense** and should ship, but as a
   **worst-dimension band-cap, not a SUSTAINABLE-only kill switch** *while the org window is still
   mis-calibrated*. Concrete rule: **overall cannot be labeled "Healthy / Thriving" if any dimension is
   CRITICAL** (cap at "Needs Attention"); reserve a hard "Non-Viable" veto for SUSTAINABLE **only after**
   the org window is re-calibrated (§1). Keep the 0–100 number; fix the **label**. This is the single
   highest-ROI credibility fix.

3. **Dimension weights:** Equal 20% is **defensible as a v1 default** and I recommend **keeping it as the
   published default**, but expose a **small number of named, evidence-tagged weighting profiles**
   (e.g. "Scale-up / Growth," "Efficiency / Turnaround," "Regulated / Resilience-first") rather than
   inventing new "true" weights. If forced to a single tilt, the defensible one is a **modest resilience
   emphasis** (SUSTAINABLE + SYMBIOTIC slightly up) because modern org-health evidence links
   collaboration + adaptive resilience most strongly to durable performance — but only *after* the
   SUSTAINABLE calibration is fixed, otherwise you are up-weighting a broken signal.

3b. **Size normalization (E-24/E-25):** **Endorsed.** A 6-person startup and a 5,000-person enterprise
   are structurally different organisms; scores **must be size-aware.** The size-relative direction is
   correct management science, not a hack.

Recommendation file: `docs/business-revision/evidence/expert-org-management.md`.

---

## 1. Organizational viability calibration — the big one

### 1.1 The management problem, stated plainly

OASIS today runs a fixed α-window (heuristic [0.2, 0.6], optimum ~0.37) as a **viability verdict**. On
real org data α lands at ~0.07–0.10, so essentially every company reads "outside the window →
unsustainable," while a literal wetland passes. The ecology panel established *why* this is not a finding
about the orgs: Fath (2019) says economic/organizational networks are **more redundant, less efficient,
and sit in a different region of the curve**, and their calibration is an **open research question**. So
the tool is currently answering a question the science says it cannot yet answer, and answering it wrong.

From a management standpoint this is fatal to trust. The first time a COO of a demonstrably successful,
growing company sees "your organization is unsustainable / non-viable," the tool loses the room. A
diagnostic that fails everyone diagnoses no one — it has **zero discriminating power** and reads as a
gimmick. No executive buys a health index that flunks the S&P 500.

### 1.2 The three options, weighed

**(a) Keep a theory-anchored band but widen/recenter it for orgs, labeled indicative.**
*Pro:* keeps a single interpretable band; minimal code. *Con:* there is **no peer-reviewed org band to
recenter onto** — you would be inventing constants and dressing them as science, which is exactly what the
project rule and the ecology panel forbid. Widening [0.2,0.6] until orgs "pass" is reverse-engineering a
result. **Reject as the primary mechanism.** (Keep the band only as a *descriptive reference*, per (b)/(c).)

**(b) Reframe from pass/fail to position-on-a-gradient with direction-of-travel.**
*Pro:* This is exactly how credible org-health instruments already work. McKinsey OHI reports a
**percentile and a quartile with improvement priorities**, not "pass/fail." Gallup Q12, the Star Model
diagnostics, and the Team-of-Teams adaptability assessments all output **"here is where you are, here is
which way to move,"** never "you are dead." The efficiency-vs-resilience (α) axis is *genuinely* a
gradient — the Ulanowicz/Fath tradeoff itself is "too rigid ↔ too diffuse, health in between." Reporting
**where on that gradient you sit and which direction reduces your dominant risk** is the honest, faithful
use of the theory. *Con:* you lose the crisp binary; you must define the arrow logic. **This is the core
recommendation.**

**(c) Benchmark-relative (percentile vs. peers).**
*Pro:* Solves the "everyone fails" problem structurally: if every org clusters at α≈0.08, then α is
**re-scaled against the org population**, so a company at the 80th percentile of its size-class reads
"more structured than most peers," which is a *true, defensible* statement that makes no claim about the
ecological optimum. This is standard consulting practice (OHI's entire value proposition is
**percentile-vs-a-database-of-companies**). *Con:* needs a reference corpus of org flow-networks; early
on it is thin. **Adopt as the calibration backbone**, seeded now and improving as the corpus grows.

### 1.3 Recommended synthesis: (b) as the framing, (c) as the calibration, (a) as descriptive context only

Report SUSTAINABLE / structural-balance as:

- **Primary output — a gradient position:** a labeled point on an efficiency↔resilience axis
  ("Over-connected / Diffuse" ← optimal band → "Over-structured / Rigid"), with the org placed by **its α
  relative to a size-matched peer distribution (percentile)**, not by the raw ecological window.
- **Direction-of-travel arrow:** a single unambiguous recommendation — *"Your structure is diffuse
  relative to peers; consolidate decision flows / strengthen core coordination to move toward balance."*
  (or the mirror image for the rare over-rigid case). This is the sentence a COO acts on.
- **Descriptive reference band:** show the ecological window as a faint reference (*"Natural ecosystems
  cluster here — shown for context; not an organizational pass/fail threshold"*). This preserves the
  intellectual lineage **without** weaponizing an unvalidated constant into a verdict.

**Is it acceptable for the tool to call almost every real company "unsustainable"? No.** That output is a
mis-calibrated-window artifact (per the ecology panel), it destroys credibility, and it is scientifically
unsupported for orgs. The fix is **not** to fudge α or fake an org optimum — it is to **stop using the
ecological window as a gate**, report **position + direction + peer-percentile**, and label the ecological
band as indicative context. This keeps the science honest and the product sellable simultaneously.

**Code implication (for the controller, not implemented here):** demote the [0.2,0.6] window from a
status gate to a descriptive band; add a percentile transform of α against a size-bucketed reference
corpus; drive the SUSTAINABLE narrative from *direction-of-travel* logic keyed on which side of the peer
median the org sits. Keep 1/e where it correctly normalizes the robustness proxy (per the ecology panel);
do **not** globally swap to 0.4596.

---

## 2. The roll-up veto (E-1) — does a viability floor make management sense?

### 2.1 The management verdict on non-compensatory scoring

**Yes — a floor is correct, and it is standard.** The current flat average lets `(100,100,100,100,0)`
average to 80 → "HEALTHY," so a collapsed dimension is silently masked. In org-health terms this is a
**category error**, and management practice already rejects it:

- **Balanced-scorecard / OKR logic:** you don't declare a business healthy because three of four
  perspectives are green while the fourth (say, financial viability) is red. A red pillar caps the
  verdict.
- **Reliability / risk framing a COO already owns:** health is closer to a **chain than a portfolio** —
  a single failed link governs the outcome. Executives intuitively accept "we don't call the org healthy
  while a core system is critical."
- **McKinsey OHI** treats the outcome dimensions as **jointly necessary** (health is the *simultaneous*
  presence of the ingredients), not as a compensable sum where surplus alignment offsets absent
  accountability.

So averaging away a collapsed dimension is indefensible. A floor/veto is the right instinct.

### 2.2 But: veto on *what*, given the calibration caveat?

Here management judgment must respect the science. E-1 (as written) proposes "overall cannot be HEALTHY if
any dimension — **especially SUSTAINABLE** — is CRITICAL." The problem: **SUSTAINABLE is currently the
mis-calibrated dimension** (§1). If we hard-veto on SUSTAINABLE *before* recalibrating, we simply
re-manufacture the "everyone is non-viable" failure at the headline level — worse, because now it is a
hard gate. That would be encoding a known artifact into the top-line verdict.

**Recommended roll-up logic (phased):**

- **Phase 1 (ship now) — worst-dimension band cap, dimension-agnostic:**
  Keep the weighted mean as the **score**. Constrain the **label**:
  *overall status cannot be "Healthy/Thriving" if **any** dimension is CRITICAL* → cap at
  **"Needs Attention."** This kills the "Non-Viable labeled Healthy" contradiction, is trivially
  explainable ("we never call you healthy while a pillar is critical"), and does **not** privilege the
  still-broken SUSTAINABLE dimension. It also must be paired with **de-saturating the caps (E-24)** so the
  four carrier dimensions stop pinning at 100 and masking the fifth.

- **Phase 2 (after §1 recalibration) — SUSTAINABLE viability veto:**
  Once SUSTAINABLE is re-expressed as a peer-relative gradient, a genuine "bottom-decile structural
  balance" **can** justifiably cap the overall at "Non-Viable / At-Risk," because at that point the signal
  is real, not an ecological-window artifact.

### 2.3 Veto vs. weighted vs. geometric mean — what to tell a COO

- **Weighted arithmetic mean (status quo):** "averages away" a collapse — reject for the *label*.
- **Hard veto:** correct instinct, but too blunt to apply to the mis-calibrated dimension today → use the
  **soft band-cap** version (Phase 1) now.
- **Geometric mean:** the *principled* long-term answer — it encodes **"all pillars must be adequate; you
  cannot buy your way out of a collapsed one"** (low-substitutability, Cobb-Douglas semantics), which is
  exactly the management truth. But it **re-baselines every score and requires re-calibrating the 60/40
  bands.** Recommend it as a **Phase 2/3 upgrade once the corpus and calibration exist**, not as the first
  move.

**Recommended overall-verdict labels** (drop "HEALTHY/WARNING/CRITICAL" clinical language for exec-facing
output; keep internally):
`Thriving` → `Healthy` → `Needs Attention` → `At Risk` → `Critical / Non-Viable`,
with the rule that **the overall label can never be more than one band above the worst dimension**, and
**never "Thriving/Healthy" while any dimension is Critical.** This is the language a COO accepts without a
statistics lecture.

---

## 3. Dimension weights & meaning — the modern-management mapping

### 3.1 Are equal 20% weights defensible?

**As a v1 default, yes — and I recommend keeping equal weights as the published default.** Equal weighting
is the honest choice when you lack an outcome-validated weighting model, it is transparent, and it avoids
implying a false precision ("we know Intelligent matters 1.4× Autonomous") that no peer-reviewed org study
supports for *these specific network constructs*. Modern frameworks do imply some dimensions carry more
outcome variance (see below), but the credible way to express that is **named weighting profiles the
client selects by context**, not a single re-tuned vector shipped as truth.

### 3.2 The five dimensions mapped to recognized org-design constructs

| OASIS dimension | Modern-management construct it credibly maps to | Anchoring framework(s) | Weight guidance |
|---|---|---|---|
| **Open** | **External adaptability / boundary-spanning / environmental sensing** — the org's connective openness to its environment and internal information bridges. | Team-of-Teams (shared consciousness, permeability); Galbraith Star (Structure/Info-flow); Aldrich/Tushman boundary-spanning; sensing side of **dynamic capabilities** (Teece). | Keep ~baseline. Elevate for scale-ups / fast-changing markets. |
| **Autonomous** | **Distributed decision rights / empowerment / local self-management** — how much coordination and control is devolved vs. centralized. | Galbraith Star (Decision rights); Teal/self-management (Laloux); Bourton/OHI "accountability"; RAPID/decision-rights literature (Rogers & Blenko). | Keep ~baseline. Over-weighting rewards decentralization *per se*, which is not universally good — caution. |
| **Symbiotic** | **Cross-functional collaboration / psychological safety / relational coordination** — the quality and reciprocity of internal collaboration. | Edmondson (psychological safety, teaming); Gittell (relational coordination); OHI "coordination & control" + "capabilities"; Team-of-Teams trust. | **Candidate for a modest up-weight** — collaboration/safety are among the most outcome-validated org-health levers. |
| **Intelligent** | **Organizational information-processing / learning / knowledge diversity** — capacity to process information and hold diverse roles/knowledge. | Galbraith information-processing view; March exploration/exploitation; Senge learning organization; sensing/seizing in dynamic capabilities. | Keep ~baseline; elevate in knowledge-intensive firms. |
| **Sustainable** | **Long-term resilience / structural balance / adaptive capacity (efficiency-vs-resilience)** — the org's structural viability over time. | Fath 2019 (window of vitality — *for orgs, calibration open*); Reeves (BCG) resilience; Holling adaptive cycle; OHI "long-term direction." | **Do not up-weight until recalibrated (§1).** Up-weighting a broken signal amplifies the artifact. Post-fix, a modest resilience emphasis is defensible. |

### 3.3 Recommended weighting policy

1. **Publish equal 20% as the default.** Transparent, honest, no false precision.
2. **Offer 2–4 named, context-tagged profiles** (weights the *client* chooses, each with a one-line
   rationale tied to a recognized framework), e.g.:
   - *Scale-up / Growth:* tilt to **Open + Intelligent** (sensing & learning dominate in growth).
   - *Efficiency / Turnaround:* tilt to **Autonomous + Sustainable** (decision clarity & structural
     discipline).
   - *Regulated / Resilience-first:* tilt to **Sustainable + Symbiotic** (durability & coordinated
     control).
3. **If a single non-equal default is ever mandated, the only defensible tilt is a modest
   Symbiotic + Sustainable emphasis** (collaboration + resilience have the strongest modern org-health
   evidence base), and **only after** the SUSTAINABLE recalibration. Ship this as a *documented profile*,
   not as silent constants.

Do **not** invent precise non-equal weights and present them as validated — there is no peer-reviewed
weighting for these specific network constructs, and doing so repeats exactly the over-claiming the
ecology panel flagged.

---

## 3b. Size normalization (E-24/E-25) — management endorsement

**Endorsed without reservation.** Small teams and large enterprises are **structurally different
organisms**, and org-design theory is explicit about it:

- **Span-of-control and structural-differentiation** research (Blau, Mintzberg) shows connectivity,
  centralization, and role differentiation scale non-linearly with headcount — a 6-person team is
  *supposed* to be densely, informally connected; a 5,000-person firm *must* be sparser and more
  modular. Applying one fixed cap/divisor across both mis-reads the small org as "over-connected" and the
  large org as "under-connected" purely as a size artifact.
- **Mintzberg's configurations** (simple structure → machine/professional bureaucracy → adhocracy) are
  literally size- and complexity-indexed; the "right" structure is contingent on scale.

So the E-24/E-25 direction — **make caps and divisors size-relative (relative to n / size bucket) rather
than fixed** — is correct management science, not a workaround. A 6-person startup should **not** be
scored on the same absolute structural yardstick as a 5,000-person enterprise. This also directly
reinforces §1's **size-matched peer percentile**: you cannot benchmark against peers without first making
the raw metrics size-comparable. **Recommendation: proceed with size-relative normalization; document
the size buckets; treat the caps as the first place to fix (they also drive the E-1 masking).**

---

## 4. Executive framing / output — turning "α = 0.09" into a decision

A consultant handing this to a C-suite needs a **diagnose-and-benchmark** deliverable, not a physics
readout. The management narrative that turns "α = 0.09" into action is:

> *"On the efficiency-vs-resilience spectrum, your organization sits in the **diffuse / under-structured**
> zone — **more decentralized and redundant than [X]% of comparably-sized organizations**. That buys
> resilience but costs coordination and speed. **The highest-leverage move is to strengthen core decision
> flows and cross-functional coordination** to pull toward the balanced zone. Your strongest pillar is
> [Symbiotic]; your binding constraint is [Sustainable/structural balance]."*

**The 3–5 things an exec actually needs (in this order):**

1. **One headline verdict + one number, in plain language** — "Needs Attention (58/100)", never
   "Non-Viable, 76 HEALTHY" (the E-1 contradiction). Consistency between label and number is table stakes.
2. **Where you stand vs. peers** — a **percentile / quartile against size-matched organizations.** This is
   the single most trusted artifact for a C-suite (it is OHI's whole franchise). Absolute ecological
   scores mean nothing to them; relative position means everything.
3. **The 1–2 binding constraints (weakest pillars) and the direction-of-travel** — not five scores, but
   *"here is what is holding you back and which way to move."* Diffuse → add structure; rigid → add slack.
4. **The top 2–3 concrete actions**, each tied to the weak dimension and phrased in management verbs
   (consolidate decision rights, strengthen cross-functional links, reduce redundant reporting lines).
5. **A trajectory / re-measure hook** — "measure again in 2 quarters to confirm movement." Executives fund
   what they can track.

**Presentation principles:** lead with the diagnosis and the peer benchmark; put α, ascendency, and the
information-theory in a methodology appendix; use the efficiency↔resilience **gradient visual** (a marker
on a spectrum with a direction arrow) as the hero chart — it makes "α = 0.09" instantly legible as
"you're over here, move that way." Never surface a raw "unsustainable" verdict on page 1.

---

## 5. Product positioning — the defensible, sellable value proposition

**OASIS is a structural / network lens on organizational health.** It measures — from real interaction and
flow data — *how an organization is actually wired*: its balance between efficiency and resilience, its
distribution of decision flows, the reciprocity of its collaboration, and the diversity of its information
processing, benchmarked against comparable organizations. It answers a question that **culture and
engagement surveys structurally cannot**: not "how do people *feel*?" but "how is the organization
*structured to adapt and coordinate*?" That makes it a **complement to — never a replacement for — OHI /
Gallup-style engagement and culture instruments**: surveys read the human/perceptual layer, OASIS reads
the structural/flow layer, and the combination is more than either alone. Its edge is **objectivity
(computed from behavioral flow data, not self-report), a rigorous complexity-science lineage
(Ulanowicz/Fath information theory), and an actionable efficiency-vs-resilience diagnosis with a clear
direction-of-travel.**

**Claims it must NOT make:** it must **not** claim a scientifically validated organizational viability
threshold or optimum (the ecology panel is explicit: the org window is an **open research question**, and
0.4596/0.37 are ecosystem values); it must **not** issue absolute "your organization is unsustainable /
non-viable" verdicts off the ecological window; it must **not** present the [0.2,0.6] band as a peer-
reviewed organizational result; it must **not** claim to measure culture, engagement, or performance
outcomes directly, or to predict financial results; and it must **not** imply the dimension weights are
empirically validated. Positioned honestly — *"a structural diagnostic and peer benchmark that complements
your culture data"* — it is credible and sellable. Positioned as *"the science says your org is
unsustainable,"* it is neither.

---

*Org-management / org-design review only. No source code modified. Not committed. The ecological-validity
verdicts in `expert-ecosystem-dynamics.md` are treated as binding; nothing here recommends faking
ecological transfer to organizations.*
