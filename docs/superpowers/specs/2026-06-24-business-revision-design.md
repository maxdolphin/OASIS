# Business Revision of OASIS — Design Spec

**Date:** 2026-06-24
**Branch:** feat/detailed-ecosystemic-report
**Status:** Approved design — ready for implementation planning

---

## 1. Objective

Produce a strategy-consultant–grade **Business Revision** of OASIS: a rigorous
diagnosis of how well its dashboards and PDF report serve their business job,
followed by a prioritized redesign roadmap to close the gaps.

The review answers one question: **Would a strategy consultant trust OASIS's
output in front of a client?** — i.e. is it defensible, decision-relevant, and
readable without an ecology PhD.

### Deliverable type
Review **plus** redesign plan: diagnosis → prescription → prioritized roadmap.
This review produces the *plan*; implementing the redesigns is separate
downstream work (its own spec → plan → build cycles).

---

## 2. Scope

| | |
|---|---|
| **In scope** | The in-app Streamlit dashboards **and** the exported PDF report |
| **Out of scope** | The scientific formulas; intervention-planning and time-tracking features |

### Guardrails
- **Formulas are fixed** (per `CLAUDE.md`). This revision changes *presentation,
  framing, information architecture, narrative, and contextualization* — never
  the Ulanowicz / OASIS math. If any finding appears to require a formula change,
  it is flagged as a research question for the `formula-validator` /
  `research-validator` path, **not** actioned in this review.
- **Job is scoped to "diagnose & benchmark."** Intervention-planning ("what
  should I change and what's the impact") and longitudinal tracking ("did it
  work over time") are explicitly out of scope and noted as future opportunities.

---

## 3. Users & the value chain

OASIS serves four personas, organized around a single **operator → executive
handoff**:

- **Operators** (who run the tool): strategy consultant; sustainability /
  transformation lead.
- **Audience** (who must act on the output): C-suite / executive; and the client
  exec a consultant ultimately serves.

The design spine is the handoff: an operator runs OASIS on an organization and
turns the output into something an executive can read, trust, and act on. The
review optimizes for that chain rather than for any single persona. Where the
personas conflict, **Decision relevance** is the tiebreaker dimension.

---

## 4. The analytical lens — Business-Utility Rubric

Every dashboard screen and report section is scored **1–5** against each of the
seven dimensions below. The scores roll up into a heatmap (surface × dimension)
that drives the roadmap.

| # | Dimension | The question it asks |
|---|-----------|----------------------|
| 1 | **Decision relevance** | Does this drive the "diagnose & benchmark" job, or is it data for data's sake? *(Tiebreaker dimension.)* |
| 2 | **So-what clarity** | Is the business implication explicit, or must the user infer it from raw metrics? |
| 3 | **Interpretability** | Can a non-ecologist executive read it without a glossary? Is jargon translated? |
| 4 | **Benchmark / context** | Is a number shown against a reference (threshold, peer, prior period) so "good vs bad" is obvious? |
| 5 | **Credibility / defensibility** | Would a consultant stake their reputation on it with a client? Sources, framework alignment, no overclaiming. |
| 6 | **Narrative flow** | Operator → executive handoff: does the story build headline → evidence → detail? |
| 7 | **Visual effectiveness** | Right chart for the message; signal over decoration. |

---

## 5. Method — structured teardown (hybrid approach)

Consultant engagement structure as the backbone, specialized agents executing the
audit dimensions, grounded in evidence from the live app and real PDF output.

### Step 1 — Inventory the surfaces
Enumerate every distinct surface into a checklist so nothing is reviewed by
impression alone.

- **Dashboards:** Core Metrics, System Health Dashboard, Sustainability
  Assessment, Window of Viability, Extended Network Metrics, Balance Indicators,
  Health Assessments, OASIS radar / gauges, network & Sankey visualizations.
- **Report (PDF):** cover / executive summary, each narrative section,
  benchmarking, risk, roadmap, ESG / framework-alignment sections, glossary
  appendix.

### Step 2 — Evidence capture
Capture real artifacts, not memory:
- Dashboard screenshots from the live app at `localhost:8501`.
- A generated PDF run on representative sample organizations.

### Step 3 — Two contrasting orgs
Audit against **two contrasting organizations** — one viable, one not — to test
whether the surfaces communicate well across outcomes (not just for one case):
- **Unsustainable exemplar:** TechFlow Innovations (already run; verdict
  "unsustainable — too chaotic," robustness ≈ 0.18).
- **Viable counterpart:** selected during execution and confirmed with the user.

### Step 4 — Three-lens parallel audit
Run the three specialized agents in parallel, each scoring its domain against all
seven rubric dimensions:

| Agent | Owns | Primary lens |
|-------|------|--------------|
| `ui-ux-decision-maker` | Dashboards | Visual effectiveness, interpretability, on-screen narrative flow |
| `sustainability-reporting-auditor` | PDF report | Credibility / defensibility, framework alignment, executive narrative |
| `ecosystem-pm` | Both | Decision relevance, so-what clarity, the operator → exec value chain |

### Step 5 — Synthesis & reconciliation
Merge the three audits into one scored matrix (surface × 7 dimensions), reconcile
disagreements, and produce a **gap heatmap** plus a ranked list of
highest-impact deficiencies. Each finding carries: the surface, the failing
dimension(s), the evidence (screenshot / quote), and the business consequence.

---

## 6. Benchmarking-basis workstream

How benchmarking should work is treated as a first-class deliverable. Rubric
dimension #4 is where consultant-grade tools live or die: a number with no
reference is uninterpretable; the same number against a band becomes a finding.

### Layered model (approved)

| Tier | Basis | Role | Status |
|------|-------|------|--------|
| **1. Theoretical norms** | Ulanowicz thresholds — Window of Viability (20–60% efficiency), robustness optimum ≈ 37% | Backbone: every metric framed against its viability band | **Now** — ships immediately, fully defensible, zero data cost |
| **2. Reference library** | The shipped real-world datasets (airports, supply chains, etc.) as anchor points | Contextual "you are here" anchors, clearly labeled illustrative (not normative) | **Near-term** |
| **3. Peer cohort** | Real orgs of similar size / sector | The benchmark execs actually want | **Future / flagged** — data-acquisition gap named explicitly; no fake peer benchmarks |

**Output:** a recommended benchmarking model specifying, for each metric, the
band, the label, and the "so-what" sentence — feeding the redesign roadmap.

---

## 7. Redesign roadmap & prioritization

### Prioritization — Impact × Effort
Every recommendation scored on:
- **Business impact** — how much it moves decision-relevance / so-what /
  credibility (weighted by the tiebreaker dimension).
- **Effort** — presentation-layer tweak vs. structural IA change.

Sorted into three horizons (matching the report's existing convention):

| Horizon | Meaning | Example shape |
|---------|---------|---------------|
| **Immediate** | High-impact, low-effort | Add viability bands + plain-language "so-what" line under each metric |
| **Short-term** | High-impact, moderate-effort | Restructure dashboard IA around the diagnose → benchmark narrative; exec-summary headline redesign |
| **Medium-term** | High-impact, higher-effort | Tier-2 reference anchors; framework-alignment depth; Tier-3 peer-data plan |

---

## 8. Final deliverable

A single **Business Revision document** (Markdown, committed to the repo,
PDF-exportable so it can itself be shown to a stakeholder):

1. **Executive summary** — the verdict in one page: is OASIS consultant-ready
   today, the 3–5 headline gaps, the redesign thesis.
2. **Method & rubric** — scope, the 7 dimensions, the two contrasting orgs.
3. **Findings** — scored matrix + gap heatmap, with evidence and business
   consequence per gap.
4. **Benchmarking strategy** — the Tier 1 / 2 / 3 model and per-metric
   contextualization.
5. **Redesign roadmap** — prioritized recommendations across the three horizons.
6. **Appendix** — full per-surface scores, agent notes.

---

## 9. Success criteria

- Every in-scope surface is inventoried and scored against all 7 dimensions for
  both contrasting orgs.
- Findings are evidence-backed (screenshot / PDF quote), not impressionistic.
- A defensible benchmarking model is recommended with per-metric
  contextualization.
- Recommendations are prioritized by Impact × Effort across three horizons.
- No recommendation alters a scientific formula; any such need is flagged for the
  validator path.
- The output is a document an operator could hand to an executive — or that could
  itself be presented to a stakeholder — without further translation.
