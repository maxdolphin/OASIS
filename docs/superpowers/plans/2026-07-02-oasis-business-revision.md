# OASIS Business Revision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a strategy-consultant–grade Business Revision of OASIS — an evidence-backed diagnosis of dashboard/report business utility plus a prioritized redesign roadmap.

**Architecture:** This plan produces an *analysis deliverable*, not shippable code. It captures real evidence from the running app and generated PDFs for two contrasting orgs, runs a three-lens specialist-agent audit against a 7-dimension rubric, synthesizes a scored gap matrix, resolves a benchmarking model, and assembles a single Business Revision document. Verification steps are completeness/evidence gates, not test runs. No scientific formulas are changed.

**Tech Stack:** Streamlit app (`app.py`), headless Chrome + CDP for dashboard screenshots, existing PDF report path, three project subagents (`ui-ux-decision-maker`, `sustainability-reporting-auditor`, `ecosystem-pm`).

**Reference spec:** `docs/superpowers/specs/2026-06-24-business-revision-design.md`

**Two contrasting orgs (resolved):**
- Unsustainable exemplar: **TechFlow Innovations (Combined Flows)** → `data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json`
- Viable counterpart: **Balanced Test Organization** → `data/synthetic_organizations/combined_flows/balanced_org_test.json`

**Deliverable output paths:**
- Document: `docs/business-revision/2026-07-02-oasis-business-revision.md`
- Evidence: `docs/business-revision/evidence/` (screenshots, PDFs, raw agent notes)

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `docs/business-revision/evidence/surface-inventory.md` | Checklist of every dashboard + report surface to audit |
| `docs/business-revision/evidence/dashboards/` | Dashboard screenshots, per org |
| `docs/business-revision/evidence/reports/` | Generated PDF reports, per org |
| `docs/business-revision/evidence/audit-uiux.md` | `ui-ux-decision-maker` raw scores + notes (dashboards) |
| `docs/business-revision/evidence/audit-report.md` | `sustainability-reporting-auditor` raw scores + notes (PDF) |
| `docs/business-revision/evidence/audit-pm.md` | `ecosystem-pm` raw scores + notes (both, value chain) |
| `docs/business-revision/evidence/scored-matrix.md` | Reconciled surface × 7-dimension matrix + gap heatmap |
| `docs/business-revision/2026-07-02-oasis-business-revision.md` | The final deliverable document |

---

## Task 1: Scaffold workspace + surface inventory

**Files:**
- Create: `docs/business-revision/evidence/surface-inventory.md`
- Create (dirs): `docs/business-revision/evidence/dashboards/`, `docs/business-revision/evidence/reports/`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p docs/business-revision/evidence/dashboards docs/business-revision/evidence/reports
```

- [ ] **Step 2: Enumerate dashboard surfaces from the code**

Read the analysis-rendering sections of `app.py` to confirm the live surface list. Run:

```bash
grep -nE "st\.(header|subheader)\(|Core Metrics|System Health|Sustainability Assessment|Window of Viability|Extended Network|Balance Indicators|Health Assessments" app.py | head -60
```

Expected: line references for each dashboard section. Record the actual section titles found.

- [ ] **Step 3: Enumerate report surfaces from the code**

Run:

```bash
grep -nE "def .*section|add_section|story\.append|Paragraph\(" src/pdf_generator.py src/publication_report.py src/oasis_report.py 2>/dev/null | head -60
```

Expected: the report section builders. Record each report section name.

- [ ] **Step 4: Write the inventory checklist**

Create `docs/business-revision/evidence/surface-inventory.md` with this structure, filling the tables from Steps 2–3:

```markdown
# Surface Inventory

## Dashboard surfaces (in-app)
| ID | Surface | app.py ref | Audited (TechFlow) | Audited (Balanced) |
|----|---------|-----------|--------------------|--------------------|
| D1 | Core Metrics | app.py:LINE | ☐ | ☐ |
| D2 | System Health Dashboard | app.py:LINE | ☐ | ☐ |
| D3 | Sustainability Assessment | app.py:LINE | ☐ | ☐ |
| D4 | Window of Viability | app.py:LINE | ☐ | ☐ |
| D5 | Extended Network Metrics | app.py:LINE | ☐ | ☐ |
| D6 | Balance Indicators | app.py:LINE | ☐ | ☐ |
| D7 | Health Assessments | app.py:LINE | ☐ | ☐ |
| D8 | OASIS radar / gauges | app.py:LINE | ☐ | ☐ |
| D9 | Network & Sankey visualizations | app.py:LINE | ☐ | ☐ |

## Report surfaces (PDF)
| ID | Surface | source ref | Audited (TechFlow) | Audited (Balanced) |
|----|---------|-----------|--------------------|--------------------|
| R1 | Cover / Executive summary | FILE:LINE | ☐ | ☐ |
| R2 | Narrative findings | FILE:LINE | ☐ | ☐ |
| R3 | Benchmarking section | FILE:LINE | ☐ | ☐ |
| R4 | Risk section | FILE:LINE | ☐ | ☐ |
| R5 | Roadmap section | FILE:LINE | ☐ | ☐ |
| R6 | ESG / framework alignment | FILE:LINE | ☐ | ☐ |
| R7 | Glossary appendix | FILE:LINE | ☐ | ☐ |
```

Replace every `LINE`/`FILE:LINE` with real references from Steps 2–3. Add or remove rows to match what the code actually renders.

- [ ] **Step 5: Verify completeness**

Confirm every row has a real code reference (no `LINE` placeholders remain). Run:

```bash
grep -c "LINE" docs/business-revision/evidence/surface-inventory.md
```

Expected: `0`.

- [ ] **Step 6: Commit**

```bash
git add docs/business-revision/evidence/surface-inventory.md
git commit -m "docs(revision): surface inventory for OASIS business revision"
```

---

## Task 2: Capture dashboard evidence — both orgs

**Files:**
- Create: `docs/business-revision/evidence/dashboards/techflow-*.png`
- Create: `docs/business-revision/evidence/dashboards/balanced-*.png`
- Create: `docs/business-revision/evidence/capture-dashboard.py` (reusable CDP driver)

- [ ] **Step 1: Ensure the app is running**

```bash
curl -s http://localhost:8501/_stcore/health || (streamlit run app.py --server.headless true --server.port 8501 > /tmp/oasis_streamlit.log 2>&1 &)
sleep 8; curl -s http://localhost:8501/_stcore/health
```

Expected: `ok`.

- [ ] **Step 2: Write the CDP capture driver**

Create `docs/business-revision/evidence/capture-dashboard.py`:

```python
"""Drive the OASIS app via Chrome DevTools Protocol, select a sample org,
run Analyze, and full-page screenshot the result. Usage:
  python capture-dashboard.py "TechFlow Innovations (Combined Flows)" techflow
"""
import json, sys, time, base64, urllib.request, websocket

ORG_LABEL, PREFIX = sys.argv[1], sys.argv[2]
OUT_DIR = "docs/business-revision/evidence/dashboards"

tabs = json.load(urllib.request.urlopen("http://localhost:9222/json"))
page = next((t for t in tabs if t.get("type") == "page"), tabs[0])
ws = websocket.create_connection(page["webSocketDebuggerUrl"], max_size=None,
        timeout=90, header=["Origin: http://localhost:9222"])
mid = 0
def cmd(m, p=None):
    global mid; mid += 1
    ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r
def ev(expr):
    r = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
    return r.get("result", {}).get("result", {}).get("value")

cmd("Page.enable"); cmd("Runtime.enable")
cmd("Page.navigate", {"url": "http://localhost:8501/"}); time.sleep(11)
ev("(()=>{const l=[...document.querySelectorAll('label')].find(x=>/use sample/i.test(x.innerText));if(l)l.click();return 1;})()")
time.sleep(6)
# select the org card whose text matches ORG_LABEL, then its Analyze button
ev(f"""(() => {{
  const cards=[...document.querySelectorAll('div')].filter(d=>d.innerText && d.innerText.includes({json.dumps(ORG_LABEL)}));
  return cards.length;
}})()""")
ev(f"""(() => {{
  const btns=[...document.querySelectorAll('button')].filter(b=>/^analyze$/i.test(b.innerText.trim()));
  // click the analyze nearest the matching org label
  const target={json.dumps(ORG_LABEL)};
  let best=btns[0];
  btns.forEach(b=>{{ if(b.closest('*') && b.parentElement.innerText.includes(target)) best=b; }});
  if(best) best.click(); return 1;
}})()""")
time.sleep(16)
res = cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
open(f"{OUT_DIR}/{PREFIX}-full.png", "wb").write(base64.b64decode(res["result"]["data"]))
print("SAVED", f"{OUT_DIR}/{PREFIX}-full.png")
ws.close()
```

- [ ] **Step 3: Launch Chrome with remote debugging**

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
pkill -f "remote-debugging-port=9222" 2>/dev/null; sleep 1
"$CHROME" --headless --disable-gpu --no-sandbox --remote-debugging-port=9222 \
  '--remote-allow-origins=*' --window-size=1440,4000 about:blank > /tmp/chrome_cdp.log 2>&1 &
sleep 2; curl -s http://localhost:9222/json | python3 -c "import sys,json;print(len(json.load(sys.stdin)),'targets')"
```

Expected: `>=1 targets`.

- [ ] **Step 4: Capture TechFlow (unsustainable)**

```bash
python3 docs/business-revision/evidence/capture-dashboard.py "TechFlow Innovations (Combined Flows)" techflow
```

Expected: `SAVED docs/business-revision/evidence/dashboards/techflow-full.png`.

- [ ] **Step 5: Capture Balanced (viable)**

```bash
python3 docs/business-revision/evidence/capture-dashboard.py "Balanced Test Organization" balanced
```

Expected: `SAVED docs/business-revision/evidence/dashboards/balanced-full.png`.

- [ ] **Step 6: Visually verify both screenshots**

Read both PNGs. Confirm each shows populated metrics (not the grey Streamlit skeleton) and that TechFlow shows an "unsustainable" verdict while Balanced differs. If either is a skeleton, increase the `time.sleep(16)` in Step 2 and re-run.

- [ ] **Step 7: Clean up Chrome + commit**

```bash
pkill -f "remote-debugging-port=9222" 2>/dev/null
git add docs/business-revision/evidence/dashboards/ docs/business-revision/evidence/capture-dashboard.py
git commit -m "docs(revision): dashboard evidence for both contrasting orgs"
```

---

## Task 3: Capture report (PDF) evidence — both orgs

**Files:**
- Create: `docs/business-revision/evidence/reports/techflow-report.pdf`
- Create: `docs/business-revision/evidence/reports/balanced-report.pdf`

- [ ] **Step 1: Locate the PDF generation entry point**

```bash
grep -nE "def .*generate.*pdf|def build_report|class .*Report|def create_pdf" src/pdf_generator.py src/publication_report.py 2>/dev/null | head
```

Expected: the callable that produces a PDF from a flow matrix + metrics. Record its module path and signature.

- [ ] **Step 2: Write a headless PDF generation snippet**

Create `docs/business-revision/evidence/gen-report.py`. Fill the import + call using the exact entry point found in Step 1 (this example assumes `UlanowiczCalculator` + a report generator; adjust names to match the real signature):

```python
"""Generate a PDF report for one sample org without the UI."""
import sys, json, numpy as np
sys.path.insert(0, ".")
from src.ulanowicz_calculator import UlanowiczCalculator
# from src.pdf_generator import <RealGeneratorFound in Step 1>

path, out = sys.argv[1], sys.argv[2]
data = json.load(open(path))
flows = np.array(data["flows"]); nodes = data["nodes"]
calc = UlanowiczCalculator(flows, nodes)
metrics = calc.get_extended_metrics()
# TODO-REPLACE with the real generator call from Step 1, e.g.:
# generate_pdf_report(metrics, nodes, flows, org_name=data.get("organization","Org"), output_path=out)
print("Generated", out)
```

Replace the commented call with the real one discovered in Step 1 (exact function name, exact args). Remove the `TODO-REPLACE` line once done.

- [ ] **Step 3: Generate TechFlow PDF**

```bash
python3 docs/business-revision/evidence/gen-report.py \
  data/synthetic_organizations/combined_flows/tech_company_combined_matrix.json \
  docs/business-revision/evidence/reports/techflow-report.pdf
```

Expected: `Generated ...techflow-report.pdf` and the file exists (`ls -la` shows non-zero size).

- [ ] **Step 4: Generate Balanced PDF**

```bash
python3 docs/business-revision/evidence/gen-report.py \
  data/synthetic_organizations/combined_flows/balanced_org_test.json \
  docs/business-revision/evidence/reports/balanced-report.pdf
```

Expected: `Generated ...balanced-report.pdf`, non-zero size.

- [ ] **Step 5: Verify PDFs open and contain the expected sections**

Read the first ~10 pages of each PDF. Confirm they render (cover, sections, charts) and are not error stubs. If generation fails via script, fall back: generate both PDFs through the running app's export button (document the manual steps in `reports/README.md`).

- [ ] **Step 6: Commit**

```bash
git add docs/business-revision/evidence/reports/ docs/business-revision/evidence/gen-report.py
git commit -m "docs(revision): PDF report evidence for both contrasting orgs"
```

---

## Task 4: Three-lens specialist audit

**Files:**
- Create: `docs/business-revision/evidence/audit-uiux.md`
- Create: `docs/business-revision/evidence/audit-report.md`
- Create: `docs/business-revision/evidence/audit-pm.md`

The rubric (score each surface 1–5 per dimension): **1** Decision relevance (tiebreaker), **2** So-what clarity, **3** Interpretability, **4** Benchmark/context, **5** Credibility/defensibility, **6** Narrative flow, **7** Visual effectiveness.

- [ ] **Step 1: Dispatch the dashboard audit**

Use the Agent tool with `subagent_type: ui-ux-decision-maker`. Prompt (verbatim intent):

> "Audit the OASIS in-app dashboards for business utility from a strategy-consultant perspective. Evidence: screenshots at `docs/business-revision/evidence/dashboards/techflow-full.png` (unsustainable org) and `balanced-full.png` (viable org); surface list at `docs/business-revision/evidence/surface-inventory.md`. For EACH dashboard surface (D1–D9), score 1–5 on all 7 rubric dimensions (Decision relevance, So-what clarity, Interpretability, Benchmark/context, Credibility/defensibility, Narrative flow, Visual effectiveness). For each score below 4, give the specific evidence (what on the screenshot) and the business consequence. Do NOT propose formula changes. Write results as a Markdown table to `docs/business-revision/evidence/audit-uiux.md` and return a one-paragraph summary of the top 3 gaps."

- [ ] **Step 2: Dispatch the report audit**

Use the Agent tool with `subagent_type: sustainability-reporting-auditor`. Prompt:

> "Audit the OASIS PDF report for business utility and audit-firm credibility. Evidence: `docs/business-revision/evidence/reports/techflow-report.pdf` and `balanced-report.pdf`; surface list at `surface-inventory.md`. For EACH report surface (R1–R7), score 1–5 on all 7 rubric dimensions (same rubric). Emphasize Credibility/defensibility and framework alignment (ESRS/GRI/TCFD). For each score below 4, give specific evidence and business consequence. No formula changes. Write results to `docs/business-revision/evidence/audit-report.md` and return the top 3 gaps."

- [ ] **Step 3: Dispatch the value-chain audit**

Use the Agent tool with `subagent_type: ecosystem-pm`. Prompt:

> "Audit both OASIS surfaces (dashboards AND PDF report) for the operator→executive value chain. Evidence: `dashboards/*.png` and `reports/*.pdf`. Focus on Decision relevance (the tiebreaker), So-what clarity, and whether a consultant/sustainability-lead could hand this to a C-suite exec without translation. Score each surface 1–5 on Decision relevance and So-what clarity, and flag any surface that fails the diagnose-&-benchmark job. Write results to `docs/business-revision/evidence/audit-pm.md` and return the 3 highest-leverage gaps."

- [ ] **Step 4: Verify all three audit files exist and are populated**

```bash
for f in audit-uiux audit-report audit-pm; do
  echo "== $f =="; wc -l docs/business-revision/evidence/$f.md
done
```

Expected: each file exists with a substantive score table (dozens of lines, no empty tables).

- [ ] **Step 5: Commit**

```bash
git add docs/business-revision/evidence/audit-uiux.md docs/business-revision/evidence/audit-report.md docs/business-revision/evidence/audit-pm.md
git commit -m "docs(revision): three-lens specialist audit results"
```

---

## Task 5: Synthesize scored matrix + gap heatmap

**Files:**
- Create: `docs/business-revision/evidence/scored-matrix.md`

- [ ] **Step 1: Build the reconciled matrix**

Read all three audit files. Create `docs/business-revision/evidence/scored-matrix.md` with one row per surface (D1–D9, R1–R7) and one column per rubric dimension (1–7), each cell the reconciled 1–5 score. Where two agents scored the same surface/dimension differently, take the lower score and add a footnote noting the disagreement and why. Structure:

```markdown
# Scored Matrix (surface × 7 dimensions)

Legend: 1 = fails badly · 5 = consultant-grade. Cells ≤2 are gaps.

| Surface | 1 DecRel | 2 SoWhat | 3 Interp | 4 Bench | 5 Credib | 6 Narr | 7 Visual | Avg |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| D1 Core Metrics | | | | | | | | |
| ... | | | | | | | | |
| R7 Glossary | | | | | | | | |
```

- [ ] **Step 2: Add the gap heatmap + ranked gap list**

Append to the same file: (a) a heatmap rendered in Markdown using emoji bands (🟥 ≤2, 🟨 3, 🟩 ≥4) so the pattern is visible at a glance; (b) a ranked list of the top 8–10 gaps sorted by lowest score × surface prominence, each with surface, failing dimension(s), evidence pointer, and business consequence.

- [ ] **Step 3: Verify every surface is scored**

Confirm the matrix has a row for every ID in `surface-inventory.md` and no blank cells. Run:

```bash
grep -cE "^\| [DR][0-9]" docs/business-revision/evidence/scored-matrix.md
```

Expected: equals the number of surfaces in the inventory (D + R rows).

- [ ] **Step 4: Commit**

```bash
git add docs/business-revision/evidence/scored-matrix.md
git commit -m "docs(revision): reconciled scored matrix + gap heatmap"
```

---

## Task 6: Resolve the benchmarking-basis model

**Files:**
- Modify: `docs/business-revision/evidence/scored-matrix.md` (reference only)
- Content feeds Task 8 §4

- [ ] **Step 1: Confirm Tier-1 theoretical thresholds from the code**

Verify the actual viability/robustness thresholds used, so the recommendation cites real numbers:

```bash
grep -nE "0\.2|0\.6|0\.37|window.*viab|robust|viable" src/ulanowicz_calculator.py | head -20
```

Expected: the efficiency band (≈20–60%) and robustness optimum (≈37%) constants. Record exact values.

- [ ] **Step 2: Inventory Tier-2 reference datasets available as anchors**

```bash
ls data/ecosystem_samples/*.json | wc -l; ls data/ecosystem_samples/ | head -20
```

Expected: the count and names of real-world networks usable as illustrative anchors.

- [ ] **Step 3: Draft the benchmarking model section**

Write a self-contained section (to be pasted into the deliverable in Task 8) covering: Tier 1 (theoretical, now) with the exact bands from Step 1; Tier 2 (reference anchors, near-term) naming datasets from Step 2 and the apples-to-oranges caveat; Tier 3 (peer cohort, deferred) with the explicit data-acquisition gap and why fake peer benchmarks are rejected. For each of the 5–6 headline metrics (TST, AMI, Ascendency, Robustness, efficiency ratio, OASIS score), specify: the band, the on-screen label, and the one-sentence "so-what." Save as `docs/business-revision/evidence/benchmarking-model.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/business-revision/evidence/benchmarking-model.md
git commit -m "docs(revision): layered benchmarking model with per-metric contextualization"
```

---

## Task 7: Build the Impact × Effort redesign roadmap

**Files:**
- Create: `docs/business-revision/evidence/roadmap.md`

- [ ] **Step 1: Convert each gap into a recommendation**

Read `scored-matrix.md` gap list and `benchmarking-model.md`. For each gap, write a recommendation with: the fix (presentation/IA/narrative only), business impact (High/Med/Low, weighted by Decision relevance), and effort (presentation tweak vs. structural IA change).

- [ ] **Step 2: Sort into three horizons**

Create `docs/business-revision/evidence/roadmap.md`:

```markdown
# Redesign Roadmap (Impact × Effort)

## Immediate (high-impact, low-effort)
| # | Recommendation | Surface(s) | Impact | Effort |
|---|----------------|-----------|:------:|:------:|

## Short-term (high-impact, moderate-effort)
| # | Recommendation | Surface(s) | Impact | Effort |
|---|----------------|-----------|:------:|:------:|

## Medium-term (high-impact, higher-effort)
| # | Recommendation | Surface(s) | Impact | Effort |
|---|----------------|-----------|:------:|:------:|
```

Populate every row from Step 1. Each recommendation must trace to at least one gap ID from the scored matrix.

- [ ] **Step 3: Add the formula-guardrail check**

Append a short subsection listing any finding that *seemed* to need a formula change, and confirm it was reframed as a presentation fix or flagged for the `formula-validator` path — never actioned here. If none, state "No recommendation touches a scientific formula."

- [ ] **Step 4: Verify traceability**

Confirm every recommendation references a gap ID and no recommendation proposes a math change. Manually scan; then:

```bash
grep -iE "formula|coefficient|equation|change the (calc|metric)" docs/business-revision/evidence/roadmap.md
```

Expected: only guardrail-context mentions, no actioned formula edits.

- [ ] **Step 5: Commit**

```bash
git add docs/business-revision/evidence/roadmap.md
git commit -m "docs(revision): Impact x Effort redesign roadmap across three horizons"
```

---

## Task 8: Assemble the Business Revision document

**Files:**
- Create: `docs/business-revision/2026-07-02-oasis-business-revision.md`

- [ ] **Step 1: Write the document shell with all six sections**

Create `docs/business-revision/2026-07-02-oasis-business-revision.md` following the spec §8 structure. Section skeleton (fill each from the evidence files, do not leave placeholders):

```markdown
# OASIS — Business Revision

## 1. Executive Summary
_One page: is OASIS consultant-ready today? The 3–5 headline gaps. The redesign thesis._

## 2. Method & Rubric
_Scope, the 7 dimensions, the two contrasting orgs (TechFlow = unsustainable, Balanced = viable)._

## 3. Findings
_The scored matrix + gap heatmap (embed from scored-matrix.md), with evidence pointers and business consequence per gap._

## 4. Benchmarking Strategy
_Tier 1/2/3 model + per-metric contextualization (from benchmarking-model.md)._

## 5. Redesign Roadmap
_Immediate / Short-term / Medium-term (from roadmap.md)._

## 6. Appendix
_Full per-surface scores; links to agent audit notes and evidence artifacts._
```

- [ ] **Step 2: Write the Executive Summary last, from the assembled body**

After sections 2–6 are filled, write section 1 as a true one-page synthesis: the verdict, the 3–5 highest-leverage gaps (pulled from the roadmap's Immediate + Short-term), and the one-line redesign thesis. Reference embedded screenshots (`evidence/dashboards/*.png`) for the marquee gap.

- [ ] **Step 3: Verify no placeholders and full spec coverage**

```bash
grep -nE "TBD|TODO|_fill|placeholder|LINE\b" docs/business-revision/2026-07-02-oasis-business-revision.md
```

Expected: no matches. Then manually confirm each spec §8 item (1–6) maps to a written section, and that sections 3/4/5 actually embed the synthesized content (not just link to it).

- [ ] **Step 4: Commit**

```bash
git add docs/business-revision/2026-07-02-oasis-business-revision.md
git commit -m "docs(revision): assemble OASIS Business Revision deliverable"
```

---

## Task 9: Final review, PDF export, and handoff note

**Files:**
- Create: `docs/business-revision/evidence/reports/business-revision.pdf` (optional export)
- Modify: `docs/business-revision/2026-07-02-oasis-business-revision.md` (handoff subsection)

- [ ] **Step 1: Cross-check deliverable against success criteria**

Re-read spec §9. For each criterion (all surfaces scored for both orgs; findings evidence-backed; benchmarking model recommended; recs prioritized; no formula changes; hand-off-ready), confirm the deliverable satisfies it. Note any miss and fix inline.

- [ ] **Step 2: Export the document to PDF (best-effort)**

If a Markdown→PDF tool is available:

```bash
command -v pandoc && pandoc docs/business-revision/2026-07-02-oasis-business-revision.md \
  -o docs/business-revision/evidence/reports/business-revision.pdf 2>&1 | tail -3 || echo "pandoc not available — skip, Markdown is the source of truth"
```

Expected: a PDF is produced, or a clear skip message. Markdown remains the canonical deliverable either way.

- [ ] **Step 3: Add a handoff subsection**

Append to the deliverable an appendix subsection "Handoff": each roadmap recommendation becomes its own downstream spec→plan→build cycle; this review delivers the plan, not the implementation. List the Immediate-horizon items as the recommended first follow-on specs.

- [ ] **Step 4: Final commit**

```bash
git add docs/business-revision/
git commit -m "docs(revision): final review, PDF export, and downstream handoff note"
```

- [ ] **Step 5: Report completion**

Summarize for the user: the verdict, the top 3 gaps, and the recommended first Immediate-horizon action — with the deliverable path.
