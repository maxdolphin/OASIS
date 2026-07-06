# Detailed Ecosystemic Sustainability Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the generated OASIS PDF report substantially more detailed and thorough by adding Benchmarking, Risk & Resilience, Prioritized Action Roadmap, and ESG-framework-mapping sections — built entirely on metrics that are already computed.

**Architecture:** A new pure-Python module `src/report_intelligence.py` transforms the existing OASIS profile + Ulanowicz metrics + recommendations into structured content (no HTML, no new scientific formulas). `src/oasis_pdf_report.py` gains new `_build_*` section methods that render that content and is wired into `generate_html()` behind a backward-compatible `detailed` flag. One new matplotlib chart (Window of Viability with the org's point) is rendered to PNG bytes for the Benchmarking section.

**Tech Stack:** Python 3, numpy, matplotlib (already deps), WeasyPrint (PDF), pytest (tests). Reuses `OASISCalculator`, `UlanowiczCalculator`, and `src/services/published_metrics_db.py`.

---

## Constraints (read before starting)

- **No new scientific formulas.** Every numeric value displayed must come from an existing computed metric, an existing codebase constant (Window-of-Viability band `[0.2, 0.6]`, robustness optimum `α ≈ 0.367879` = 1/e), or a published reference value in `published_metrics_db.py`. The robustness curve `R(α) = -α·ln(α)` is already used in the engine and may be plotted, but must not be re-derived or altered.
- **Total functions.** Every `report_intelligence` function must use `.get(key, default)` and never raise on a sparse metric/profile dict.
- **Backward compatible.** The current lean report must remain reproducible via `detailed=False`.
- Use git identity `Massimo Mistretta <maxdolphin@gmail.com>` for all commits (configure with `git -c user.email=... -c user.name=...` or rely on repo config).

## Existing data contracts (verified — do not re-discover)

`OASISCalculator.get_oasis_profile()` returns:
```python
{
  'dimension_scores':  {'open': float, 'autonomous': float, 'symbiotic': float, 'intelligent': float, 'sustainable': float},  # 0-100
  'dimension_details': {dim: {'metrics': {...}, 'weights': {...}, ...}},   # sustainable.metrics has 'relative_ascendency', 'robustness', 'is_viable'
  'overall_score':  float,            # 0-100
  'weights':        {dim: float},
  'dimension_status':  {dim: 'HEALTHY'|'WARNING'|'CRITICAL'},
  'overall_status':    'HEALTHY'|'WARNING'|'CRITICAL',
}
```
`OASISCalculator.get_recommendations()` returns a priority-sorted list of:
```python
{'priority': 'CRITICAL'|'HIGH'|'MEDIUM'|'LOW', 'dimension': 'OPEN'|..., 'issue': str, 'action': str, 'metrics_to_improve': [str, ...]}
```
`UlanowiczCalculator.get_extended_metrics()` returns a dict including: `total_system_throughput`, `average_mutual_information`, `ascendency`, `development_capacity`, `overhead`, `ascendency_ratio` (this is α), `overhead_ratio`, `robustness`, `redundancy`, `is_viable`, `connectance`, `effective_link_density`, `flow_diversity`, `trophic_depth`.

`src/services/published_metrics_db.py`: `list_networks() -> List[str]`, `get_network_info(network_id) -> Optional[Dict]`, `get_published_metric(network_id, metric_name) -> Optional[float]`. Reference networks include `cone_spring_original` (relative_ascendency 0.505), `cone_spring_eutrophicated` (0.529), `crystal_river_creek`.

---

## File Structure

- **Create** `src/report_intelligence.py` — pure content-synthesis functions + WoV chart renderer.
- **Create** `tests/test_report_intelligence.py` — unit tests for the above.
- **Create** `tests/test_report_sections.py` — smoke test for the assembled report.
- **Modify** `src/oasis_pdf_report.py` — new `_build_*` methods; `generate_html()` ordering; `detailed` flag on `OASISPDFReport.__init__` and `generate_oasis_pdf_report()`.

---

### Task 1: Module scaffold + constants + `executive_verdict`

**Files:**
- Create: `src/report_intelligence.py`
- Test: `tests/test_report_intelligence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_intelligence.py
from src import report_intelligence as ri


def _profile(overall=72.0, status='HEALTHY'):
    return {
        'dimension_scores': {'open': 70, 'autonomous': 55, 'symbiotic': 80,
                             'intelligent': 60, 'sustainable': 78},
        'dimension_status': {'open': 'HEALTHY', 'autonomous': 'WARNING',
                             'symbiotic': 'HEALTHY', 'intelligent': 'WARNING',
                             'sustainable': 'HEALTHY'},
        'dimension_details': {'sustainable': {'metrics': {
            'relative_ascendency': 0.42, 'robustness': 0.36, 'is_viable': True}}},
        'overall_score': overall, 'overall_status': status,
        'weights': {'open': 0.2, 'autonomous': 0.2, 'symbiotic': 0.2,
                    'intelligent': 0.2, 'sustainable': 0.2},
    }


def test_constants_match_codebase_window():
    assert ri.VIABILITY_LOWER == 0.2
    assert ri.VIABILITY_UPPER == 0.6
    assert abs(ri.ROBUSTNESS_OPTIMUM - 0.367879441) < 1e-6


def test_executive_verdict_mentions_score_and_status():
    v = ri.executive_verdict(_profile(overall=72.0, status='HEALTHY'))
    assert '72' in v
    assert 'HEALTHY' in v.upper()


def test_executive_verdict_handles_empty_profile():
    # Must not raise on a sparse profile
    assert isinstance(ri.executive_verdict({}), str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'src.report_intelligence' has no attribute ...`

- [ ] **Step 3: Write minimal implementation**

```python
# src/report_intelligence.py
"""
Report intelligence: deterministic synthesis of OASIS profile + Ulanowicz metrics
into structured content for the detailed report.

IMPORTANT: This module contains NO scientific formulas. It classifies, sequences,
and looks up values that are already computed elsewhere. The only constants are the
Window-of-Viability bounds and the robustness optimum, which are existing codebase
constants (Ulanowicz; alpha = 1/e maximizes R = -alpha*ln(alpha)).
"""
from typing import Any, Dict, List

# Window of Viability — existing engine constants (Ulanowicz et al. 2009)
VIABILITY_LOWER = 0.2
VIABILITY_UPPER = 0.6
ROBUSTNESS_OPTIMUM = 0.367879441  # 1/e, where R = -alpha*ln(alpha) is maximal


def _alpha(profile: Dict[str, Any], metrics: Dict[str, Any] = None) -> float:
    """Read relative ascendency (alpha) from metrics, falling back to profile."""
    if metrics and 'ascendency_ratio' in metrics:
        return float(metrics.get('ascendency_ratio', 0.0))
    return float(profile.get('dimension_details', {})
                 .get('sustainable', {}).get('metrics', {})
                 .get('relative_ascendency', 0.0))


def executive_verdict(profile: Dict[str, Any]) -> str:
    """One-sentence plain-language overall verdict for the executive layer."""
    score = float(profile.get('overall_score', 0.0))
    status = str(profile.get('overall_status', 'UNKNOWN'))
    scores = profile.get('dimension_scores', {})
    if scores:
        weakest = min(scores, key=scores.get)
        tail = f" The weakest dimension is {weakest.upper()} ({scores[weakest]:.0f}/100)."
    else:
        tail = ""
    return (f"Overall organizational health is {score:.0f}/100 ({status})."
            f"{tail}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/report_intelligence.py tests/test_report_intelligence.py
git commit -m "feat(report): scaffold report_intelligence with verdict + viability constants"
```

---

### Task 2: `build_benchmark_view`

**Files:**
- Modify: `src/report_intelligence.py`
- Test: `tests/test_report_intelligence.py`

- [ ] **Step 1: Write the failing test** (append to the test file)

```python
def _metrics(alpha=0.42, robustness=0.36):
    return {'ascendency_ratio': alpha, 'robustness': robustness,
            'development_capacity': 100.0, 'ascendency': 42.0,
            'overhead_ratio': 1 - alpha, 'redundancy': 0.5}


def test_benchmark_view_position_in_window():
    v = ri.build_benchmark_view(_metrics(alpha=0.42), _profile())
    assert v['alpha'] == 0.42
    assert v['in_window'] is True
    assert v['lower'] == 0.2 and v['upper'] == 0.6
    assert abs(v['distance_to_optimum'] - abs(0.42 - ri.ROBUSTNESS_OPTIMUM)) < 1e-9
    assert isinstance(v['reference_anchors'], list)


def test_benchmark_view_out_of_window_rigid():
    v = ri.build_benchmark_view(_metrics(alpha=0.7), _profile())
    assert v['in_window'] is False
    assert v['position'] == 'above'  # too rigid / over-organized


def test_benchmark_view_handles_missing_metrics():
    v = ri.build_benchmark_view({}, {})
    assert 'alpha' in v and 'reference_anchors' in v
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_benchmark_view'`

- [ ] **Step 3: Write minimal implementation** (append to module)

```python
def build_benchmark_view(metrics: Dict[str, Any],
                         profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Position the organization against the Window of Viability and published
    ecological reference points. No new computation — reads existing alpha/robustness
    and looks up published relative_ascendency values.
    """
    alpha = _alpha(profile, metrics)
    robustness = float(metrics.get('robustness',
                       profile.get('dimension_details', {}).get('sustainable', {})
                       .get('metrics', {}).get('robustness', 0.0)))

    if alpha < VIABILITY_LOWER:
        position = 'below'
    elif alpha > VIABILITY_UPPER:
        position = 'above'
    else:
        position = 'within'

    anchors = _reference_anchors()

    return {
        'alpha': alpha,
        'robustness': robustness,
        'lower': VIABILITY_LOWER,
        'upper': VIABILITY_UPPER,
        'optimum': ROBUSTNESS_OPTIMUM,
        'in_window': VIABILITY_LOWER <= alpha <= VIABILITY_UPPER,
        'position': position,
        'distance_to_optimum': abs(alpha - ROBUSTNESS_OPTIMUM),
        'reference_anchors': anchors,
    }


def _reference_anchors() -> List[Dict[str, Any]]:
    """Published ecological reference points (labelled, NOT organizational targets)."""
    try:
        from src.services import published_metrics_db as pdb
    except Exception:
        return []
    anchors = []
    for net_id in pdb.list_networks():
        ra = pdb.get_published_metric(net_id, 'relative_ascendency')
        if ra is None:
            continue
        info = pdb.get_network_info(net_id) or {}
        anchors.append({
            'id': net_id,
            'label': net_id.replace('_', ' ').title(),
            'relative_ascendency': float(ra),
            'source': info.get('source', ''),
            'note': 'Scientific reference point, not an organizational target.',
        })
    return anchors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/report_intelligence.py tests/test_report_intelligence.py
git commit -m "feat(report): add build_benchmark_view with viability position + reference anchors"
```

---

### Task 3: `build_risk_view`

**Files:**
- Modify: `src/report_intelligence.py`
- Test: `tests/test_report_intelligence.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_risk_view_brittle_when_alpha_high():
    v = ri.build_risk_view(_metrics(alpha=0.72), _profile())
    assert v['fragility'] == 'over-organized'
    assert any('rigid' in item['title'].lower() or 'brittle' in item['title'].lower()
               for item in v['items'])


def test_risk_view_chaotic_when_alpha_low():
    v = ri.build_risk_view(_metrics(alpha=0.12), _profile())
    assert v['fragility'] == 'under-organized'


def test_risk_view_balanced_in_window():
    v = ri.build_risk_view(_metrics(alpha=0.4), _profile())
    assert v['fragility'] == 'balanced'


def test_risk_view_flags_critical_dimensions():
    prof = _profile()
    prof['dimension_status']['autonomous'] = 'CRITICAL'
    v = ri.build_risk_view(_metrics(alpha=0.4), prof)
    assert any(it['severity'] == 'CRITICAL' for it in v['items'])


def test_risk_view_handles_empty():
    v = ri.build_risk_view({}, {})
    assert 'fragility' in v and isinstance(v['items'], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: FAIL — no attribute `build_risk_view`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def build_risk_view(metrics: Dict[str, Any],
                    profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fragility/resilience narrative built from existing alpha, overhead, redundancy,
    and per-dimension status. No new computation.
    """
    alpha = _alpha(profile, metrics)
    overhead_ratio = float(metrics.get('overhead_ratio', 0.0))
    redundancy = float(metrics.get('redundancy', 0.0))

    if alpha < VIABILITY_LOWER:
        fragility = 'under-organized'
    elif alpha > VIABILITY_UPPER:
        fragility = 'over-organized'
    else:
        fragility = 'balanced'

    items: List[Dict[str, Any]] = []

    if fragility == 'over-organized':
        items.append({
            'severity': 'HIGH',
            'title': 'System is rigid / brittle (over-organized)',
            'evidence': f'Relative ascendency alpha = {alpha:.3f} exceeds the upper '
                        f'viability bound ({VIABILITY_UPPER}).',
            'implication': 'Low adaptive reserve; efficient but vulnerable to shocks '
                           'and unexpected change.',
        })
    elif fragility == 'under-organized':
        items.append({
            'severity': 'HIGH',
            'title': 'System is chaotic (under-organized)',
            'evidence': f'Relative ascendency alpha = {alpha:.3f} is below the lower '
                        f'viability bound ({VIABILITY_LOWER}).',
            'implication': 'Abundant redundancy but weak coordination; activity may '
                           'not translate into reliable outcomes.',
        })
    else:
        items.append({
            'severity': 'LOW',
            'title': 'System operates within the Window of Viability',
            'evidence': f'Relative ascendency alpha = {alpha:.3f} lies within '
                        f'[{VIABILITY_LOWER}, {VIABILITY_UPPER}].',
            'implication': 'Healthy balance of efficiency and resilience; maintain and '
                           'monitor.',
        })

    # Distance-from-bound early warnings (within window but near an edge)
    if fragility == 'balanced':
        if (alpha - VIABILITY_LOWER) < 0.05:
            items.append({'severity': 'MEDIUM',
                          'title': 'Approaching lower viability bound',
                          'evidence': f'alpha = {alpha:.3f} is within 0.05 of {VIABILITY_LOWER}.',
                          'implication': 'Trend toward disorganization warrants monitoring.'})
        if (VIABILITY_UPPER - alpha) < 0.05:
            items.append({'severity': 'MEDIUM',
                          'title': 'Approaching upper viability bound',
                          'evidence': f'alpha = {alpha:.3f} is within 0.05 of {VIABILITY_UPPER}.',
                          'implication': 'Trend toward rigidity warrants monitoring.'})

    # Per-dimension status escalation
    for dim, status in profile.get('dimension_status', {}).items():
        if status in ('CRITICAL', 'WARNING'):
            items.append({
                'severity': status,
                'title': f'{dim.upper()} dimension flagged {status}',
                'evidence': f'OASIS {dim} status = {status}.',
                'implication': 'Targeted intervention recommended — see Action Roadmap.',
            })

    sev_order = {'CRITICAL': 0, 'HIGH': 1, 'WARNING': 2, 'MEDIUM': 3, 'LOW': 4}
    items.sort(key=lambda it: sev_order.get(it['severity'], 5))

    return {
        'fragility': fragility,
        'overhead_ratio': overhead_ratio,
        'redundancy': redundancy,
        'items': items,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/report_intelligence.py tests/test_report_intelligence.py
git commit -m "feat(report): add build_risk_view fragility + resilience analysis"
```

---

### Task 4: `build_action_roadmap`

**Files:**
- Modify: `src/report_intelligence.py`
- Test: `tests/test_report_intelligence.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def _recs():
    return [
        {'priority': 'CRITICAL', 'dimension': 'SUSTAINABLE', 'issue': 'Too rigid',
         'action': 'Diversify pathways', 'metrics_to_improve': ['redundancy']},
        {'priority': 'HIGH', 'dimension': 'OPEN', 'issue': 'Low interconnectivity',
         'action': 'Add cross-functional channels', 'metrics_to_improve': ['connectance']},
        {'priority': 'MEDIUM', 'dimension': 'SYMBIOTIC', 'issue': 'Inequality',
         'action': 'Redistribute resources', 'metrics_to_improve': ['gini_coefficient']},
    ]


def test_roadmap_buckets_by_horizon():
    r = ri.build_action_roadmap(_recs(), _profile())
    assert len(r['immediate']) == 1 and r['immediate'][0]['dimension'] == 'SUSTAINABLE'
    assert len(r['short_term']) == 1 and r['short_term'][0]['dimension'] == 'OPEN'
    assert len(r['medium_term']) == 1


def test_roadmap_items_carry_expected_impact():
    r = ri.build_action_roadmap(_recs(), _profile())
    assert 'expected_impact' in r['immediate'][0]


def test_roadmap_handles_no_recs():
    r = ri.build_action_roadmap([], _profile())
    assert r['immediate'] == [] and r['short_term'] == [] and r['medium_term'] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: FAIL — no attribute `build_action_roadmap`

- [ ] **Step 3: Write minimal implementation** (append)

```python
# Qualitative expected-impact phrasing per dimension (lookup, NOT a scoring model)
_IMPACT_BY_DIMENSION = {
    'OPEN': 'Improves interconnectivity and information circulation across units.',
    'AUTONOMOUS': 'Strengthens feedback loops and institutional learning.',
    'SYMBIOTIC': 'Rebalances resource distribution and cooperation.',
    'INTELLIGENT': 'Increases functional diversity and specialization.',
    'SUSTAINABLE': 'Moves the system toward the Window of Viability (efficiency/'
                   'resilience balance).',
}


def build_action_roadmap(recommendations: List[Dict[str, Any]],
                         profile: Dict[str, Any]) -> Dict[str, Any]:
    """Sequence existing recommendations into Immediate/Short/Medium-term horizons."""
    horizons = {'immediate': [], 'short_term': [], 'medium_term': []}
    bucket = {'CRITICAL': 'immediate', 'HIGH': 'short_term',
              'MEDIUM': 'medium_term', 'LOW': 'medium_term'}
    for rec in recommendations or []:
        prio = rec.get('priority', 'MEDIUM')
        dim = rec.get('dimension', 'N/A')
        item = {
            'priority': prio,
            'dimension': dim,
            'issue': rec.get('issue', ''),
            'action': rec.get('action', ''),
            'metrics_to_improve': rec.get('metrics_to_improve', []),
            'expected_impact': _IMPACT_BY_DIMENSION.get(dim, 'Improves overall health.'),
        }
        horizons[bucket.get(prio, 'medium_term')].append(item)
    return horizons
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/report_intelligence.py tests/test_report_intelligence.py
git commit -m "feat(report): add build_action_roadmap horizon sequencing"
```

---

### Task 5: `build_esg_crosswalk`

**Files:**
- Modify: `src/report_intelligence.py`
- Test: `tests/test_report_intelligence.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_esg_crosswalk_covers_all_dimensions():
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    dims = {row['oasis_dimension'] for row in rows}
    assert {'OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE'} <= dims


def test_esg_crosswalk_rows_have_framework_refs():
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    r = rows[0]
    assert all(k in r for k in ('gri_ref', 'esrs_ref', 'tcfd_ref', 'finding_summary'))


def test_esg_crosswalk_handles_empty_profile():
    rows = ri.build_esg_crosswalk({}, {})
    assert len(rows) == 5  # still emits one indicative row per dimension
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: FAIL — no attribute `build_esg_crosswalk`

- [ ] **Step 3: Write minimal implementation** (append)

```python
# Indicative qualitative crosswalk (navigation/credibility aid, NOT a compliance map)
_ESG_CROSSWALK = {
    'OPEN':        {'gri': 'GRI 2-9/2-29 (governance, stakeholder engagement)',
                    'esrs': 'ESRS 2 GOV/SBM (strategy & stakeholder interaction)',
                    'tcfd': 'Governance (board oversight of interconnected risks)',
                    'theme': 'interconnectivity and information circulation'},
    'AUTONOMOUS':  {'gri': 'GRI 3-3 (management of material topics)',
                    'esrs': 'ESRS 2 IRO (impact, risk & opportunity management)',
                    'tcfd': 'Risk Management (processes to identify/learn)',
                    'theme': 'organizational learning and feedback'},
    'SYMBIOTIC':   {'gri': 'GRI 3-3 / 207 (equitable value distribution)',
                    'esrs': 'ESRS S/G (own workforce, business conduct)',
                    'tcfd': 'Strategy (resource dependencies)',
                    'theme': 'resource equity and mutualism'},
    'INTELLIGENT': {'gri': 'GRI 2-17 (collective knowledge of governance body)',
                    'esrs': 'ESRS 2 GOV (skills/expertise of administrative bodies)',
                    'tcfd': 'Governance (competencies to assess risk)',
                    'theme': 'functional diversity and capability'},
    'SUSTAINABLE': {'gri': 'GRI 201-2 (financial implications/risks of change)',
                    'esrs': 'ESRS 2 SBM-3 (resilience of strategy & business model)',
                    'tcfd': 'Strategy — Resilience (scenario/long-term viability)',
                    'theme': 'efficiency/resilience balance (Window of Viability)'},
}


def build_esg_crosswalk(profile: Dict[str, Any],
                        metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Indicative crosswalk from OASIS findings to GRI/ESRS-CSRD/TCFD disclosure areas.
    Qualitative navigation aid only — NOT a compliance attestation.
    """
    scores = profile.get('dimension_scores', {})
    status = profile.get('dimension_status', {})
    rows = []
    for dim in ['OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE']:
        cw = _ESG_CROSSWALK[dim]
        key = dim.lower()
        sc = scores.get(key)
        stt = status.get(key, 'N/A')
        finding = (f"{dim.title()} ({cw['theme']}): "
                   + (f"score {sc:.0f}/100, status {stt}." if sc is not None
                      else "not assessed."))
        rows.append({
            'oasis_dimension': dim,
            'finding_summary': finding,
            'gri_ref': cw['gri'],
            'esrs_ref': cw['esrs'],
            'tcfd_ref': cw['tcfd'],
        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/report_intelligence.py tests/test_report_intelligence.py
git commit -m "feat(report): add indicative ESG (GRI/ESRS/TCFD) crosswalk"
```

---

### Task 6: Window-of-Viability chart renderer (PNG bytes)

**Files:**
- Modify: `src/report_intelligence.py`
- Test: `tests/test_report_intelligence.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_wov_chart_returns_png_bytes():
    png = ri.render_window_of_viability_png(alpha=0.42, robustness=0.36)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic number


def test_wov_chart_handles_zero_alpha():
    png = ri.render_window_of_viability_png(alpha=0.0, robustness=0.0)
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: FAIL — no attribute `render_window_of_viability_png`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def render_window_of_viability_png(alpha: float, robustness: float) -> bytes:
    """
    Render the robustness curve R(alpha) = -alpha*ln(alpha) with the organization's
    (alpha, robustness) point and the viability band shaded. Returns PNG bytes.

    The curve R = -alpha*ln(alpha) is the existing engine robustness definition
    (Ulanowicz et al. 2009); it is plotted, not re-derived. Light theme for print.
    """
    import io
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = np.linspace(0.001, 0.999, 400)
    ys = -xs * np.log(xs)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.plot(xs, ys, color='#1a5f35', linewidth=2, label='Robustness R(α) = −α·ln(α)')
    ax.axvspan(VIABILITY_LOWER, VIABILITY_UPPER, color='#48c9b0', alpha=0.15,
               label=f'Window of Viability [{VIABILITY_LOWER}, {VIABILITY_UPPER}]')
    ax.axvline(ROBUSTNESS_OPTIMUM, color='#d4a843', linestyle='--', linewidth=1,
               label=f'Optimum α ≈ {ROBUSTNESS_OPTIMUM:.2f}')

    a = max(0.0, min(1.0, float(alpha)))
    r = float(robustness) if robustness else (-a * np.log(a) if 0 < a < 1 else 0.0)
    ax.scatter([a], [r], color='#c0392b', s=90, zorder=5,
               label=f'This organization (α={a:.3f})')

    ax.set_xlabel('Relative Ascendency (α = A/C)')
    ax.set_ylabel('Robustness (R)')
    ax.set_title('Position Relative to the Window of Viability')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    return buf.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_intelligence.py -q`
Expected: PASS (all report_intelligence tests green)

- [ ] **Step 5: Commit**

```bash
git add src/report_intelligence.py tests/test_report_intelligence.py
git commit -m "feat(report): add Window-of-Viability chart renderer (PNG)"
```

---

### Task 7: New report sections in `OASISPDFReport` + `detailed` flag

**Files:**
- Modify: `src/oasis_pdf_report.py`
- Test: `tests/test_report_sections.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_sections.py
import numpy as np
from src.ulanowicz_calculator import UlanowiczCalculator
from src.oasis_calculator import OASISCalculator
from src.oasis_pdf_report import OASISPDFReport


def _build_report(detailed=True):
    flow = np.array([
        [0, 10, 0, 0, 5],
        [0, 0, 8, 2, 0],
        [0, 0, 0, 7, 1],
        [3, 0, 0, 0, 6],
        [0, 4, 0, 0, 0],
    ], dtype=float)
    nodes = ['A', 'B', 'C', 'D', 'E']
    uc = UlanowiczCalculator(flow, nodes)
    oc = OASISCalculator(uc)
    return OASISPDFReport(
        org_name='Test Org',
        oasis_profile=oc.get_oasis_profile(),
        ulanowicz_metrics=uc.get_extended_metrics(),
        interpretations=oc.get_oasis_interpretation(),
        recommendations=oc.get_recommendations(),
        detailed=detailed,
    )


def test_detailed_report_contains_new_sections():
    html = _build_report(detailed=True).generate_html()
    assert 'Benchmarking' in html
    assert 'Risk &amp; Resilience' in html or 'Risk & Resilience' in html
    assert 'Action Roadmap' in html
    assert 'Framework Mapping' in html or 'ESG' in html


def test_lean_report_excludes_new_sections():
    html = _build_report(detailed=False).generate_html()
    assert 'Action Roadmap' not in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_sections.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'detailed'`

- [ ] **Step 3: Implement — add `detailed` param and new builders**

In `src/oasis_pdf_report.py`, modify `OASISPDFReport.__init__` signature (around line 241-251) to accept `detailed`:

```python
    def __init__(
        self,
        org_name: str,
        oasis_profile: Dict[str, Any],
        ulanowicz_metrics: Dict[str, Any],
        interpretations: Dict[str, str],
        recommendations: List[Dict[str, Any]],
        chart_images: Optional[Dict[str, bytes]] = None,
        logo_path: Optional[str] = None,
        analyst_name: str = "OASIS Analysis System",
        detailed: bool = True,
    ):
```

At the end of `__init__` body (after `self.page_number = 0`), add:

```python
        self.detailed = detailed
        # Lazily computed report-intelligence views (built on existing data only)
        from src import report_intelligence as _ri
        self._ri = _ri
        if detailed:
            self.benchmark = _ri.build_benchmark_view(self.metrics, self.profile)
            self.risk = _ri.build_risk_view(self.metrics, self.profile)
            self.roadmap = _ri.build_action_roadmap(self.recommendations, self.profile)
            self.esg = _ri.build_esg_crosswalk(self.profile, self.metrics)
            # Inject WoV chart into chart_images so existing chart pipeline renders it
            try:
                self.charts.setdefault(
                    'window_viability',
                    _ri.render_window_of_viability_png(
                        self.benchmark['alpha'], self.benchmark['robustness']))
            except Exception:
                pass
```

Add four new builder methods (place after `_build_executive_summary`, before `_build_methodology`):

```python
    def _build_benchmarking(self) -> str:
        """Benchmarking & position vs the Window of Viability and reference points."""
        b = self.benchmark
        pos_text = {
            'within': 'within the Window of Viability',
            'above': 'above the viability band (tending rigid / over-organized)',
            'below': 'below the viability band (tending chaotic / under-organized)',
        }.get(b['position'], 'undetermined')

        anchor_rows = ""
        for a in b['reference_anchors']:
            anchor_rows += f"""
            <tr>
                <td>{_escape(a['label'])}</td>
                <td class="numeric">{a['relative_ascendency']:.3f}</td>
                <td>{_escape(a['source'])}</td>
            </tr>"""
        if not anchor_rows:
            anchor_rows = '<tr><td colspan="3">No reference data available.</td></tr>'

        return f"""
        <div class="page-break"></div>
        <h1>2. Benchmarking &amp; Position</h1>
        <p>
            The organization's relative ascendency is
            <strong>α = {b['alpha']:.3f}</strong>, placing it {pos_text}
            (viable band {b['lower']}&ndash;{b['upper']}; robustness optimum
            α &asymp; {b['optimum']:.2f}). Distance to the robustness optimum is
            <strong>{b['distance_to_optimum']:.3f}</strong>.
        </p>
        <h2>2.1 Ecological Reference Points</h2>
        <p class="text-small text-muted">
            Published ecosystem values are shown as scientific reference points for the
            viability scale&mdash;not as organizational targets.
        </p>
        <table>
            <thead><tr><th>Reference Network</th>
                <th style="text-align:right;">Relative Ascendency (α)</th>
                <th>Source</th></tr></thead>
            <tbody>{anchor_rows}</tbody>
            <caption>Table 2. Published reference networks (relative ascendency).</caption>
        </table>
        """

    def _build_risk_resilience(self) -> str:
        """Risk & resilience analysis section."""
        r = self.risk
        items_html = ""
        for it in r['items']:
            sev = _escape(it['severity'])
            items_html += f"""
            <div class="recommendation priority-{sev.lower()}">
                <div class="recommendation-header">
                    <span class="recommendation-dimension">{_escape(it['title'])}</span>
                    <span class="priority-tag {sev.lower()}">{sev}</span>
                </div>
                <p style="margin:1mm 0;"><strong>Evidence:</strong> {_escape(it['evidence'])}</p>
                <p style="margin:1mm 0;"><strong>Implication:</strong> {_escape(it['implication'])}</p>
            </div>"""
        return f"""
        <div class="page-break"></div>
        <h1>3. Risk &amp; Resilience Analysis</h1>
        <p>
            Overall fragility classification: <strong>{_escape(r['fragility'])}</strong>.
            Adaptive reserve indicators &mdash; overhead ratio
            {r['overhead_ratio']*100:.1f}%, redundancy {r['redundancy']:.3f}.
        </p>
        {items_html}
        """

    def _build_action_roadmap(self) -> str:
        """Prioritized action roadmap section."""
        def horizon_html(title, items):
            if not items:
                return f"<h2>{title}</h2><p class='text-muted'>No actions in this horizon.</p>"
            rows = ""
            for it in items:
                rows += f"""
                <div class="recommendation priority-{it['priority'].lower()}">
                    <div class="recommendation-header">
                        <span class="recommendation-dimension">{_escape(it['dimension'])}</span>
                        <span class="priority-tag {it['priority'].lower()}">{_escape(it['priority'])}</span>
                    </div>
                    <p style="margin:1mm 0; font-weight:600;">{_escape(it['issue'])}</p>
                    <p style="margin:1mm 0;">{_escape(it['action'])}</p>
                    <p class="text-small text-muted">Expected impact: {_escape(it['expected_impact'])}<br>
                       Metrics to improve: {_escape(', '.join(it['metrics_to_improve']) or 'N/A')}</p>
                </div>"""
            return f"<h2>{title}</h2>{rows}"

        return f"""
        <div class="page-break"></div>
        <h1>4. Prioritized Action Roadmap</h1>
        {horizon_html('4.1 Immediate (0&ndash;3 months)', self.roadmap['immediate'])}
        {horizon_html('4.2 Short-Term (3&ndash;9 months)', self.roadmap['short_term'])}
        {horizon_html('4.3 Medium-Term (9&ndash;18 months)', self.roadmap['medium_term'])}
        """

    def _build_esg_mapping(self) -> str:
        """ESG framework mapping section (indicative)."""
        rows = ""
        for row in self.esg:
            rows += f"""
            <tr>
                <td><strong>{_escape(row['oasis_dimension'])}</strong><br>
                    <span class="text-small text-muted">{_escape(row['finding_summary'])}</span></td>
                <td>{_escape(row['gri_ref'])}</td>
                <td>{_escape(row['esrs_ref'])}</td>
                <td>{_escape(row['tcfd_ref'])}</td>
            </tr>"""
        return f"""
        <div class="page-break"></div>
        <h1>7. ESG Framework Mapping</h1>
        <p class="text-small text-muted">
            Indicative crosswalk linking OASIS findings to recognized disclosure
            frameworks. Provided for navigation and context only; not a compliance
            attestation.
        </p>
        <table>
            <thead><tr><th>OASIS Finding</th><th>GRI</th><th>ESRS / CSRD</th><th>TCFD</th></tr></thead>
            <tbody>{rows}</tbody>
            <caption>Table 7. Indicative OASIS-to-ESG framework crosswalk.</caption>
        </table>
        """
```

- [ ] **Step 4: Wire sections into `generate_html()`**

Replace the body assembly in `generate_html()` (around lines 1411-1422) with conditional inclusion:

```python
<body>

{self._build_cover_page()}
{self._build_executive_summary()}
{self._build_benchmarking() if self.detailed else ""}
{self._build_risk_resilience() if self.detailed else ""}
{self._build_action_roadmap() if self.detailed else ""}
{self._build_methodology()}
{self._build_results()}
{self._build_esg_mapping() if self.detailed else ""}
{self._build_discussion()}
{self._build_references()}
{self._build_appendix()}

</body>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_sections.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/oasis_pdf_report.py tests/test_report_sections.py
git commit -m "feat(report): add benchmarking, risk, roadmap, ESG sections behind detailed flag"
```

---

### Task 8: Backward-compatible `detailed` param on the convenience function

**Files:**
- Modify: `src/oasis_pdf_report.py` (function `generate_oasis_pdf_report`, ~line 1489)
- Test: `tests/test_report_sections.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_convenience_function_detailed_default(tmp_path):
    import numpy as np
    from src.ulanowicz_calculator import UlanowiczCalculator
    from src.oasis_calculator import OASISCalculator
    from src.oasis_pdf_report import generate_oasis_pdf_report

    flow = np.array([[0, 10, 5], [2, 0, 8], [6, 1, 0]], dtype=float)
    uc = UlanowiczCalculator(flow, ['A', 'B', 'C'])
    oc = OASISCalculator(uc)
    out = tmp_path / "r.html"
    generate_oasis_pdf_report(oc, uc, org_name='X', output_path=str(out))
    html = out.read_text()
    assert 'Action Roadmap' in html  # detailed=True is the default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report_sections.py::test_convenience_function_detailed_default -q`
Expected: FAIL — `generate_oasis_pdf_report` does not pass `detailed`, so HTML lacks the section (or `save_html` not invoked). If it errors instead, confirm the cause is the missing wiring.

- [ ] **Step 3: Implement — thread `detailed` through**

Modify `generate_oasis_pdf_report` signature to add `detailed: bool = True` and pass it into the `OASISPDFReport(...)` construction. Locate the existing call (constructs `report = OASISPDFReport(org_name=..., ...)` near line 1517) and add `detailed=detailed,` to its kwargs. Add to the signature:

```python
def generate_oasis_pdf_report(
    oasis_calculator,
    ulanowicz_calculator,
    org_name: str = "Organization",
    chart_images: Optional[Dict[str, bytes]] = None,
    logo_path: Optional[str] = None,
    output_path: Optional[str] = None,
    detailed: bool = True,
) -> Optional[bytes]:
```

Verify `save_html(output_path)` is called when `output_path` is provided (it already returns HTML/bytes). If the existing function only writes HTML when `output_path` is set, that path is exercised by the test.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_report_sections.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/oasis_pdf_report.py tests/test_report_sections.py
git commit -m "feat(report): expose detailed flag on generate_oasis_pdf_report (default on)"
```

---

### Task 9: Full regression run + appendix glossary (optional polish)

**Files:**
- Modify: `src/oasis_pdf_report.py` (extend `_build_appendix` with a metric glossary)
- Test: existing suites

- [ ] **Step 1: Add a glossary appendix subsection**

In `_build_appendix`, append a glossary table after the existing weight tables. Pull
definitions from `src/docs_registry.py` if available, else use a static fallback for the
core metrics displayed in Table 2:

```python
        glossary_terms = [
            ('Total System Throughput (TST)', 'Sum of all flows; overall activity scale.'),
            ('Average Mutual Information (AMI)', 'Average constraint/organization per unit flow.'),
            ('Ascendency (A)', 'Organized power: TST × AMI.'),
            ('Development Capacity (C)', 'Upper bound on ascendency: TST × flow diversity.'),
            ('Overhead (Φ)', 'Reserve capacity C − A; supports resilience.'),
            ('Relative Ascendency (α)', 'A / C; efficiency-vs-resilience balance.'),
            ('Robustness (R)', '−α·ln(α); maximized near α ≈ 0.37.'),
            ('Window of Viability', 'Empirical sustainable band α ∈ [0.2, 0.6].'),
        ]
        glossary_rows = "".join(
            f"<tr><td>{_escape(t)}</td><td>{_escape(d)}</td></tr>"
            for t, d in glossary_terms
        )
        glossary_html = f"""
        <h2>Appendix B: Metric Glossary</h2>
        <table>
            <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
            <tbody>{glossary_rows}</tbody>
            <caption>Table A3. Glossary of core metrics.</caption>
        </table>
        """
```

Return the existing appendix HTML with `+ glossary_html` appended before the closing of the method's returned string.

- [ ] **Step 2: Run the full test suite**

Run: `python3 -m pytest tests/test_report_intelligence.py tests/test_report_sections.py -q`
Expected: PASS (all green)

- [ ] **Step 3: Manual smoke check (optional)**

Run: `python3 -c "from tests.test_report_sections import _build_report; open('/tmp/oasis_demo.html','w').write(_build_report(True).generate_html()); print('wrote /tmp/oasis_demo.html')"`
Expected: file written; open to eyeball the new sections.

- [ ] **Step 4: Commit**

```bash
git add src/oasis_pdf_report.py
git commit -m "feat(report): add metric glossary appendix"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Benchmarking → Task 2 + Task 7 `_build_benchmarking` + Task 6 chart. ✓
- Risk & Resilience → Task 3 + Task 7 `_build_risk_resilience`. ✓
- Prioritized Action Roadmap → Task 4 + Task 7 `_build_action_roadmap`. ✓
- ESG framework mapping (GRI/ESRS/TCFD) → Task 5 + Task 7 `_build_esg_mapping`. ✓
- Layered exec+analyst structure → section ordering in Task 7 Step 4. ✓
- No new formulas → constants/lookups only; robustness curve plotted not re-derived. ✓
- Backward compatibility (`detailed` flag) → Task 7 + Task 8. ✓
- Testing (unit + smoke) → Tasks 1-6 unit, Task 7-8 smoke. ✓
- Glossary appendix (analyst depth) → Task 9. ✓

**Placeholder scan:** No TBD/TODO; all code steps contain complete code. ✓

**Type consistency:** `build_benchmark_view` returns dict with keys `alpha/robustness/lower/upper/optimum/in_window/position/distance_to_optimum/reference_anchors` — consumed identically in `_build_benchmarking`. `build_risk_view` keys `fragility/overhead_ratio/redundancy/items` (item keys `severity/title/evidence/implication`) — match `_build_risk_resilience`. `build_action_roadmap` keys `immediate/short_term/medium_term` (item keys include `expected_impact`) — match `_build_action_roadmap`. `build_esg_crosswalk` row keys `oasis_dimension/finding_summary/gri_ref/esrs_ref/tcfd_ref` — match `_build_esg_mapping`. ✓

**Note for executor:** section numbers in headings (2–7) assume the detailed layout. When `detailed=False`, headings retain their original numbering from the untouched methods; the numeric labels in the new sections are cosmetic and only appear in detailed mode. If exact sequential numbering across modes is later required, switch to CSS counters — out of scope here.
