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

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH — efficiency (alpha) interpretation bands (E-19 fix)
# ---------------------------------------------------------------------------
# The efficiency label of alpha (= network_efficiency = A/C) MUST agree with the
# Window-of-Viability risk framing. Under that model, HIGH efficiency is NOT
# "good" — it is over-organized / brittle. The interior sub-bands (0.35, 0.45)
# split the in-window range into developing / optimal / efficient.
#   alpha < 0.2                : under-organized / chaotic  (below window)
#   0.2  <= alpha < 0.35       : developing
#   0.35 <= alpha < 0.45       : optimal (near 1/e robustness peak)
#   0.45 <= alpha < 0.6        : efficient (watch for rigidity)
#   alpha >= 0.6               : over-organized / brittle    (above window)
EFFICIENCY_BAND_LOWER = VIABILITY_LOWER   # 0.2  — below = under-organized
EFFICIENCY_BAND_DEVELOPING = 0.35         # 0.35 — developing -> optimal
EFFICIENCY_BAND_OPTIMAL = 0.45            # 0.45 — optimal -> efficient
EFFICIENCY_BAND_UPPER = VIABILITY_UPPER   # 0.6  — above = over-organized/brittle

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH — robustness "high" threshold (E-20 fix)
# ---------------------------------------------------------------------------
# The lower rung 0.2 was already shared across paths; the "high" cutoff differed
# (0.20 on the PDF path, 0.25 on LaTeX/CLI). Unified to 0.25 so R = 0.22 no
# longer flips verdict by export type. Documented choice: 0.25 is the more
# conservative rung and matches the LaTeX/CLI narrative already in production.
ROBUSTNESS_HIGH_THRESHOLD = 0.25


def categorize_efficiency_label(alpha: float) -> str:
    """
    Viability-anchored efficiency label for alpha (= network_efficiency = A/C).

    Single source of truth for the E-19 efficiency labels. HIGH efficiency is
    framed as over-organized/brittle (NOT "good"), consistent with the risk view.
    """
    if alpha < EFFICIENCY_BAND_LOWER:
        return "Under-organized"
    elif alpha < EFFICIENCY_BAND_DEVELOPING:
        return "Developing"
    elif alpha < EFFICIENCY_BAND_OPTIMAL:
        return "Optimal"
    elif alpha < EFFICIENCY_BAND_UPPER:
        return "Efficient"
    else:
        return "Over-organized"


def categorize_robustness_label(robustness: float) -> str:
    """
    Single source of truth for robustness labels (E-20). The "high" rung uses
    ROBUSTNESS_HIGH_THRESHOLD (0.25) consistently across all report paths.
    """
    if robustness < 0.1:
        return "Very Low"
    elif robustness < 0.15:
        return "Low"
    elif robustness < VIABILITY_LOWER:      # 0.2
        return "Moderate"
    elif robustness < ROBUSTNESS_HIGH_THRESHOLD:  # 0.25
        return "High"
    else:
        return "Very High"


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
        try:
            from src.services import published_metrics_db as pdb
        except Exception:
            from services import published_metrics_db as pdb  # 'src' on sys.path
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
