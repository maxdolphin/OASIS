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
# INDICATIVE-REFERENCE CAVEAT — single source of truth for the framing note
# ---------------------------------------------------------------------------
# The [0.2, 0.6] band and its 1/e optimum are ECOLOGICAL reference points. Their
# transfer to organizational networks is NOT established (Fath 2019: organizational
# networks are more redundant and sit elsewhere on the curve; org calibration is an
# open question). We therefore present the band as an *indicative directional
# reference*, never as an absolute organizational pass/fail threshold.
INDICATIVE_REFERENCE_CAVEAT = (
    "Reference band derived from ecological systems; organizational calibration "
    "is an active area — read this as a directional indicator, not a compliance "
    "threshold."
)
# The center of the indicative reference band, used only as a neutral gradient
# anchor for direction-of-travel (NOT a target). This is the midpoint of the
# existing bounds, introducing no new threshold constant.
_INDICATIVE_BAND_CENTER = (VIABILITY_LOWER + VIABILITY_UPPER) / 2.0  # 0.4


def assess_alpha_position(alpha: float) -> Dict[str, Any]:
    """
    Gradient classifier for relative ascendency (alpha) against the INDICATIVE
    ecological reference band [VIABILITY_LOWER, VIABILITY_UPPER].

    This is the single source of truth for reframing the old binary
    "Viable / Non-Viable (PASS/FAIL)" verdict into a *position-on-a-gradient*
    with a *direction-of-travel*. It introduces NO new threshold constants and
    changes NO score formula — it only classifies an already-computed alpha
    relative to the existing bounds, framed as an indicative reference.

    Returns a dict with:
      - position: 'under-organized' (alpha < VIABILITY_LOWER),
                  'balanced' (VIABILITY_LOWER <= alpha <= VIABILITY_UPPER),
                  'over-organized' (alpha > VIABILITY_UPPER).
      - direction_of_travel: plain-language nudge back toward balance.
      - descriptor: short plain-English phrase describing the position relative
                    to the indicative reference band.
      - relative_distance: signed gradient value. Negative = below the lower
                    edge (by how much); positive = above the upper edge; when
                    inside the band it is the signed offset from the band center
                    (negative = below center, positive = above center). This is a
                    gradient, NOT a pass/fail flag.
      - lower / upper / center: the indicative reference band bounds/center.
      - caveat: the indicative-reference caveat string.
    """
    alpha = float(alpha)

    if alpha < VIABILITY_LOWER:
        position = 'under-organized'
        direction = 'increase structure / coordination'
        descriptor = ('diffuse / under-structured relative to the indicative '
                      'reference band')
        # signed: how far *below* the lower edge (negative)
        relative_distance = alpha - VIABILITY_LOWER
    elif alpha > VIABILITY_UPPER:
        position = 'over-organized'
        direction = 'increase redundancy / flexibility'
        descriptor = ('highly streamlined / over-structured relative to the '
                      'indicative reference band')
        # signed: how far *above* the upper edge (positive)
        relative_distance = alpha - VIABILITY_UPPER
    else:
        position = 'balanced'
        direction = 'maintain balance'
        descriptor = ('within the indicative reference band '
                      '(balanced structure and flexibility)')
        # signed offset from the band center (a gradient, not a verdict)
        relative_distance = alpha - _INDICATIVE_BAND_CENTER

    return {
        'alpha': alpha,
        'position': position,
        'direction_of_travel': direction,
        'descriptor': descriptor,
        'relative_distance': relative_distance,
        'lower': VIABILITY_LOWER,
        'upper': VIABILITY_UPPER,
        'center': _INDICATIVE_BAND_CENTER,
        'caveat': INDICATIVE_REFERENCE_CAVEAT,
    }

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


def sustainable_verdict_narrative(sust_score: float, alpha: float) -> str:
    """
    Single source of truth for the SUSTAINABLE-dimension narrative verdict.

    Reframes the old binary "Viable / Non-Viable" language into a
    position-on-a-gradient + direction-of-travel against the *indicative*
    ecological reference band. The numeric SUSTAINABLE score is unchanged; only
    the wording changes. Never renders a bare absolute "Non-Viable" /
    "UNSUSTAINABLE" pass/fail organizational judgment.
    """
    sust_score = float(sust_score)
    grad = assess_alpha_position(alpha)
    position = grad['position']
    direction = grad['direction_of_travel']

    if sust_score >= 75:
        band_phrase = (
            "sits within the indicative reference band"
            if position == 'balanced'
            else f"sits {position} relative to the indicative reference band"
        )
        return (
            f"Strong sustainability balance (score: {sust_score:.0f}/100). "
            f"On the efficiency/resilience gradient the organization {band_phrase} "
            f"(alpha={alpha:.3f}); direction of travel: {direction}. "
            f"{INDICATIVE_REFERENCE_CAVEAT}"
        )
    elif sust_score >= 50:
        return (
            f"Moderate sustainability (score: {sust_score:.0f}/100). "
            f"On the efficiency/resilience gradient the organization reads as "
            f"{position} relative to the indicative reference band "
            f"(alpha={alpha:.3f}); direction of travel: {direction}. "
            f"{INDICATIVE_REFERENCE_CAVEAT}"
        )
    else:
        return (
            f"Sustainability warrants attention (score: {sust_score:.0f}/100). "
            f"On the efficiency/resilience gradient the organization reads as "
            f"{position} relative to the indicative reference band "
            f"(alpha={alpha:.3f}); direction of travel: {direction}. "
            f"{INDICATIVE_REFERENCE_CAVEAT}"
        )


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
            'title': 'Balanced position within the indicative reference band',
            'evidence': f'Relative ascendency alpha = {alpha:.3f} lies within the '
                        f'indicative reference band [{VIABILITY_LOWER}, {VIABILITY_UPPER}]. '
                        f'{INDICATIVE_REFERENCE_CAVEAT}',
            'implication': 'Healthy balance of efficiency and resilience; direction of '
                           'travel: maintain balance.',
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


# ---------------------------------------------------------------------------
# ESG FRAMEWORK CROSSWALK — indicative structural-lens mapping (NOT compliance)
# ---------------------------------------------------------------------------
# This is a FINDING-SPECIFIC crosswalk: it reads OASIS *structure* (how the org
# is wired) and points to the disclosure areas that structural evidence informs.
# It is NOT a compliance mapping and does NOT attest to any GRI/ESRS/TCFD
# requirement. Where a framework code is a genuine analogue rather than a direct
# disclosure (notably TCFD, a climate-financial framework, against non-climate
# structural findings) it is flagged `contextual` and carries an explicit caveat
# — never presented as a direct disclosure. The dimension->construct mapping
# follows docs/business-revision/evidence/expert-org-management.md §3.2.
INDICATIVE_ESG_CAVEAT = (
    "Indicative structural-lens crosswalk — not a compliance attestation. It maps "
    "OASIS network-structure findings to the disclosure areas they inform; it does "
    "not verify, satisfy, or attest to any GRI, ESRS/CSRD, or TCFD requirement."
)

# Real framework structure used below (series/pillar granularity, no invented codes):
#   GRI 2 (General Disclosures 2021): 2-13 delegation, 2-16 critical concerns,
#         2-17 collective knowledge of the highest governance body, 2-29 stakeholder
#         engagement. GRI 3 (Material Topics 2021): 3-3 management of material topics.
#         GRI 401 Employment; GRI 404 Training & education.
#   ESRS 2 General Disclosures: GOV-1 role/expertise of admin bodies, GOV-2 information
#         to bodies, SBM-2 stakeholder interests/views, SBM-3 material IROs & business-
#         model resilience, IRO-1 process to identify/assess IROs. ESRS S1 Own workforce;
#         ESRS G1 Business conduct.
#   TCFD pillars: Governance, Strategy, Risk Management, Metrics & Targets (climate-scoped).
_ESG_CROSSWALK = {
    'OPEN': {
        'construct': 'boundary-spanning / information circulation / stakeholder connectivity',
        'theme': 'interconnectivity and information circulation',
        'frameworks': [
            {'standard': 'GRI', 'code': 'GRI 2-29, 2-16',
             'label': 'Approach to stakeholder engagement; communication of critical concerns'},
            {'standard': 'ESRS', 'code': 'ESRS 2 SBM-2, GOV-2',
             'label': 'Interests/views of stakeholders; information flow to administrative bodies'},
            {'standard': 'TCFD', 'code': 'Governance',
             'label': 'Board oversight — the channels by which risk/opportunity information reaches oversight',
             'contextual': True,
             'caveat': 'TCFD is climate-scoped; used here as a structural analogue for '
                       'information-flow-to-oversight, not a climate disclosure.'},
        ],
        'disclosure_relevance': (
            'Open (boundary-spanning and information circulation) evidences whether the '
            'stakeholder-engagement and information-flow processes disclosed under GRI 2-29 '
            'and ESRS 2 SBM-2/GOV-2 actually carry information across the organization and up '
            'to its oversight bodies — the structural substrate beneath those qualitative claims.'),
    },
    'AUTONOMOUS': {
        'construct': 'distributed decision rights / empowerment / feedback loops',
        'theme': 'organizational learning and devolved decision-making',
        'frameworks': [
            {'standard': 'GRI', 'code': 'GRI 2-13, 3-3',
             'label': 'Delegation of responsibility for managing impacts; management of material topics'},
            {'standard': 'ESRS', 'code': 'ESRS 2 GOV-1, IRO-1',
             'label': 'Role of administrative bodies; process to identify/assess/manage impacts, risks & opportunities'},
            {'standard': 'TCFD', 'code': 'Risk Management',
             'label': 'Processes to identify, assess and manage risks — whether detection/response is embedded and devolved',
             'contextual': True,
             'caveat': 'TCFD Risk Management is climate-scoped; the structural reading of '
                       'devolved risk-detection is an analogue, not a climate disclosure.'},
        ],
        'disclosure_relevance': (
            'Autonomous (distributed decision rights and feedback loops) informs how '
            'responsibility for managing impacts is delegated (GRI 2-13, ESRS 2 GOV-1) and '
            'whether risk identification and response are embedded across the organization '
            'rather than centralized (ESRS IRO-1; TCFD Risk Management as an analogue).'),
    },
    'SYMBIOTIC': {
        'construct': 'cross-functional collaboration / relational coordination / reciprocity',
        'theme': 'cross-functional reciprocity and relational coordination',
        'frameworks': [
            {'standard': 'GRI', 'code': 'GRI 3-3, 401',
             'label': 'Management of material social topics; employment / relational conditions'},
            {'standard': 'ESRS', 'code': 'ESRS S1; G1',
             'label': 'Own workforce (social dialogue, working conditions); corporate culture / business conduct'},
            {'standard': 'TCFD', 'code': 'Governance (contextual)',
             'label': 'Cross-functional collaboration is not a direct TCFD disclosure',
             'contextual': True,
             'caveat': 'TCFD is a climate-financial framework; cross-functional collaboration '
                       'is shown only as contextual organizational-resilience input, not a TCFD disclosure.'},
        ],
        'disclosure_relevance': (
            'Symbiotic (cross-functional reciprocity and relational coordination) evidences '
            'the collaboration and relational conditions in the own workforce that underlie '
            'ESRS S1 social disclosures and the corporate-culture element of ESRS G1 / GRI 3-3 '
            '— the structural reciprocity beneath those qualitative social claims.'),
    },
    'INTELLIGENT': {
        'construct': 'information-processing / learning / knowledge & functional diversity',
        'theme': 'functional diversity and information-processing capacity',
        'frameworks': [
            {'standard': 'GRI', 'code': 'GRI 2-17, 404',
             'label': 'Collective knowledge of the highest governance body; training & education'},
            {'standard': 'ESRS', 'code': 'ESRS 2 GOV-1; S1',
             'label': 'Expertise/skills of administrative bodies; skills development in own workforce'},
            {'standard': 'TCFD', 'code': 'Governance',
             'label': 'Board competencies to assess and oversee risk',
             'contextual': True,
             'caveat': 'TCFD scopes board competency to climate risk; used here as an analogue '
                       'for information-processing capacity, not a climate-competency disclosure.'},
        ],
        'disclosure_relevance': (
            'Intelligent (functional diversity and information-processing capacity) informs the '
            'collective-knowledge and expertise conditions disclosed under GRI 2-17 and ESRS 2 '
            'GOV-1 — the structural diversity that determines whether governance and workforce '
            'bodies can actually process the matters they are disclosed as overseeing.'),
    },
    'SUSTAINABLE': {
        'construct': 'structural balance / efficiency-vs-resilience / adaptive capacity',
        'theme': 'efficiency/resilience structural balance (Window of Viability)',
        'frameworks': [
            {'standard': 'GRI', 'code': 'GRI 3-3',
             'label': 'Management of the material topic of long-term organizational resilience',
             'contextual': True,
             'caveat': 'GRI has no dedicated structural-resilience disclosure; GRI 201-2 '
                       '(financial implications of climate change) is deliberately NOT used — '
                       'OASIS structural balance is not a climate-financial metric.'},
            {'standard': 'ESRS', 'code': 'ESRS 2 SBM-3',
             'label': 'Material impacts, risks & opportunities and the resilience of the business model'},
            {'standard': 'TCFD', 'code': 'Strategy — Resilience',
             'label': 'Resilience of the strategy',
             'contextual': True,
             'caveat': 'TCFD frames strategic resilience under climate scenarios; OASIS measures '
                       'structural (network) resilience — a contextual analogue, not a '
                       'climate-scenario disclosure.'},
        ],
        'disclosure_relevance': (
            'Sustainable (efficiency/resilience structural balance — the Window of Viability) '
            'provides a network-structural indicator of adaptive capacity that informs the '
            'business-model-resilience narrative of ESRS 2 SBM-3. It is explicitly distinct '
            'from the climate-financial risk addressed by GRI 201-2 / TCFD climate-scenario '
            'analysis, which this structural framework does not measure.'),
    },
}


def _esg_materiality(status: str) -> Dict[str, Any]:
    """
    Status-driven materiality flag: reflects THIS org's finding, not a static table.

    CRITICAL -> flagged for attention (potentially material disclosure area);
    WARNING  -> watch (emerging materiality signal);
    HEALTHY  -> supporting evidence (structural conditions favorable);
    otherwise -> not assessed. Reads the precomputed OASIS dimension status; it
    recomputes nothing.
    """
    s = (status or 'N/A').upper()
    if s == 'CRITICAL':
        return {'flag': 'attention', 'material': True,
                'label': 'Flagged for attention — potentially material disclosure area'}
    if s == 'WARNING':
        return {'flag': 'watch', 'material': True,
                'label': 'Watch — emerging materiality signal in this disclosure area'}
    if s == 'HEALTHY':
        return {'flag': 'supporting', 'material': False,
                'label': 'Supporting evidence — structural conditions favorable for this disclosure area'}
    return {'flag': 'not_assessed', 'material': False,
            'label': 'Not assessed'}


def _esg_ref_string(frameworks: List[Dict[str, Any]], standard: str) -> str:
    """Backward-compatible per-standard reference string (with contextual marker)."""
    parts = []
    for fw in frameworks:
        if fw['standard'] != standard:
            continue
        code = fw['code']
        if fw.get('contextual') and 'contextual' not in code.lower():
            code = f"{code} (contextual)"
        parts.append(code)
    return '; '.join(parts) if parts else 'N/A'


def build_esg_crosswalk(profile: Dict[str, Any],
                        metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Finding-specific, status-driven crosswalk from OASIS structural findings to
    GRI / ESRS-CSRD / TCFD disclosure areas.

    For each of the five dimensions it returns: the relevant framework mappings
    (with contextual caveats where a code is an analogue rather than a direct
    disclosure), a disclosure-relevance sentence describing what the structural
    finding informs, and a materiality flag driven by the org's ACTUAL dimension
    status from the precomputed OASIS profile. This is an INDICATIVE structural
    lens only — NOT a compliance attestation (see INDICATIVE_ESG_CAVEAT). No
    scores or statuses are recomputed here.
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
        frameworks = cw['frameworks']
        rows.append({
            'oasis_dimension': dim,
            'construct': cw['construct'],
            'finding_summary': finding,
            'frameworks': frameworks,
            'disclosure_relevance': cw['disclosure_relevance'],
            'materiality': _esg_materiality(stt),
            # backward-compatible per-standard strings for existing consumers:
            'gri_ref': _esg_ref_string(frameworks, 'GRI'),
            'esrs_ref': _esg_ref_string(frameworks, 'ESRS'),
            'tcfd_ref': _esg_ref_string(frameworks, 'TCFD'),
            'caveat': INDICATIVE_ESG_CAVEAT,
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
               label=f'Indicative reference band [{VIABILITY_LOWER}, {VIABILITY_UPPER}]')
    ax.axvline(ROBUSTNESS_OPTIMUM, color='#d4a843', linestyle='--', linewidth=1,
               label=f'Optimum α ≈ {ROBUSTNESS_OPTIMUM:.2f}')

    a = max(0.0, min(1.0, float(alpha)))
    r = float(robustness) if robustness else (-a * np.log(a) if 0 < a < 1 else 0.0)
    ax.scatter([a], [r], color='#c0392b', s=90, zorder=5,
               label=f'This organization (α={a:.3f})')

    ax.set_xlabel('Relative Ascendency (α = A/C)')
    ax.set_ylabel('Robustness (R)')
    ax.set_title('Position Relative to the Indicative Reference Band')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    return buf.getvalue()
