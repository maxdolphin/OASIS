"""
Tests for the FINDING-SPECIFIC ESG framework crosswalk.

The crosswalk is an INDICATIVE structural-lens mapping from OASIS dimensions to
GRI / ESRS-CSRD / TCFD disclosure areas — NOT a compliance attestation. These
tests pin the substantive upgrade from the old one-to-one code lookup:
per-dimension framework list, a disclosure-relevance sentence, and a
status-driven materiality flag; plus the fix/caveat of the previously-stretched
SUSTAINABLE -> climate-financial mapping.
"""
import numpy as np

from src import report_intelligence as ri

DIMS = ['OPEN', 'AUTONOMOUS', 'SYMBIOTIC', 'INTELLIGENT', 'SUSTAINABLE']


def _profile(statuses=None):
    statuses = statuses or {'open': 'HEALTHY', 'autonomous': 'WARNING',
                            'symbiotic': 'HEALTHY', 'intelligent': 'WARNING',
                            'sustainable': 'HEALTHY'}
    return {
        'dimension_scores': {'open': 70, 'autonomous': 55, 'symbiotic': 80,
                             'intelligent': 60, 'sustainable': 78},
        'dimension_status': dict(statuses),
        'dimension_details': {'sustainable': {'metrics': {
            'relative_ascendency': 0.42, 'robustness': 0.36, 'is_viable': True}}},
        'overall_score': 70.0, 'overall_status': 'HEALTHY',
    }


def _metrics(alpha=0.42):
    return {'ascendency_ratio': alpha, 'robustness': 0.36,
            'overhead_ratio': 1 - alpha, 'redundancy': 0.5}


# --- Structure: every dimension is substantively populated -------------------

def test_covers_all_five_dimensions():
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    assert {r['oasis_dimension'] for r in rows} == set(DIMS)


def test_each_dimension_has_frameworks_relevance_and_materiality():
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    for r in rows:
        # >= 1 framework mapping, each with a standard + code
        assert isinstance(r['frameworks'], list) and len(r['frameworks']) >= 1
        for fw in r['frameworks']:
            assert fw.get('standard') and fw.get('code')
        # non-empty disclosure-relevance sentence
        assert isinstance(r['disclosure_relevance'], str)
        assert len(r['disclosure_relevance'].strip()) > 20
        # materiality field driven by status
        assert isinstance(r['materiality'], dict)
        assert r['materiality'].get('flag')
        assert r['materiality'].get('label')


def test_frameworks_span_gri_esrs_tcfd():
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    for r in rows:
        stds = {fw['standard'] for fw in r['frameworks']}
        # each dimension must touch the three families (even if some are caveated)
        assert {'GRI', 'ESRS', 'TCFD'} <= stds, f"{r['oasis_dimension']} missing a family"


def test_backward_compatible_ref_strings_present():
    # Existing report_intelligence tests / pdf path still read these keys.
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    for r in rows:
        for k in ('oasis_dimension', 'finding_summary', 'gri_ref', 'esrs_ref', 'tcfd_ref'):
            assert k in r and isinstance(r[k], str) and r[k]


def test_handles_empty_profile():
    rows = ri.build_esg_crosswalk({}, {})
    assert len(rows) == 5
    for r in rows:
        assert r['materiality']['flag'] == 'not_assessed'


# --- Materiality reflects the org's ACTUAL status ----------------------------

def test_materiality_flags_critical_dimension_as_attention():
    prof = _profile({'open': 'HEALTHY', 'autonomous': 'HEALTHY',
                     'symbiotic': 'HEALTHY', 'intelligent': 'HEALTHY',
                     'sustainable': 'CRITICAL'})
    rows = ri.build_esg_crosswalk(prof, _metrics())
    sust = next(r for r in rows if r['oasis_dimension'] == 'SUSTAINABLE')
    assert sust['materiality']['flag'] == 'attention'
    assert sust['materiality']['material'] is True
    assert 'material' in sust['materiality']['label'].lower()


def test_materiality_reads_healthy_as_supporting_evidence():
    prof = _profile({'open': 'HEALTHY', 'autonomous': 'HEALTHY',
                     'symbiotic': 'HEALTHY', 'intelligent': 'HEALTHY',
                     'sustainable': 'HEALTHY'})
    rows = ri.build_esg_crosswalk(prof, _metrics())
    sust = next(r for r in rows if r['oasis_dimension'] == 'SUSTAINABLE')
    assert sust['materiality']['flag'] == 'supporting'
    assert sust['materiality']['material'] is False
    assert 'supporting' in sust['materiality']['label'].lower()


def test_materiality_warning_is_a_watch_signal():
    prof = _profile({'open': 'WARNING', 'autonomous': 'HEALTHY',
                     'symbiotic': 'HEALTHY', 'intelligent': 'HEALTHY',
                     'sustainable': 'HEALTHY'})
    rows = ri.build_esg_crosswalk(prof, _metrics())
    op = next(r for r in rows if r['oasis_dimension'] == 'OPEN')
    assert op['materiality']['flag'] == 'watch'


# --- The audited STRETCH is fixed / caveated ---------------------------------

def test_sustainable_no_longer_bare_climate_financial_mapping():
    """R17 audit: SUSTAINABLE -> GRI 201-2 (climate financial implications) conflated
    an information-theoretic balance metric with climate risk. It must be dropped, or
    only referenced with an explicit contextual caveat."""
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    sust = next(r for r in rows if r['oasis_dimension'] == 'SUSTAINABLE')
    for fw in sust['frameworks']:
        code = fw['code'].lower()
        if '201-2' in code or 'climate' in code:
            assert fw.get('caveat'), "climate-financial mapping must carry a caveat"
    # the relevance sentence must distinguish structural resilience from climate risk
    rel = sust['disclosure_relevance'].lower()
    assert 'climate' in rel and ('structural' in rel or 'network' in rel)


def test_stretched_mappings_carry_a_caveat():
    """Any framework flagged as a stretch/analogue must be explicitly caveated,
    never presented as a direct disclosure."""
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    for r in rows:
        for fw in r['frameworks']:
            if fw.get('contextual'):
                assert fw.get('caveat'), f"{r['oasis_dimension']} contextual mapping needs caveat"


# --- The 'indicative / not attestation' caveat is rendered -------------------

def test_module_caveat_is_defensible_language():
    cav = ri.INDICATIVE_ESG_CAVEAT
    assert 'indicative' in cav.lower()
    assert 'not' in cav.lower()
    assert 'attestation' in cav.lower() or 'compliance' in cav.lower()


def test_every_row_carries_the_caveat():
    rows = ri.build_esg_crosswalk(_profile(), _metrics())
    for r in rows:
        assert r['caveat'] == ri.INDICATIVE_ESG_CAVEAT


def test_caveat_present_in_rendered_pdf_esg_section():
    """The rendered ESG section of the app's actual PDF must carry the
    indicative / not-a-compliance-attestation caveat."""
    import io
    from pypdf import PdfReader
    from src.ulanowicz_calculator import UlanowiczCalculator
    from src.publication_report import PublicationReportGenerator
    from src.pdf_generator import generate_pdf_report

    flow = np.array([
        [0, 10, 0, 0, 5],
        [0, 0, 8, 2, 0],
        [0, 0, 0, 7, 1],
        [3, 0, 0, 0, 6],
        [0, 4, 0, 0, 0],
    ], dtype=float)
    nodes = ['A', 'B', 'C', 'D', 'E']
    calc = UlanowiczCalculator(flow, nodes)
    metrics = calc.get_extended_metrics()
    assessments = calc.assess_regenerative_health()
    rg = PublicationReportGenerator(
        calculator=calc, metrics=metrics, assessments=assessments,
        org_name='Test Org', flow_matrix=flow, node_names=nodes)
    pdf = generate_pdf_report(rg, calc, metrics, charts=None)
    reader = PdfReader(io.BytesIO(pdf))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    assert 'ESG Framework Mapping' in text
    # caveat phrasing
    low = text.lower()
    assert 'indicative' in low
    assert 'not a compliance attestation' in low or 'not a compliance' in low
    # richer content: a disclosure-relevance sentence and a materiality flag render
    assert 'Materiality' in text or 'materiality' in low
