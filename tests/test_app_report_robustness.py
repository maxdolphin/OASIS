"""
Regression tests for the post-refactor crash sweep + scale-aware guard.

Covers three failure classes that crashed the Streamlit analysis/report paths
after the precompute / gradient-reframe / metric-sentinel refactor:

  1. Brittle ``metrics['key']`` bracket access when the app's passed dict lacks
     the key (e.g. tier-2 cache-reconstruction has no ``viability_lower_bound``).
  2. ``:.Nf`` formatting applied to sentinel strings ('insufficient',
     'skipped_large_graph', 'not_computed_large_graph') or None.
  3. JSON-stringified node-index keys breaking ``node_names[node_id]`` lookups.

Plus the PART-2 scale guard: ``AdvancedNetworkAnalyzer.get_all_metrics()`` must
complete quickly on a 300-node graph and report ``computation_mode='approximate'``.

The Streamlit display functions are exercised through a fake ``streamlit`` module
whose calls are no-ops but STILL evaluate their arguments, so f-string
formatting / KeyError / NameError surface as real exceptions.
"""
import os
import sys
import time
import types
import json

import numpy as np
import pytest

# --------------------------------------------------------------------------
# Fake streamlit: no-op UI, but arguments are still evaluated.
# Installed BEFORE importing app. No other test imports streamlit, and none of
# the report/network modules import it, so this does not pollute the suite.
# --------------------------------------------------------------------------

class _Ctx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __call__(self, *a, **k):
        return self
    def __getattr__(self, name):
        return _Ctx()
    def __iter__(self):
        return iter([])


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def get(self, name, default=None):
        return dict.get(self, name, default)


class _FakeStreamlit(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.session_state = _SessionState()
        self.sidebar = _Ctx()
        self.column_config = _Ctx()

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]

    def tabs(self, labels, **k):
        return [_Ctx() for _ in range(len(labels))]

    def container(self, *a, **k): return _Ctx()
    def expander(self, *a, **k): return _Ctx()
    def spinner(self, *a, **k): return _Ctx()
    def form(self, *a, **k): return _Ctx()
    def status(self, *a, **k): return _Ctx()
    def popover(self, *a, **k): return _Ctx()
    def empty(self, *a, **k): return _Ctx()
    def progress(self, *a, **k): return _Ctx()
    def set_page_config(self, *a, **k): return None

    def cache_data(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda fn: fn

    def cache_resource(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda fn: fn

    def button(self, *a, **k): return False
    def download_button(self, *a, **k): return False
    def checkbox(self, *a, **k): return bool(k.get('value', False))
    def toggle(self, *a, **k): return bool(k.get('value', False))

    def radio(self, label, options=(), index=0, **k):
        try:
            return list(options)[index or 0]
        except Exception:
            return None

    def selectbox(self, label, options=(), index=0, **k):
        try:
            return list(options)[index or 0]
        except Exception:
            return None

    def multiselect(self, label, options=(), default=None, **k):
        return list(default) if default else []

    def slider(self, label, min_value=0, max_value=100, value=None, **k):
        return value if value is not None else min_value

    def number_input(self, label, min_value=None, max_value=None, value=0, **k):
        return value if value is not None else (min_value or 0)

    def text_input(self, *a, **k): return k.get('value', '')
    def text_area(self, *a, **k): return k.get('value', '')
    def file_uploader(self, *a, **k): return None
    def color_picker(self, *a, **k): return k.get('value', '#000000')
    def date_input(self, *a, **k): return None
    def rerun(self, *a, **k): return None
    def stop(self, *a, **k): return None

    def __getattr__(self, name):
        return lambda *a, **k: None


def _install_fake_streamlit():
    mod = _FakeStreamlit('streamlit')
    comp = types.ModuleType('streamlit.components')
    v1 = types.ModuleType('streamlit.components.v1')
    v1.html = lambda *a, **k: None
    v1.iframe = lambda *a, **k: None
    v1.declare_component = lambda *a, **k: (lambda *aa, **kk: None)
    comp.v1 = v1
    mod.components = comp
    sys.modules['streamlit'] = mod
    sys.modules['streamlit.components'] = comp
    sys.modules['streamlit.components.v1'] = v1
    return mod


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

_st = _install_fake_streamlit()

import app  # noqa: E402  (import after fake streamlit is installed)
from ulanowicz_calculator import UlanowiczCalculator  # noqa: E402
from network_analyzer import AdvancedNetworkAnalyzer  # noqa: E402
from publication_report import PublicationReportGenerator  # noqa: E402
from database.full_profile import precompute_full_profile  # noqa: E402
from database.precompute_pipeline import get_precompute_pipeline  # noqa: E402

SMALL = os.path.join(REPO, 'data/ecosystem_samples/cone_spring_original.json')
LARGE = os.path.join(REPO, 'data/ecosystem_samples/enzyme_network.json')


def _load(path):
    with open(path) as f:
        d = json.load(f)
    fm = np.asarray(d.get('flow_matrix', d.get('flows')), dtype=np.float64)
    nn = d.get('node_names', d.get('nodes')) or [f'N{i}' for i in range(fm.shape[0])]
    name = d.get('org_name', d.get('name', os.path.basename(path)))
    return fm, list(nn), name


def _build_data(path):
    """Mirror the app's cache-RECONSTRUCTION path (the risky one): metrics come
    from fresh tier-2 vectorized metrics (no viability bounds), plus the SI/ELD/
    TD fills and is_viable the app adds — deliberately NOT the fuller
    get_extended_metrics(), to exercise the missing-key code paths."""
    fm, nn, name = _load(path)
    pipeline = get_precompute_pipeline()
    profile = precompute_full_profile(fm, nn, org_name=name)
    _st.session_state['full_profile'] = profile
    _st.session_state['analysis_data'] = {
        'flow_matrix': fm, 'node_names': nn, 'org_name': name,
    }

    calc = UlanowiczCalculator(fm, nn, use_vectorized=True)
    em = dict(pipeline._get_vectorized_metrics(fm, nn))
    if 'is_viable' not in em:
        alpha = em.get('relative_ascendency', 0)
        em['is_viable'] = 0.2 <= alpha <= 0.6
    core = profile.get('core', {})
    for key, method in (
        ('structural_information', 'calculate_structural_information'),
        ('effective_link_density', 'calculate_effective_link_density'),
        ('trophic_depth', 'calculate_trophic_depth'),
    ):
        if key not in em or em.get(key, 0) == 0:
            stored = core.get(key)
            em[key] = stored if (stored is not None and stored != 0) else getattr(calc, method)()
    assess = calc.assess_regenerative_health()
    oasis = profile.get('oasis') if isinstance(profile.get('oasis'), dict) and \
        'dimension_scores' in profile.get('oasis', {}) else None
    return dict(calc=calc, em=em, assess=assess, name=name, fm=fm, nn=nn, oasis=oasis)


# --------------------------------------------------------------------------
# PART 1 — display functions run clean for small AND large orgs
# --------------------------------------------------------------------------

@pytest.mark.parametrize('path', [SMALL, LARGE])
def test_display_functions_run_clean(path):
    d = _build_data(path)
    app.display_core_metrics_combined(d['em'], d['assess'], d['name'], d['fm'], d['nn'])
    app.display_visual_summary_cards(d['em'], d['assess'])
    app.display_network_analysis(d['calc'], d['em'], d['fm'], d['nn'])
    app.display_oasis_health(d['calc'], d['em'], d['fm'], d['nn'], d['name'])
    app.display_detailed_report(d['calc'], d['em'], d['assess'], d['name'])


# --------------------------------------------------------------------------
# PART 1b — both report generators for small AND large org
# --------------------------------------------------------------------------

@pytest.mark.parametrize('path', [SMALL, LARGE])
def test_report_generators(path):
    from src.pdf_generator import generate_pdf_report
    d = _build_data(path)
    gen = PublicationReportGenerator(
        calculator=d['calc'], metrics=d['em'], assessments=d['assess'],
        org_name=d['name'], flow_matrix=d['fm'], node_names=d['nn'],
        oasis_profile=d['oasis'])
    report = gen.generate_full_report()
    assert isinstance(report, str) and len(report) > 500
    # Key sections present.
    for marker in ('RESULTS', 'RECOMMENDATIONS'):
        assert marker in report.upper()
    pdf = generate_pdf_report(gen, d['calc'], d['em'], {})
    assert pdf is not None and len(pdf) > 0


def test_report_missing_viability_bounds_does_not_crash():
    """The exact KeyError('viability_lower_bound') regression: reconstruction
    metrics lack the viability bounds; the report must backfill, not crash."""
    fm, nn, name = _load(SMALL)
    calc = UlanowiczCalculator(fm, nn, use_vectorized=True)
    minimal = {
        'ascendency_ratio': 0.35, 'relative_ascendency': 0.35, 'robustness': 0.4,
        'redundancy': 0.5, 'overhead': 100.0, 'overhead_ratio': 0.6,
        'network_efficiency': 0.35, 'ascendency': 60.0, 'development_capacity': 160.0,
        'flow_diversity': 2.5, 'total_system_throughput': 100.0,
    }  # deliberately NO viability_lower_bound / viability_upper_bound / is_viable
    gen = PublicationReportGenerator(
        calculator=calc, metrics=minimal, assessments={}, org_name=name,
        flow_matrix=fm, node_names=nn)
    report = gen.generate_full_report()
    assert isinstance(report, str) and len(report) > 500
    assert gen.metrics['viability_lower_bound'] == pytest.approx(0.2)
    assert gen.metrics['viability_upper_bound'] == pytest.approx(0.6)


# --------------------------------------------------------------------------
# PART 2 — scale-aware guard
# --------------------------------------------------------------------------

def _random_flow_matrix(n, density=0.03, seed=7):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    fm = np.where(m < density, rng.random((n, n)) * 10, 0.0)
    np.fill_diagonal(fm, 0.0)
    return fm


def test_get_all_metrics_large_graph_is_approximate_and_fast():
    n = 300
    fm = _random_flow_matrix(n)
    nn = [f'N{i}' for i in range(n)]
    analyzer = AdvancedNetworkAnalyzer(fm, nn)
    t0 = time.time()
    metrics = analyzer.get_all_metrics()
    elapsed = time.time() - t0
    assert elapsed < 20.0, f"get_all_metrics on {n} nodes took {elapsed:.1f}s"
    assert metrics['computation_mode'] == 'approximate'
    assert 'betweenness_centrality' in metrics['approximated_metrics']
    # Sentinels present for the skipped metrics.
    assert metrics['small_world']['small_world_sigma'] == 'not_computed_large_graph'
    assert metrics['rich_club']['rich_club_coefficient'] == 'skipped_large_graph'


def test_small_graph_is_full_mode():
    fm, nn, _ = _load(SMALL)
    analyzer = AdvancedNetworkAnalyzer(fm, nn)
    metrics = analyzer.get_all_metrics()
    assert metrics['computation_mode'] == 'full'
    assert metrics['approximated_metrics'] == []


def test_summary_report_handles_sentinels():
    """get_summary_report must format sentinel small-world/rich-club values as
    text, never raising on ':.2f'."""
    fm = _random_flow_matrix(300)
    nn = [f'N{i}' for i in range(300)]
    analyzer = AdvancedNetworkAnalyzer(fm, nn)
    text = analyzer.get_summary_report()  # would ValueError if sentinels hit :.2f
    assert 'not_computed_large_graph' in text


def test_safe_fmt_helper():
    assert app._safe_fmt(0.12345) == '0.12'
    assert app._safe_fmt(0.12345, '.3f') == '0.123'
    assert app._safe_fmt('insufficient') == 'insufficient'
    assert app._safe_fmt('skipped_large_graph') == 'skipped_large_graph'
    assert app._safe_fmt(None) == 'N/A'
    assert app._safe_fmt(True) == 'True'  # bool is not treated as a number


def test_coerce_int_keys_and_node_label():
    d = {'0': 0.5, '1': 0.3, 'x': 0.1}
    coerced = app._coerce_int_keys(d)
    assert coerced[0] == 0.5 and coerced[1] == 0.3 and coerced['x'] == 0.1
    node_names = ['A', 'B', 'C']
    assert app._node_label(node_names, '2') == 'C'
    assert app._node_label(node_names, 1) == 'B'
    assert app._node_label(node_names, 'missing') == 'missing'
