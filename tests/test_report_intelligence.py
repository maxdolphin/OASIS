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


def _metrics(alpha=0.42, robustness=0.36):
    return {'ascendency_ratio': alpha, 'robustness': robustness,
            'development_capacity': 100.0, 'ascendency': 42.0,
            'overhead_ratio': 1 - alpha, 'redundancy': 0.5}


def _recs():
    return [
        {'priority': 'CRITICAL', 'dimension': 'SUSTAINABLE', 'issue': 'Too rigid',
         'action': 'Diversify pathways', 'metrics_to_improve': ['redundancy']},
        {'priority': 'HIGH', 'dimension': 'OPEN', 'issue': 'Low interconnectivity',
         'action': 'Add cross-functional channels', 'metrics_to_improve': ['connectance']},
        {'priority': 'MEDIUM', 'dimension': 'SYMBIOTIC', 'issue': 'Inequality',
         'action': 'Redistribute resources', 'metrics_to_improve': ['gini_coefficient']},
    ]


# --- Task 1: constants + verdict ---

def test_constants_match_codebase_window():
    assert ri.VIABILITY_LOWER == 0.2
    assert ri.VIABILITY_UPPER == 0.6
    assert abs(ri.ROBUSTNESS_OPTIMUM - 0.367879441) < 1e-6


def test_executive_verdict_mentions_score_and_status():
    v = ri.executive_verdict(_profile(overall=72.0, status='HEALTHY'))
    assert '72' in v
    assert 'HEALTHY' in v.upper()


def test_executive_verdict_handles_empty_profile():
    assert isinstance(ri.executive_verdict({}), str)


# --- Task 2: benchmark ---

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
    assert v['position'] == 'above'


def test_benchmark_view_handles_missing_metrics():
    v = ri.build_benchmark_view({}, {})
    assert 'alpha' in v and 'reference_anchors' in v


# --- Task 3: risk ---

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


# --- Task 4: roadmap ---

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


# --- Task 5: ESG crosswalk ---

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
    assert len(rows) == 5


# --- Task 6: WoV chart ---

def test_wov_chart_returns_png_bytes():
    png = ri.render_window_of_viability_png(alpha=0.42, robustness=0.36)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b'\x89PNG\r\n\x1a\n'


def test_wov_chart_handles_zero_alpha():
    png = ri.render_window_of_viability_png(alpha=0.0, robustness=0.0)
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
