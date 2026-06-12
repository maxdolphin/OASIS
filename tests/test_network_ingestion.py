import numpy as np
import pytest

from src import network_ingestion as ni


# ---- matrix format ----

def test_parse_matrix_basic():
    csv = (",A,B,C\n"
           "A,0,5,0\n"
           "B,0,0,3\n"
           "C,2,0,0\n")
    res = ni.parse_network_csv(csv)
    assert res.fmt == 'matrix'
    assert res.node_names == ['A', 'B', 'C']
    assert res.flow_matrix.shape == (3, 3)
    assert res.flow_matrix[0, 1] == 5
    assert res.flow_matrix[2, 0] == 2


def test_parse_matrix_non_square_raises():
    csv = ",A,B,C\nA,0,5,0\nB,0,0,3\n"
    with pytest.raises(ni.NetworkIngestionError):
        ni.parse_network_csv(csv)


def test_parse_matrix_non_numeric_raises():
    csv = ",A,B\nA,0,x\nB,1,0\n"
    with pytest.raises(ni.NetworkIngestionError):
        ni.parse_network_csv(csv)


def test_parse_matrix_negative_raises():
    csv = ",A,B\nA,0,-5\nB,1,0\n"
    with pytest.raises(ni.NetworkIngestionError):
        ni.parse_network_csv(csv)


# ---- edge list format ----

def test_parse_edge_list_with_headers():
    csv = ("source,target,weight\n"
           "A,B,5\n"
           "B,C,3\n"
           "C,A,2\n")
    res = ni.parse_network_csv(csv)
    assert res.fmt == 'edgelist'
    assert res.node_names == ['A', 'B', 'C']
    assert res.flow_matrix[0, 1] == 5
    assert res.flow_matrix[1, 2] == 3
    assert res.flow_matrix[2, 0] == 2


def test_parse_edge_list_synonym_headers():
    csv = ("from,to,count\n"
           "Sales,IT,10\n"
           "IT,Sales,4\n")
    res = ni.parse_network_csv(csv)
    assert res.fmt == 'edgelist'
    assert set(res.node_names) == {'Sales', 'IT'}


def test_parse_edge_list_aggregates_duplicates():
    csv = ("source,target,weight\n"
           "A,B,5\n"
           "A,B,3\n"
           "B,A,1\n")
    res = ni.parse_network_csv(csv)
    i = res.node_names.index('A')
    j = res.node_names.index('B')
    assert res.flow_matrix[i, j] == 8


def test_parse_edge_list_no_weight_defaults_to_one():
    csv = ("source,target\n"
           "A,B\n"
           "A,B\n"
           "B,A\n")
    res = ni.parse_network_csv(csv)
    i = res.node_names.index('A')
    j = res.node_names.index('B')
    assert res.flow_matrix[i, j] == 2
    assert any('counted as 1' in w for w in res.warnings)


def test_parse_edge_list_heuristic_no_known_headers():
    # No recognized headers, but two label columns + a numeric column.
    csv = ("dept_a,dept_b,n\n"
           "X,Y,7\n"
           "Y,X,2\n")
    res = ni.parse_network_csv(csv)
    assert res.fmt == 'edgelist'
    assert set(res.node_names) == {'X', 'Y'}


# ---- validation warnings ----

def test_isolated_node_warning():
    # D appears as a column but receives/sends nothing in edge form -> use matrix
    csv = (",A,B,D\n"
           "A,0,5,0\n"
           "B,3,0,0\n"
           "D,0,0,0\n")
    res = ni.parse_network_csv(csv)
    assert any('isolated' in w.lower() for w in res.warnings)


def test_self_loop_warning():
    csv = (",A,B\n"
           "A,2,5\n"
           "B,3,0\n")
    res = ni.parse_network_csv(csv)
    assert any('self-loop' in w.lower() for w in res.warnings)


def test_zero_total_flow_raises():
    csv = ",A,B\nA,0,0\nB,0,0\n"
    with pytest.raises(ni.NetworkIngestionError):
        ni.parse_network_csv(csv)


def test_empty_raises():
    with pytest.raises(ni.NetworkIngestionError):
        ni.parse_network_csv("\n")


# ---- templates ----

def test_templates_roundtrip():
    m = ni.parse_network_csv(ni.matrix_template_csv())
    assert m.fmt == 'matrix' and len(m.node_names) == 4
    e = ni.parse_network_csv(ni.edgelist_template_csv())
    assert e.fmt == 'edgelist' and len(e.node_names) == 4


# ---- end-to-end into the engine ----

def test_ingested_matrix_feeds_calculator():
    from src.ulanowicz_calculator import UlanowiczCalculator
    res = ni.parse_network_csv(ni.edgelist_template_csv())
    calc = UlanowiczCalculator(res.flow_matrix, res.node_names)
    metrics = calc.get_extended_metrics()
    assert metrics['total_system_throughput'] > 0
