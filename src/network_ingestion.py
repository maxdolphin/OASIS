"""
Network ingestion: parse user-supplied organizational network data (CSV) into a
flow matrix the analysis engine can consume.

Supports two CSV shapes:

1. **Adjacency matrix** — a square table with a header row and an index column of
   identical node labels; cell (i, j) is the flow from row i to column j.

2. **Edge list** — one row per directed flow, with source/target columns and an
   optional weight column (defaults to 1.0). This is the shape most organizational
   tools export (email logs, Teams/Slack messages, Jira transitions).

The module is pure (no Streamlit) so it is unit-testable in isolation and reusable by
both the upload UI and future connector pipelines.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Column-header synonyms used to recognize an edge list.
_SOURCE_HEADERS = {"source", "from", "sender", "origin", "src", "from_node", "from_dept"}
_TARGET_HEADERS = {"target", "to", "recipient", "destination", "dest", "dst",
                   "to_node", "to_dept"}
_WEIGHT_HEADERS = {"weight", "value", "count", "flow", "amount", "volume", "frequency",
                   "messages", "emails"}


@dataclass
class ParseResult:
    """Outcome of parsing a network file."""
    flow_matrix: np.ndarray
    node_names: List[str]
    fmt: str  # 'matrix' or 'edgelist'
    warnings: List[str] = field(default_factory=list)


class NetworkIngestionError(ValueError):
    """Raised for fatal, user-actionable ingestion problems."""


def _read_csv(source) -> pd.DataFrame:
    """Read CSV from a path, file-like, bytes, or raw string into a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if isinstance(source, str) and ("\n" in source or "," in source):
        return pd.read_csv(io.StringIO(source))
    return pd.read_csv(source)


def _identify_edge_columns(df: pd.DataFrame
                           ) -> Optional[Tuple[str, str, Optional[str]]]:
    """Return (source_col, target_col, weight_col|None) if df looks like an edge list."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    src = next((lookup[h] for h in _SOURCE_HEADERS if h in lookup), None)
    tgt = next((lookup[h] for h in _TARGET_HEADERS if h in lookup), None)
    wgt = next((lookup[h] for h in _WEIGHT_HEADERS if h in lookup), None)
    if src is not None and tgt is not None:
        return src, tgt, wgt
    # Heuristic fallback: 2-3 columns whose first two are non-numeric labels.
    if 2 <= len(df.columns) <= 3:
        first_two = df.columns[:2]
        if all(not _is_numeric_series(df[c]) for c in first_two):
            wgt = df.columns[2] if len(df.columns) == 3 else None
            return df.columns[0], df.columns[1], wgt
    return None


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().all()


def parse_edge_list(df: pd.DataFrame, source_col, target_col,
                    weight_col=None) -> ParseResult:
    """Build a square flow matrix from a directed edge list."""
    warnings: List[str] = []
    src = df[source_col].astype(str).str.strip()
    tgt = df[target_col].astype(str).str.strip()

    if weight_col is not None:
        w = pd.to_numeric(df[weight_col], errors="coerce")
        if w.isna().any():
            warnings.append(
                f"{int(w.isna().sum())} edge(s) had non-numeric weights; treated as 0.")
        weights = w.fillna(0.0).to_numpy(dtype=float)
    else:
        warnings.append("No weight column detected; each edge counted as 1.0.")
        weights = np.ones(len(df), dtype=float)

    nodes = sorted(set(src) | set(tgt))
    if len(nodes) < 2:
        raise NetworkIngestionError(
            "An edge list needs at least two distinct nodes.")
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=float)
    for s, t, wv in zip(src, tgt, weights):
        matrix[index[s], index[t]] += float(wv)

    warnings.extend(_validate(matrix, nodes))
    return ParseResult(matrix, nodes, "edgelist", warnings)


def parse_matrix(df: pd.DataFrame) -> ParseResult:
    """Parse an adjacency-matrix DataFrame (first column = row labels)."""
    if df.shape[1] < 2:
        raise NetworkIngestionError(
            "A matrix needs a label column plus one column per node.")
    labels = df.iloc[:, 0].astype(str).str.strip().tolist()
    values = df.iloc[:, 1:]
    node_names = [str(c).strip() for c in values.columns]

    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise NetworkIngestionError(
            "Matrix contains non-numeric cells. Every flow value must be a number.")
    matrix = numeric.to_numpy(dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        raise NetworkIngestionError(
            f"Flow matrix must be square; got {matrix.shape[0]} rows "
            f"x {matrix.shape[1]} columns.")

    warnings: List[str] = []
    if labels != node_names:
        warnings.append(
            "Row labels and column headers differ; using column headers as node names.")
    warnings.extend(_validate(matrix, node_names))
    return ParseResult(matrix, node_names, "matrix", warnings)


def _validate(matrix: np.ndarray, node_names: List[str]) -> List[str]:
    """Return non-fatal warnings; raise NetworkIngestionError on fatal problems."""
    warnings: List[str] = []
    if matrix.size == 0 or matrix.shape[0] < 2:
        raise NetworkIngestionError("Network must contain at least two nodes.")
    if np.isnan(matrix).any():
        raise NetworkIngestionError("Flow matrix contains missing (NaN) values.")
    if (matrix < 0).any():
        raise NetworkIngestionError(
            "Flow values must be non-negative (flows represent magnitudes).")
    if matrix.sum() <= 0:
        raise NetworkIngestionError(
            "Total flow is zero; the network has no activity to analyze.")

    if np.trace(matrix) > 0:
        warnings.append(
            "Self-loops detected on the diagonal; these are retained but typically "
            "represent intra-unit flow.")
    isolated = [node_names[i] for i in range(matrix.shape[0])
                if matrix[i, :].sum() == 0 and matrix[:, i].sum() == 0]
    if isolated:
        preview = ", ".join(isolated[:5]) + ("…" if len(isolated) > 5 else "")
        warnings.append(
            f"{len(isolated)} isolated node(s) with no flows: {preview}.")
    return warnings


def parse_network_csv(source) -> ParseResult:
    """
    Parse a network CSV (path, file-like, bytes, raw string, or DataFrame),
    auto-detecting matrix vs edge-list format.
    """
    try:
        df = _read_csv(source)
    except Exception as exc:  # pragma: no cover - passthrough of pandas errors
        raise NetworkIngestionError(f"Could not read CSV: {exc}") from exc

    if df.empty:
        raise NetworkIngestionError("The uploaded file is empty.")

    edge_cols = _identify_edge_columns(df)
    if edge_cols is not None:
        return parse_edge_list(df, *edge_cols)
    return parse_matrix(df)


def matrix_template_csv() -> str:
    """Return a downloadable adjacency-matrix CSV template."""
    return (
        ",Sales,Marketing,IT,HR\n"
        "Sales,0,8,3,2\n"
        "Marketing,6,0,2,1\n"
        "IT,4,5,0,3\n"
        "HR,3,2,4,0\n"
    )


def edgelist_template_csv() -> str:
    """Return a downloadable edge-list CSV template."""
    return (
        "source,target,weight\n"
        "Sales,Marketing,8\n"
        "Sales,IT,3\n"
        "Marketing,Sales,6\n"
        "IT,HR,3\n"
        "HR,Sales,3\n"
    )
