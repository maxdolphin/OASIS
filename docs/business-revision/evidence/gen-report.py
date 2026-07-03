#!/usr/bin/env python3
"""
Headless PDF report generator for OASIS sample organizations.

Replicates the app.py PDF-export path (app.py ~line 4887) without Streamlit:
    - load a flow-network JSON (keys: organization / nodes / flows)
    - build UlanowiczCalculator -> get_extended_metrics()
    - build assessments via calculator.assess_regenerative_health()
    - build a PublicationReportGenerator
    - call generate_pdf_report(report_generator, calculator, metrics, charts=None)

The PDF generator itself computes the OASIS health, benchmarking, risk,
roadmap and ESG sections internally (see src/pdf_generator.py), so no chart
figures are required for a text-complete, section-complete report.

Usage:
    python3 docs/business-revision/evidence/gen-report.py <input.json> <output.pdf>
"""
import json
import os
import sys

import numpy as np

# Make src/ importable exactly as the app does (its modules use bare imports
# such as `from ulanowicz_calculator import UlanowiczCalculator`).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from ulanowicz_calculator import UlanowiczCalculator          # noqa: E402
from publication_report import PublicationReportGenerator      # noqa: E402
from pdf_generator import generate_pdf_report                  # noqa: E402


def load_network(path):
    """Load a flow-network JSON. All three sample files share the schema
    {organization: str, nodes: [str], flows: [[float]]}."""
    with open(path) as fh:
        data = json.load(fh)

    org_name = data.get("organization") or data.get("name") or "Organization"
    node_names = data.get("nodes") or data.get("node_names")
    flows = data.get("flows")
    if flows is None:
        flows = data.get("flow_matrix")
    if node_names is None or flows is None:
        raise ValueError(f"Could not find nodes/flows in {path}; keys={list(data.keys())}")

    flow_matrix = np.array(flows, dtype=float)
    if flow_matrix.shape[0] != flow_matrix.shape[1]:
        raise ValueError(f"Flow matrix must be square, got {flow_matrix.shape}")
    if len(node_names) != flow_matrix.shape[0]:
        raise ValueError("Node count does not match flow matrix dimension")
    return org_name, node_names, flow_matrix


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    in_path, out_path = sys.argv[1], sys.argv[2]
    org_name, node_names, flow_matrix = load_network(in_path)

    # Mirror the app: vectorized calculator, extended metrics, assessments.
    calculator = UlanowiczCalculator(flow_matrix, node_names, use_vectorized=True)
    metrics = calculator.get_extended_metrics()

    # Ensure extended metrics the report relies on are present (app.py does the same).
    if not metrics.get("structural_information"):
        metrics["structural_information"] = calculator.calculate_structural_information()
    if not metrics.get("effective_link_density"):
        metrics["effective_link_density"] = calculator.calculate_effective_link_density()
    if not metrics.get("trophic_depth"):
        metrics["trophic_depth"] = calculator.calculate_trophic_depth()

    assessments = calculator.assess_regenerative_health()

    report_generator = PublicationReportGenerator(
        calculator=calculator,
        metrics=metrics,
        assessments=assessments,
        org_name=org_name,
        flow_matrix=calculator.flow_matrix,
        node_names=calculator.node_names,
    )

    # charts=None -> text/section-complete PDF (OASIS, benchmarking, risk,
    # roadmap and ESG are computed inside the PDF generator).
    pdf_bytes = generate_pdf_report(report_generator, calculator, metrics, charts=None)
    if not pdf_bytes:
        print("ERROR: generate_pdf_report returned no content", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(pdf_bytes)

    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
