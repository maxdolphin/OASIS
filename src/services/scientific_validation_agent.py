"""
Scientific Validation Agent

Agent responsible for ensuring scientific accuracy of all formulas by validating
computed metrics against published research papers.

This agent orchestrates validation by:
1. Loading network data
2. Computing metrics using UlanowiczCalculator
3. Comparing against published values
4. Verifying mathematical relationships
5. Generating validation reports

References:
- Ulanowicz et al. (2009): Ecological Complexity 6, 27-36
- Ulanowicz & Norden (1990): Int. J. Systems Sci. 21(2), 429-437
"""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ulanowicz_calculator import UlanowiczCalculator
from src.services.published_metrics_db import (
    PUBLISHED_METRICS,
    NETWORK_DATA_FILES,
    VALIDATION_CHECKS,
    LogBase,
    get_published_metric,
    get_tolerance,
    get_log_base,
    list_networks,
    list_metrics,
    get_network_info,
)


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    metric_name: str
    published_value: Optional[float]
    computed_value: float
    difference: Optional[float]
    percent_error: Optional[float]
    tolerance: float
    status: ValidationStatus
    note: Optional[str] = None


@dataclass
class CheckResult:
    """Result of a validation check."""
    check_name: str
    description: str
    status: ValidationStatus
    details: str
    error_value: Optional[float] = None


@dataclass
class NetworkValidationResult:
    """Complete validation result for a network."""
    network_id: str
    network_name: str
    source: str
    timestamp: str
    computed_metrics: Dict[str, float]
    metric_comparisons: List[MetricComparison]
    validation_checks: List[CheckResult]
    overall_status: ValidationStatus
    summary: str


@dataclass
class ValidationReport:
    """Complete validation report for all networks."""
    timestamp: str
    networks_validated: int
    formulas_checked: int
    overall_status: ValidationStatus
    network_results: List[NetworkValidationResult]
    summary: str


class ScientificValidationAgent:
    """
    Agent responsible for ensuring scientific accuracy of all formulas.

    Invoked when:
    - New metrics are added
    - Code changes affect calculations
    - On-demand validation runs
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the validation agent.

        Args:
            base_path: Base path to the project root. If None, auto-detect.
        """
        if base_path is None:
            # Auto-detect project root
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)

    def _load_network_data(self, network_id: str) -> Optional[Dict[str, Any]]:
        """Load network data from JSON file."""
        if network_id not in NETWORK_DATA_FILES:
            return None

        file_path = self.base_path / NETWORK_DATA_FILES[network_id]
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return json.load(f)

    def _compute_metrics(self, flow_matrix: np.ndarray, log_base: LogBase) -> Dict[str, float]:
        """
        Compute all metrics for a flow matrix.

        Args:
            flow_matrix: The network flow matrix
            log_base: The logarithm base to use (for comparison with published values)

        Returns:
            Dictionary of computed metrics
        """
        calc = UlanowiczCalculator(flow_matrix)

        # Get extended metrics
        metrics = calc.get_extended_metrics()

        # If paper used log base 2, we need to convert our natural log results
        # Conversion: log2(x) = ln(x) / ln(2)
        if log_base == LogBase.LOG2:
            ln2 = math.log(2)
            # Scale information-theoretic metrics
            if 'development_capacity' in metrics:
                metrics['development_capacity_log2'] = metrics['development_capacity'] / ln2
            if 'ascendency' in metrics:
                metrics['ascendency_log2'] = metrics['ascendency'] / ln2
            if 'reserve' in metrics:
                metrics['reserve_log2'] = metrics['reserve'] / ln2
            if 'average_mutual_information' in metrics:
                metrics['ami_log2'] = metrics['average_mutual_information'] / ln2

        return metrics

    def _compare_metric(
        self,
        metric_name: str,
        published_value: Optional[float],
        computed_value: float,
        tolerance: float
    ) -> MetricComparison:
        """Compare a computed metric against published value."""
        if published_value is None:
            return MetricComparison(
                metric_name=metric_name,
                published_value=None,
                computed_value=computed_value,
                difference=None,
                percent_error=None,
                tolerance=tolerance,
                status=ValidationStatus.SKIP,
                note="No published value available"
            )

        difference = computed_value - published_value
        if published_value != 0:
            percent_error = abs(difference / published_value) * 100
        else:
            percent_error = float('inf') if computed_value != 0 else 0

        if percent_error <= tolerance * 100:
            status = ValidationStatus.PASS
            note = None
        elif percent_error <= tolerance * 200:  # Within 2x tolerance
            status = ValidationStatus.WARNING
            note = f"Within 2x tolerance ({percent_error:.1f}% vs {tolerance*100:.1f}%)"
        else:
            status = ValidationStatus.FAIL
            note = f"Exceeds tolerance ({percent_error:.1f}% vs {tolerance*100:.1f}%)"

        return MetricComparison(
            metric_name=metric_name,
            published_value=published_value,
            computed_value=computed_value,
            difference=difference,
            percent_error=percent_error,
            tolerance=tolerance,
            status=status,
            note=note
        )

    def _run_validation_checks(self, metrics: Dict[str, float]) -> List[CheckResult]:
        """Run all validation checks on computed metrics."""
        results = []

        # Check 1: Fundamental Relationship (C = A + Phi)
        c = metrics.get('development_capacity', 0)
        a = metrics.get('ascendency', 0)
        phi = metrics.get('reserve', 0)
        if c > 0:
            calculated_c = a + phi
            error = abs(c - calculated_c) / c
            status = ValidationStatus.PASS if error < 0.001 else ValidationStatus.FAIL
            results.append(CheckResult(
                check_name="fundamental_relationship",
                description="C = A + Phi",
                status=status,
                details=f"C={c:.2f}, A+Phi={calculated_c:.2f}",
                error_value=error * 100
            ))

        # Check 2: Relative Ascendency Range
        alpha = metrics.get('relative_ascendency', 0)
        if 0 <= alpha <= 1:
            status = ValidationStatus.PASS
            details = f"alpha = {alpha:.4f} is in valid range [0, 1]"
        else:
            status = ValidationStatus.FAIL
            details = f"alpha = {alpha:.4f} is OUT OF RANGE [0, 1]"
        results.append(CheckResult(
            check_name="range_alpha",
            description="0 <= alpha <= 1",
            status=status,
            details=details
        ))

        # Check 3: Thermodynamic Constraint (A <= C)
        if c > 0:
            if a <= c:
                status = ValidationStatus.PASS
                details = f"A={a:.2f} <= C={c:.2f}"
            else:
                status = ValidationStatus.FAIL
                details = f"A={a:.2f} > C={c:.2f} - VIOLATES thermodynamics"
            results.append(CheckResult(
                check_name="thermodynamic_constraint",
                description="A <= C",
                status=status,
                details=details
            ))

        # Check 4: Positive TST
        tst = metrics.get('total_system_throughput', 0)
        if tst > 0:
            status = ValidationStatus.PASS
            details = f"TST = {tst:.2f} > 0"
        else:
            status = ValidationStatus.FAIL
            details = f"TST = {tst:.2f} - must be positive"
        results.append(CheckResult(
            check_name="positive_tst",
            description="TST > 0",
            status=status,
            details=details
        ))

        # Check 5: Non-negative Reserve
        if phi >= 0:
            status = ValidationStatus.PASS
            details = f"Reserve = {phi:.2f} >= 0"
        else:
            status = ValidationStatus.FAIL
            details = f"Reserve = {phi:.2f} - must be non-negative"
        results.append(CheckResult(
            check_name="non_negative_reserve",
            description="Reserve >= 0",
            status=status,
            details=details
        ))

        # Check 6: FCI Range (if available)
        fci = metrics.get('finn_cycling_index')
        if fci is not None:
            if 0 <= fci <= 1:
                status = ValidationStatus.PASS
                details = f"FCI = {fci:.4f} is in valid range [0, 1]"
            else:
                status = ValidationStatus.FAIL
                details = f"FCI = {fci:.4f} is OUT OF RANGE [0, 1]"
            results.append(CheckResult(
                check_name="range_fci",
                description="0 <= FCI <= 1",
                status=status,
                details=details
            ))

        return results

    def validate_network(self, network_id: str) -> NetworkValidationResult:
        """
        Validate a single network against published data.

        Args:
            network_id: Network identifier

        Returns:
            NetworkValidationResult with all validation details
        """
        timestamp = datetime.now().isoformat()

        # Get network info
        network_info = get_network_info(network_id)
        if network_info is None:
            return NetworkValidationResult(
                network_id=network_id,
                network_name="Unknown",
                source="Unknown",
                timestamp=timestamp,
                computed_metrics={},
                metric_comparisons=[],
                validation_checks=[],
                overall_status=ValidationStatus.ERROR,
                summary=f"Network '{network_id}' not found in published metrics database"
            )

        # Load network data
        network_data = self._load_network_data(network_id)
        if network_data is None:
            return NetworkValidationResult(
                network_id=network_id,
                network_name=network_info.get('source', 'Unknown'),
                source=network_info['source'],
                timestamp=timestamp,
                computed_metrics={},
                metric_comparisons=[],
                validation_checks=[],
                overall_status=ValidationStatus.ERROR,
                summary=f"Could not load network data file for '{network_id}'"
            )

        # Get flow matrix
        flow_matrix = np.array(network_data['flows'], dtype=float)
        network_name = network_data.get('organization', network_id)

        # Get log base and tolerance
        log_base = get_log_base(network_id)
        tolerance = get_tolerance(network_id)

        # Compute metrics
        computed_metrics = self._compute_metrics(flow_matrix, log_base)

        # Compare against published values
        metric_comparisons = []
        published_metrics = network_info['metrics']

        # Map metric names for comparison
        metric_mapping = {
            'total_system_throughput': 'total_system_throughput',
            'development_capacity': 'development_capacity_log2' if log_base == LogBase.LOG2 else 'development_capacity',
            'ascendency': 'ascendency_log2' if log_base == LogBase.LOG2 else 'ascendency',
            'reserve': 'reserve_log2' if log_base == LogBase.LOG2 else 'reserve',
            'relative_ascendency': 'relative_ascendency',
            'average_mutual_information': 'ami_log2' if log_base == LogBase.LOG2 else 'average_mutual_information',
        }

        for pub_name, pub_data in published_metrics.items():
            computed_name = metric_mapping.get(pub_name, pub_name)
            computed_value = computed_metrics.get(computed_name, computed_metrics.get(pub_name, 0))
            published_value = pub_data['value'] if pub_data['reported'] else None

            comparison = self._compare_metric(
                metric_name=pub_name,
                published_value=published_value,
                computed_value=computed_value,
                tolerance=tolerance
            )
            metric_comparisons.append(comparison)

        # Run validation checks
        validation_checks = self._run_validation_checks(computed_metrics)

        # Determine overall status
        all_statuses = [c.status for c in metric_comparisons if c.status != ValidationStatus.SKIP]
        all_statuses.extend([c.status for c in validation_checks])

        if ValidationStatus.ERROR in all_statuses:
            overall_status = ValidationStatus.ERROR
        elif ValidationStatus.FAIL in all_statuses:
            overall_status = ValidationStatus.FAIL
        elif ValidationStatus.WARNING in all_statuses:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASS

        # Generate summary
        passed = sum(1 for s in all_statuses if s == ValidationStatus.PASS)
        total = len(all_statuses)
        summary = f"{passed}/{total} checks passed"

        return NetworkValidationResult(
            network_id=network_id,
            network_name=network_name,
            source=network_info['source'],
            timestamp=timestamp,
            computed_metrics=computed_metrics,
            metric_comparisons=metric_comparisons,
            validation_checks=validation_checks,
            overall_status=overall_status,
            summary=summary
        )

    def validate_all(self) -> ValidationReport:
        """
        Validate all networks in the database.

        Returns:
            ValidationReport with results for all networks
        """
        timestamp = datetime.now().isoformat()
        network_results = []
        formulas_checked = 0

        for network_id in list_networks():
            result = self.validate_network(network_id)
            network_results.append(result)
            formulas_checked += len(result.validation_checks) + len(result.metric_comparisons)

        # Determine overall status
        all_statuses = [r.overall_status for r in network_results]

        if ValidationStatus.ERROR in all_statuses:
            overall_status = ValidationStatus.ERROR
        elif ValidationStatus.FAIL in all_statuses:
            overall_status = ValidationStatus.FAIL
        elif ValidationStatus.WARNING in all_statuses:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASS

        # Generate summary
        passed = sum(1 for s in all_statuses if s == ValidationStatus.PASS)
        total = len(all_statuses)
        status_emoji = {
            ValidationStatus.PASS: "PASS",
            ValidationStatus.FAIL: "FAIL",
            ValidationStatus.WARNING: "WARNING",
            ValidationStatus.ERROR: "ERROR"
        }
        summary = f"{passed}/{total} networks pass validation - Overall: {status_emoji[overall_status]}"

        return ValidationReport(
            timestamp=timestamp,
            networks_validated=len(network_results),
            formulas_checked=formulas_checked,
            overall_status=overall_status,
            network_results=network_results,
            summary=summary
        )

    def validate_formula(self, formula_name: str) -> Dict[str, Any]:
        """
        Validate a specific formula across all networks.

        Args:
            formula_name: Name of the formula/metric to validate

        Returns:
            Dictionary with validation results across networks
        """
        results = {}

        for network_id in list_networks():
            network_result = self.validate_network(network_id)

            # Find the metric comparison for this formula
            for comparison in network_result.metric_comparisons:
                if comparison.metric_name == formula_name:
                    results[network_id] = {
                        'published': comparison.published_value,
                        'computed': comparison.computed_value,
                        'percent_error': comparison.percent_error,
                        'status': comparison.status.value
                    }
                    break

        return {
            'formula': formula_name,
            'networks_checked': len(results),
            'results': results
        }

    def generate_report(self, format: str = "text") -> str:
        """
        Generate a validation report.

        Args:
            format: Output format ('text', 'html', 'markdown')

        Returns:
            Formatted report string
        """
        report = self.validate_all()

        if format == "text":
            return self._generate_text_report(report)
        elif format == "html":
            return self._generate_html_report(report)
        elif format == "markdown":
            return self._generate_markdown_report(report)
        else:
            return self._generate_text_report(report)

    def _generate_text_report(self, report: ValidationReport) -> str:
        """Generate text format report."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"         SCIENTIFIC VALIDATION REPORT - {report.timestamp[:10]}")
        lines.append("=" * 70)
        lines.append(f" Networks Validated: {report.networks_validated}")
        lines.append(f" Formulas Checked: {report.formulas_checked}")

        status_symbols = {
            ValidationStatus.PASS: "PASS",
            ValidationStatus.FAIL: "FAIL",
            ValidationStatus.WARNING: "WARN",
            ValidationStatus.SKIP: "SKIP",
            ValidationStatus.ERROR: "ERR"
        }
        lines.append(f" Status: {status_symbols[report.overall_status]}")
        lines.append("=" * 70)

        for result in report.network_results:
            lines.append("")
            lines.append(f" NETWORK: {result.network_name}")
            lines.append(f" Source: {result.source}")
            lines.append(f" Status: {status_symbols[result.overall_status]}")
            lines.append("-" * 70)

            # Metric comparisons
            for comp in result.metric_comparisons:
                if comp.published_value is not None:
                    symbol = status_symbols[comp.status]
                    lines.append(f"   {comp.metric_name}:")
                    lines.append(f"     Computed: {comp.computed_value:.4f}")
                    lines.append(f"     Published: {comp.published_value:.4f}")
                    if comp.percent_error is not None:
                        lines.append(f"     Error: {comp.percent_error:.2f}% [{symbol}]")
                    if comp.note:
                        lines.append(f"     Note: {comp.note}")

            # Validation checks
            lines.append("")
            lines.append("   Validation Checks:")
            for check in result.validation_checks:
                symbol = status_symbols[check.status]
                lines.append(f"     [{symbol}] {check.description}: {check.details}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f" SUMMARY: {report.summary}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_markdown_report(self, report: ValidationReport) -> str:
        """Generate markdown format report."""
        lines = []
        lines.append(f"# Scientific Validation Report")
        lines.append(f"**Date**: {report.timestamp[:10]}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- **Networks Validated**: {report.networks_validated}")
        lines.append(f"- **Formulas Checked**: {report.formulas_checked}")
        lines.append(f"- **Overall Status**: {report.overall_status.value.upper()}")
        lines.append("")

        for result in report.network_results:
            lines.append(f"## {result.network_name}")
            lines.append(f"**Source**: {result.source}")
            lines.append(f"**Status**: {result.overall_status.value.upper()}")
            lines.append("")

            # Metric comparisons table
            lines.append("### Metric Comparisons")
            lines.append("| Metric | Published | Computed | Error | Status |")
            lines.append("|--------|-----------|----------|-------|--------|")
            for comp in result.metric_comparisons:
                pub = f"{comp.published_value:.4f}" if comp.published_value else "N/A"
                err = f"{comp.percent_error:.2f}%" if comp.percent_error else "N/A"
                lines.append(f"| {comp.metric_name} | {pub} | {comp.computed_value:.4f} | {err} | {comp.status.value} |")

            lines.append("")
            lines.append("### Validation Checks")
            for check in result.validation_checks:
                emoji = {"pass": "v", "fail": "x", "warning": "!", "skip": "-", "error": "!"}
                lines.append(f"- [{emoji.get(check.status.value, '?')}] **{check.description}**: {check.details}")

            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML format report."""
        status_colors = {
            ValidationStatus.PASS: "#28a745",
            ValidationStatus.FAIL: "#dc3545",
            ValidationStatus.WARNING: "#ffc107",
            ValidationStatus.SKIP: "#6c757d",
            ValidationStatus.ERROR: "#dc3545"
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Scientific Validation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: #e9ecef; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .network {{ border: 1px solid #dee2e6; border-radius: 5px; margin: 20px 0; padding: 20px; }}
        .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: white; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
        th {{ background: #f8f9fa; }}
        .check {{ margin: 5px 0; padding: 5px 10px; background: #f8f9fa; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Scientific Validation Report</h1>
        <p><strong>Date:</strong> {report.timestamp[:10]}</p>

        <div class="summary">
            <h3>Summary</h3>
            <p><strong>Networks Validated:</strong> {report.networks_validated}</p>
            <p><strong>Formulas Checked:</strong> {report.formulas_checked}</p>
            <p><strong>Overall Status:</strong> <span class="status" style="background: {status_colors[report.overall_status]}">{report.overall_status.value.upper()}</span></p>
        </div>
"""

        for result in report.network_results:
            html += f"""
        <div class="network">
            <h2>{result.network_name}</h2>
            <p><strong>Source:</strong> {result.source}</p>
            <p><strong>Status:</strong> <span class="status" style="background: {status_colors[result.overall_status]}">{result.overall_status.value.upper()}</span></p>

            <h3>Metric Comparisons</h3>
            <table>
                <tr><th>Metric</th><th>Published</th><th>Computed</th><th>Error</th><th>Status</th></tr>
"""
            for comp in result.metric_comparisons:
                pub = f"{comp.published_value:.4f}" if comp.published_value else "N/A"
                err = f"{comp.percent_error:.2f}%" if comp.percent_error else "N/A"
                html += f"                <tr><td>{comp.metric_name}</td><td>{pub}</td><td>{comp.computed_value:.4f}</td><td>{err}</td><td><span class='status' style='background: {status_colors[comp.status]}'>{comp.status.value}</span></td></tr>\n"

            html += """            </table>

            <h3>Validation Checks</h3>
"""
            for check in result.validation_checks:
                html += f"            <div class='check'><span class='status' style='background: {status_colors[check.status]}'>{check.status.value}</span> <strong>{check.description}:</strong> {check.details}</div>\n"

            html += "        </div>\n"

        html += """
    </div>
</body>
</html>
"""
        return html
