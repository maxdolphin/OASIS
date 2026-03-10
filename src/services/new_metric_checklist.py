"""
New Metric Checklist

Enforces a scientific checklist when adding new metrics to ensure:
1. Formula is properly documented with paper reference
2. Published values are extracted from papers
3. Test datasets are identified
4. Expected values are documented
5. Unit tests are created
6. Validation against published data passes
7. Tolerance threshold is defined

This prevents adding metrics that haven't been properly validated
against peer-reviewed research.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class ChecklistItemStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ChecklistItem:
    """A single item in the new metric checklist."""
    name: str
    description: str
    required: bool = True
    status: ChecklistItemStatus = ChecklistItemStatus.PENDING
    notes: Optional[str] = None
    evidence: Optional[str] = None  # Link or reference to evidence


@dataclass
class MetricDefinition:
    """Definition of a new metric to be added."""
    name: str
    symbol: str
    formula: str
    description: str
    unit: str
    source_paper: str
    doi: Optional[str] = None
    page: Optional[int] = None
    equation_number: Optional[str] = None
    expected_range: Optional[tuple] = None
    tolerance: float = 0.05
    test_networks: List[str] = field(default_factory=list)
    published_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChecklistResult:
    """Result of running the checklist for a new metric."""
    metric_name: str
    timestamp: str
    total_items: int
    completed_items: int
    failed_items: int
    pending_items: int
    is_ready: bool
    items: List[ChecklistItem]
    summary: str
    blocking_issues: List[str]


class NewMetricChecklist:
    """
    Checklist for adding new metrics to the system.

    Ensures scientific rigor by requiring:
    1. Formula documented with paper reference
    2. Published values extracted from paper
    3. Test dataset identified
    4. Expected values documented
    5. Unit tests created
    6. Validation against published data passes
    7. Tolerance threshold defined
    """

    # Standard checklist items required for every new metric
    STANDARD_CHECKLIST = [
        ChecklistItem(
            name="formula_documented",
            description="Formula is documented with exact mathematical notation",
            required=True
        ),
        ChecklistItem(
            name="paper_reference",
            description="Source paper is cited with DOI or full reference",
            required=True
        ),
        ChecklistItem(
            name="equation_identified",
            description="Specific equation number/figure in paper is identified",
            required=True
        ),
        ChecklistItem(
            name="published_values_extracted",
            description="Published values for at least one test network are extracted",
            required=True
        ),
        ChecklistItem(
            name="test_network_identified",
            description="Test network with known values is identified",
            required=True
        ),
        ChecklistItem(
            name="expected_values_documented",
            description="Expected computed values are documented",
            required=True
        ),
        ChecklistItem(
            name="unit_specified",
            description="Units of the metric are clearly specified",
            required=True
        ),
        ChecklistItem(
            name="range_defined",
            description="Valid range of metric values is defined (if applicable)",
            required=False
        ),
        ChecklistItem(
            name="tolerance_defined",
            description="Validation tolerance threshold is defined",
            required=True
        ),
        ChecklistItem(
            name="implementation_created",
            description="Implementation code is written",
            required=True
        ),
        ChecklistItem(
            name="unit_tests_created",
            description="Unit tests with known values are created",
            required=True
        ),
        ChecklistItem(
            name="validation_passes",
            description="Validation against published data passes within tolerance",
            required=True
        ),
        ChecklistItem(
            name="documentation_added",
            description="Metric is documented in codebase documentation",
            required=False
        ),
    ]

    def __init__(self):
        """Initialize the checklist."""
        self.metrics_in_progress: Dict[str, List[ChecklistItem]] = {}

    def start_new_metric(self, metric_definition: MetricDefinition) -> str:
        """
        Start the checklist for a new metric.

        Args:
            metric_definition: Definition of the metric to add

        Returns:
            Metric name/ID for tracking
        """
        # Create a copy of the standard checklist
        items = [
            ChecklistItem(
                name=item.name,
                description=item.description,
                required=item.required
            )
            for item in self.STANDARD_CHECKLIST
        ]

        self.metrics_in_progress[metric_definition.name] = items

        # Auto-complete some items based on provided definition
        self._auto_check_items(metric_definition)

        return metric_definition.name

    def _auto_check_items(self, metric_definition: MetricDefinition):
        """Auto-check items based on provided metric definition."""
        items = self.metrics_in_progress.get(metric_definition.name, [])

        for item in items:
            if item.name == "formula_documented" and metric_definition.formula:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = f"Formula: {metric_definition.formula}"

            elif item.name == "paper_reference" and metric_definition.source_paper:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = metric_definition.source_paper
                if metric_definition.doi:
                    item.evidence += f" (DOI: {metric_definition.doi})"

            elif item.name == "equation_identified" and metric_definition.equation_number:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = f"Equation {metric_definition.equation_number}"
                if metric_definition.page:
                    item.evidence += f", page {metric_definition.page}"

            elif item.name == "published_values_extracted" and metric_definition.published_values:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = str(metric_definition.published_values)

            elif item.name == "test_network_identified" and metric_definition.test_networks:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = ", ".join(metric_definition.test_networks)

            elif item.name == "unit_specified" and metric_definition.unit:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = metric_definition.unit

            elif item.name == "range_defined" and metric_definition.expected_range:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = f"[{metric_definition.expected_range[0]}, {metric_definition.expected_range[1]}]"

            elif item.name == "tolerance_defined":
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = f"{metric_definition.tolerance * 100}%"

    def mark_complete(
        self,
        metric_name: str,
        item_name: str,
        evidence: Optional[str] = None
    ) -> bool:
        """
        Mark a checklist item as complete.

        Args:
            metric_name: Name of the metric
            item_name: Name of the checklist item
            evidence: Evidence/reference for completion

        Returns:
            True if successfully marked, False otherwise
        """
        items = self.metrics_in_progress.get(metric_name)
        if items is None:
            return False

        for item in items:
            if item.name == item_name:
                item.status = ChecklistItemStatus.COMPLETE
                item.evidence = evidence
                return True

        return False

    def mark_failed(
        self,
        metric_name: str,
        item_name: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Mark a checklist item as failed.

        Args:
            metric_name: Name of the metric
            item_name: Name of the checklist item
            notes: Notes on why it failed

        Returns:
            True if successfully marked, False otherwise
        """
        items = self.metrics_in_progress.get(metric_name)
        if items is None:
            return False

        for item in items:
            if item.name == item_name:
                item.status = ChecklistItemStatus.FAILED
                item.notes = notes
                return True

        return False

    def mark_not_applicable(
        self,
        metric_name: str,
        item_name: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Mark a checklist item as not applicable.

        Args:
            metric_name: Name of the metric
            item_name: Name of the checklist item
            reason: Reason why it's not applicable

        Returns:
            True if successfully marked, False otherwise
        """
        items = self.metrics_in_progress.get(metric_name)
        if items is None:
            return False

        for item in items:
            if item.name == item_name:
                item.status = ChecklistItemStatus.NOT_APPLICABLE
                item.notes = reason
                return True

        return False

    def check_metric_ready(self, metric_name: str) -> ChecklistResult:
        """
        Check if a metric is ready to be added to the system.

        Args:
            metric_name: Name of the metric to check

        Returns:
            ChecklistResult with detailed status
        """
        items = self.metrics_in_progress.get(metric_name)
        timestamp = datetime.now().isoformat()

        if items is None:
            return ChecklistResult(
                metric_name=metric_name,
                timestamp=timestamp,
                total_items=0,
                completed_items=0,
                failed_items=0,
                pending_items=0,
                is_ready=False,
                items=[],
                summary="Metric not found in checklist",
                blocking_issues=["Metric not started - call start_new_metric() first"]
            )

        completed = sum(1 for i in items if i.status == ChecklistItemStatus.COMPLETE)
        failed = sum(1 for i in items if i.status == ChecklistItemStatus.FAILED)
        pending = sum(1 for i in items if i.status == ChecklistItemStatus.PENDING)
        na = sum(1 for i in items if i.status == ChecklistItemStatus.NOT_APPLICABLE)

        # Find blocking issues (required items that are pending or failed)
        blocking_issues = []
        for item in items:
            if item.required:
                if item.status == ChecklistItemStatus.PENDING:
                    blocking_issues.append(f"PENDING: {item.description}")
                elif item.status == ChecklistItemStatus.FAILED:
                    reason = f" - {item.notes}" if item.notes else ""
                    blocking_issues.append(f"FAILED: {item.description}{reason}")

        # Metric is ready only if all required items are complete
        is_ready = len(blocking_issues) == 0

        if is_ready:
            summary = f"Metric '{metric_name}' is READY to be added ({completed}/{len(items)} complete)"
        else:
            summary = f"Metric '{metric_name}' is NOT READY - {len(blocking_issues)} blocking issue(s)"

        return ChecklistResult(
            metric_name=metric_name,
            timestamp=timestamp,
            total_items=len(items),
            completed_items=completed,
            failed_items=failed,
            pending_items=pending,
            is_ready=is_ready,
            items=items.copy(),
            summary=summary,
            blocking_issues=blocking_issues
        )

    def get_checklist_status(self, metric_name: str) -> Optional[List[ChecklistItem]]:
        """Get current status of all checklist items for a metric."""
        return self.metrics_in_progress.get(metric_name)

    def generate_checklist_report(self, metric_name: str) -> str:
        """
        Generate a formatted checklist report.

        Args:
            metric_name: Name of the metric

        Returns:
            Formatted report string
        """
        result = self.check_metric_ready(metric_name)

        lines = []
        lines.append("=" * 60)
        lines.append(f"  NEW METRIC CHECKLIST: {metric_name}")
        lines.append("=" * 60)
        lines.append(f"  Status: {'READY' if result.is_ready else 'NOT READY'}")
        lines.append(f"  Timestamp: {result.timestamp}")
        lines.append("-" * 60)

        status_symbols = {
            ChecklistItemStatus.COMPLETE: "[x]",
            ChecklistItemStatus.PENDING: "[ ]",
            ChecklistItemStatus.FAILED: "[!]",
            ChecklistItemStatus.NOT_APPLICABLE: "[~]"
        }

        for item in result.items:
            symbol = status_symbols[item.status]
            required = "*" if item.required else " "
            lines.append(f"  {symbol}{required} {item.description}")
            if item.evidence:
                lines.append(f"         Evidence: {item.evidence}")
            if item.notes:
                lines.append(f"         Notes: {item.notes}")

        lines.append("-" * 60)
        lines.append(f"  Complete: {result.completed_items}/{result.total_items}")
        lines.append(f"  Pending:  {result.pending_items}")
        lines.append(f"  Failed:   {result.failed_items}")

        if result.blocking_issues:
            lines.append("")
            lines.append("  BLOCKING ISSUES:")
            for issue in result.blocking_issues:
                lines.append(f"    - {issue}")

        lines.append("=" * 60)
        lines.append("")
        lines.append("  Legend: [x] Complete  [ ] Pending  [!] Failed  [~] N/A")
        lines.append("          * = Required item")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# EXAMPLE METRIC DEFINITIONS
# =============================================================================

EXAMPLE_METRICS = {
    "ascendency": MetricDefinition(
        name="ascendency",
        symbol="A",
        formula="A = Sum(T_ij * log((T_ij * TST) / (T_i. * T_.j)))",
        description="Scaled mutual information representing organized power",
        unit="flow-bits",
        source_paper="Ulanowicz et al. (2009) Quantifying sustainability",
        doi="10.1016/j.ecocom.2008.10.005",
        page=29,
        equation_number="12",
        expected_range=(0, None),  # Non-negative
        tolerance=0.05,
        test_networks=["cone_spring_original", "prawns_alligator_original"],
        published_values={
            "cone_spring_original": 68191,  # Using log2
            "prawns_alligator_original": 53.9
        }
    ),

    "development_capacity": MetricDefinition(
        name="development_capacity",
        symbol="C",
        formula="C = -Sum(T_ij * log(T_ij / TST))",
        description="Scaled system indeterminacy - capacity for development",
        unit="flow-bits",
        source_paper="Ulanowicz et al. (2009) Quantifying sustainability",
        doi="10.1016/j.ecocom.2008.10.005",
        page=29,
        equation_number="11",
        expected_range=(0, None),
        tolerance=0.05,
        test_networks=["cone_spring_original"],
        published_values={
            "cone_spring_original": 135000  # Using log2
        }
    ),

    "relative_ascendency": MetricDefinition(
        name="relative_ascendency",
        symbol="alpha",
        formula="alpha = A / C",
        description="Key sustainability metric - fraction of capacity realized as organization",
        unit="dimensionless",
        source_paper="Ulanowicz et al. (2009) Quantifying sustainability",
        doi="10.1016/j.ecocom.2008.10.005",
        page=29,
        equation_number="(derived)",
        expected_range=(0, 1),
        tolerance=0.05,
        test_networks=["cone_spring_original", "cone_spring_eutrophicated", "crystal_river_creek"],
        published_values={
            "cone_spring_original": 0.505,
            "cone_spring_eutrophicated": 0.529,
            "crystal_river_creek": 0.552
        }
    ),
}
