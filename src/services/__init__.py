"""
Services Package

Contains validation and scientific analysis services for the
Adaptive Organization Analysis System.

Services:
- ScientificValidationAgent: Validates computed metrics against published research
- PublishedMetricsDB: Database of published metric values from research papers
- NewMetricChecklist: Checklist for adding new scientifically validated metrics
"""

from src.services.published_metrics_db import (
    PUBLISHED_METRICS,
    NETWORK_DATA_FILES,
    VALIDATION_CHECKS,
    LogBase,
    PublishedMetric,
    NetworkPublishedData,
    get_published_metric,
    get_tolerance,
    get_log_base,
    list_networks,
    list_metrics,
    get_network_info,
)

from src.services.scientific_validation_agent import (
    ScientificValidationAgent,
    ValidationStatus,
    MetricComparison,
    CheckResult,
    NetworkValidationResult,
    ValidationReport,
)

from src.services.new_metric_checklist import (
    NewMetricChecklist,
    MetricDefinition,
    ChecklistItem,
    ChecklistItemStatus,
    ChecklistResult,
    EXAMPLE_METRICS,
)


__all__ = [
    # Published Metrics Database
    'PUBLISHED_METRICS',
    'NETWORK_DATA_FILES',
    'VALIDATION_CHECKS',
    'LogBase',
    'PublishedMetric',
    'NetworkPublishedData',
    'get_published_metric',
    'get_tolerance',
    'get_log_base',
    'list_networks',
    'list_metrics',
    'get_network_info',

    # Scientific Validation Agent
    'ScientificValidationAgent',
    'ValidationStatus',
    'MetricComparison',
    'CheckResult',
    'NetworkValidationResult',
    'ValidationReport',

    # New Metric Checklist
    'NewMetricChecklist',
    'MetricDefinition',
    'ChecklistItem',
    'ChecklistItemStatus',
    'ChecklistResult',
    'EXAMPLE_METRICS',
]
