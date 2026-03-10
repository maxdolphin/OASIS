#!/usr/bin/env python3
"""
Scientific Validation Runner

CLI tool for running scientific validation against published research data.

Usage:
    # Validate all networks against published data
    python scripts/run_scientific_validation.py --all

    # Validate specific network
    python scripts/run_scientific_validation.py --network crystal_river_creek

    # Generate HTML report
    python scripts/run_scientific_validation.py --all --report html

    # Generate markdown report
    python scripts/run_scientific_validation.py --all --report markdown --output report.md

    # Check specific metric across all networks
    python scripts/run_scientific_validation.py --metric relative_ascendency

    # List available networks
    python scripts/run_scientific_validation.py --list-networks

    # Verbose output
    python scripts/run_scientific_validation.py --all --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.scientific_validation_agent import (
    ScientificValidationAgent,
    ValidationStatus,
)
from src.services.published_metrics_db import (
    list_networks,
    list_metrics,
    get_network_info,
    PUBLISHED_METRICS,
)


def print_colored(text: str, color: str = "default"):
    """Print colored text for terminal output."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "bold": "\033[1m",
        "default": "\033[0m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")


def status_symbol(status: ValidationStatus) -> str:
    """Get status symbol for display."""
    symbols = {
        ValidationStatus.PASS: "[PASS]",
        ValidationStatus.FAIL: "[FAIL]",
        ValidationStatus.WARNING: "[WARN]",
        ValidationStatus.SKIP: "[SKIP]",
        ValidationStatus.ERROR: "[ERR]"
    }
    return symbols.get(status, "[???]")


def status_color(status: ValidationStatus) -> str:
    """Get color for status."""
    colors = {
        ValidationStatus.PASS: "green",
        ValidationStatus.FAIL: "red",
        ValidationStatus.WARNING: "yellow",
        ValidationStatus.SKIP: "blue",
        ValidationStatus.ERROR: "red"
    }
    return colors.get(status, "default")


def validate_all(agent: ScientificValidationAgent, verbose: bool = False):
    """Validate all networks."""
    print_colored("\n" + "=" * 70, "bold")
    print_colored("         SCIENTIFIC VALIDATION - ALL NETWORKS", "bold")
    print_colored("=" * 70, "bold")

    report = agent.validate_all()

    print(f"\nNetworks Validated: {report.networks_validated}")
    print(f"Formulas Checked: {report.formulas_checked}")
    print(f"Timestamp: {report.timestamp[:19]}")
    print()

    for result in report.network_results:
        color = status_color(result.overall_status)
        symbol = status_symbol(result.overall_status)
        print_colored(f"{symbol} {result.network_name}", color)
        print(f"    Source: {result.source}")
        print(f"    Summary: {result.summary}")

        if verbose:
            print()
            # Show metric comparisons
            for comp in result.metric_comparisons:
                if comp.published_value is not None:
                    comp_color = status_color(comp.status)
                    comp_symbol = status_symbol(comp.status)
                    error_str = f"{comp.percent_error:.2f}%" if comp.percent_error else "N/A"
                    print_colored(f"      {comp_symbol} {comp.metric_name}: "
                                  f"computed={comp.computed_value:.4f}, "
                                  f"published={comp.published_value:.4f}, "
                                  f"error={error_str}", comp_color)

            # Show validation checks
            print("      Validation Checks:")
            for check in result.validation_checks:
                check_color = status_color(check.status)
                check_symbol = status_symbol(check.status)
                print_colored(f"        {check_symbol} {check.description}: {check.details}", check_color)

        print()

    print_colored("-" * 70, "bold")
    overall_color = status_color(report.overall_status)
    print_colored(f"OVERALL: {report.summary}", overall_color)
    print_colored("=" * 70 + "\n", "bold")

    return report.overall_status == ValidationStatus.PASS


def validate_network(agent: ScientificValidationAgent, network_id: str, verbose: bool = False):
    """Validate a single network."""
    print_colored(f"\n{'=' * 70}", "bold")
    print_colored(f"         VALIDATING: {network_id}", "bold")
    print_colored("=" * 70, "bold")

    result = agent.validate_network(network_id)

    color = status_color(result.overall_status)
    print()
    print(f"Network: {result.network_name}")
    print(f"Source: {result.source}")
    print(f"Timestamp: {result.timestamp[:19]}")
    print_colored(f"Status: {status_symbol(result.overall_status)}", color)
    print()

    # Show metric comparisons
    print("Metric Comparisons:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Published':>12} {'Computed':>12} {'Error':>10} {'Status':>8}")
    print("-" * 70)

    for comp in result.metric_comparisons:
        pub_str = f"{comp.published_value:.4f}" if comp.published_value else "N/A"
        error_str = f"{comp.percent_error:.2f}%" if comp.percent_error else "N/A"
        comp_color = status_color(comp.status)
        print_colored(f"{comp.metric_name:<30} {pub_str:>12} {comp.computed_value:>12.4f} "
                      f"{error_str:>10} {status_symbol(comp.status):>8}", comp_color)

    print()
    print("Validation Checks:")
    print("-" * 70)

    for check in result.validation_checks:
        check_color = status_color(check.status)
        print_colored(f"  {status_symbol(check.status)} {check.description}: {check.details}", check_color)

    print()
    print_colored("-" * 70, "bold")
    print_colored(f"Summary: {result.summary}", color)
    print_colored("=" * 70 + "\n", "bold")

    if verbose:
        print("Computed Metrics:")
        for name, value in sorted(result.computed_metrics.items()):
            if isinstance(value, float):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")
        print()

    return result.overall_status == ValidationStatus.PASS


def validate_metric(agent: ScientificValidationAgent, metric_name: str):
    """Validate a specific metric across all networks."""
    print_colored(f"\n{'=' * 70}", "bold")
    print_colored(f"         VALIDATING METRIC: {metric_name}", "bold")
    print_colored("=" * 70, "bold")

    result = agent.validate_formula(metric_name)

    print(f"\nMetric: {result['formula']}")
    print(f"Networks Checked: {result['networks_checked']}")
    print()

    print("-" * 70)
    print(f"{'Network':<35} {'Published':>12} {'Computed':>12} {'Error':>10} {'Status':>8}")
    print("-" * 70)

    all_pass = True
    for network_id, data in result['results'].items():
        pub_str = f"{data['published']:.4f}" if data['published'] else "N/A"
        error_str = f"{data['percent_error']:.2f}%" if data['percent_error'] else "N/A"

        status = ValidationStatus(data['status'])
        color = status_color(status)

        if status != ValidationStatus.PASS:
            all_pass = False

        print_colored(f"{network_id:<35} {pub_str:>12} {data['computed']:>12.4f} "
                      f"{error_str:>10} {status_symbol(status):>8}", color)

    print_colored("=" * 70 + "\n", "bold")

    return all_pass


def list_available_networks():
    """List all available networks."""
    print_colored("\n" + "=" * 70, "bold")
    print_colored("         AVAILABLE NETWORKS", "bold")
    print_colored("=" * 70, "bold")
    print()

    for network_id in list_networks():
        info = get_network_info(network_id)
        if info:
            print_colored(f"  {network_id}", "blue")
            print(f"    Source: {info['source']}")
            print(f"    Metrics: {', '.join(info['metrics'].keys())}")
            print(f"    Tolerance: {info['tolerance']*100:.1f}%")
            print()

    print_colored("=" * 70 + "\n", "bold")


def generate_report(agent: ScientificValidationAgent, format: str, output: str = None):
    """Generate a validation report."""
    print(f"Generating {format} report...")

    report_content = agent.generate_report(format=format)

    if output:
        with open(output, 'w') as f:
            f.write(report_content)
        print_colored(f"Report saved to: {output}", "green")
    else:
        print(report_content)


def main():
    parser = argparse.ArgumentParser(
        description="Scientific Validation Runner - Validate metrics against published research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Validate all networks"
    )
    group.add_argument(
        "--network",
        type=str,
        metavar="NETWORK_ID",
        help="Validate a specific network"
    )
    group.add_argument(
        "--metric",
        type=str,
        metavar="METRIC_NAME",
        help="Validate a specific metric across all networks"
    )
    group.add_argument(
        "--list-networks",
        action="store_true",
        help="List all available networks"
    )

    # Output options
    parser.add_argument(
        "--report",
        type=str,
        choices=["text", "html", "markdown"],
        help="Generate a formatted report"
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        help="Output file for report (default: stdout)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed metrics"
    )

    args = parser.parse_args()

    # Initialize agent
    agent = ScientificValidationAgent(base_path=project_root)

    # Execute based on mode
    success = True

    if args.list_networks:
        list_available_networks()

    elif args.all:
        if args.report:
            generate_report(agent, args.report, args.output)
        else:
            success = validate_all(agent, args.verbose)

    elif args.network:
        if args.network not in list_networks():
            print_colored(f"Error: Network '{args.network}' not found.", "red")
            print("Available networks:")
            for net in list_networks():
                print(f"  - {net}")
            sys.exit(1)
        success = validate_network(agent, args.network, args.verbose)

    elif args.metric:
        success = validate_metric(agent, args.metric)

    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
