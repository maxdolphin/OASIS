#!/usr/bin/env python3
"""
Migration Script: Precompute All Existing Networks

This script migrates all existing network files to the SQLite database
with precomputed metrics.

Usage:
    python scripts/migrate_precompute.py

Options:
    --dry-run    List networks without computing
    --verbose    Show detailed progress
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from database import get_database_manager, get_precompute_pipeline


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def main():
    parser = argparse.ArgumentParser(
        description='Migrate existing networks to SQLite with precomputed metrics'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='List networks without computing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress')
    args = parser.parse_args()

    print("=" * 60)
    print("Network Migration to SQLite Database")
    print("=" * 60)
    print()

    # Initialize
    db = get_database_manager()
    pipeline = get_precompute_pipeline(db)

    # Get initial stats
    stats = db.get_stats()
    print(f"Database: {stats['database_path']}")
    print(f"Existing networks: {stats['network_count']}")
    print(f"Existing metrics: {stats['metrics_entries']}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would process the following networks:")
        print()

        # Just list files
        from database.precompute_pipeline import Path as PPPath
        base_path = Path('data')
        network_dirs = [
            base_path / 'ecosystem_samples',
            base_path / 'synthetic_organizations' / 'combined_flows',
            base_path / 'synthetic_organizations' / 'email_flows',
            base_path / 'synthetic_organizations' / 'document_flows',
            base_path / 'real_world_datasets' / 'energy',
            base_path / 'real_world_datasets' / 'supply_chain',
            base_path / 'real_world_datasets' / 'financial',
            base_path / 'real_world_datasets' / 'trade_materials',
            base_path / 'user_saved_networks',
            base_path / 'extracted_networks',
        ]

        total = 0
        for dir_path in network_dirs:
            if dir_path.exists():
                files = list(dir_path.glob('*.json'))
                if files:
                    print(f"  {dir_path}: {len(files)} files")
                    total += len(files)

        print()
        print(f"Total: {total} network files")
        return

    # Progress callback
    start_time = time.time()

    def progress_callback(current: int, total: int, name: str):
        elapsed = time.time() - start_time
        if current > 1:
            eta = (elapsed / (current - 1)) * (total - current + 1)
        else:
            eta = 0

        if args.verbose:
            print(f"[{current}/{total}] {name}...")
        else:
            # Simple progress bar
            bar_width = 40
            filled = int(bar_width * current / total)
            bar = '=' * filled + '-' * (bar_width - filled)
            print(f"\r[{bar}] {current}/{total} ETA: {format_time(eta)}", end='', flush=True)

    print("Migrating networks...")
    print()

    # Run migration
    results = pipeline.precompute_all_existing(progress_callback=progress_callback)

    if not args.verbose:
        print()  # New line after progress bar

    elapsed = time.time() - start_time

    # Print summary
    print()
    print("=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print()
    print(f"Total networks found: {results['total']}")
    print(f"Successfully computed: {results['success']}")
    print(f"Skipped (already cached): {results['skipped']}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Total time: {format_time(elapsed)}")
    print()

    # Print errors if any
    if results['errors']:
        print("Errors:")
        for err in results['errors']:
            print(f"  - {err['file']}: {err['error']}")
        print()

    # Print network details if verbose
    if args.verbose and results['networks']:
        print("Network details:")
        for net in results['networks']:
            status = net['status']
            if status == 'computed':
                print(f"  - {net['name']}: computed in {net.get('time_ms', 0)}ms")
            else:
                print(f"  - {net['name']}: {status}")
        print()

    # Final stats
    final_stats = db.get_stats()
    print("Final database stats:")
    print(f"  Networks: {final_stats['network_count']}")
    print(f"  Metrics entries: {final_stats['metrics_entries']}")
    print(f"  Database size: {final_stats['database_size_mb']:.2f} MB")


if __name__ == '__main__':
    main()
