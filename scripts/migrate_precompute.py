#!/usr/bin/env python3
"""
Migration script to precompute metrics for all existing networks.

Scans all network JSON files, computes metrics, and stores them in the SQLite database.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import get_database_manager, get_precompute_pipeline


def main():
    """Run migration to precompute all existing networks."""
    print("=" * 60)
    print("Network Metrics Migration")
    print("=" * 60)

    # Get pipeline
    pipeline = get_precompute_pipeline()

    def progress_callback(current, total, name):
        status = f"[{current}/{total}]"
        print(f"{status} Processing: {name}")

    # Run migration
    results = pipeline.precompute_all_existing(progress_callback=progress_callback)

    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total files scanned: {results['total']}")
    print(f"Successfully computed: {results['success']}")
    print(f"Skipped (already cached): {results['skipped']}")
    print(f"Errors: {len(results['errors'])}")

    if results['errors']:
        print("\nErrors:")
        for err in results['errors']:
            print(f"  - {err['file']}: {err['error']}")

    # Show database stats
    db = get_database_manager()
    stats = db.get_stats()
    print(f"\nDatabase: {stats['database_path']}")
    print(f"Size: {stats['database_size_mb']:.2f} MB")
    print(f"Networks stored: {stats['network_count']}")
    print(f"Metrics entries: {stats['metrics_entries']}")


if __name__ == "__main__":
    main()
