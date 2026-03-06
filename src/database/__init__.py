"""
Database module for network storage and precomputed metrics.

This module provides SQLite-backed persistence for network data and
precomputed metrics, replacing the JSON file-based caching system.
"""

from .db_manager import DatabaseManager, get_database_manager
from .precompute_pipeline import PrecomputePipeline, get_precompute_pipeline

__all__ = [
    'DatabaseManager',
    'get_database_manager',
    'PrecomputePipeline',
    'get_precompute_pipeline',
]
