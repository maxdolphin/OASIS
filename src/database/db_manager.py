"""
SQLite Database Manager for Networks and Precomputed Metrics.

This module provides persistent storage for network data and their
precomputed metrics, replacing the JSON file-based caching system.
"""

import sqlite3
import hashlib
import json
import numpy as np
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for networks and metrics.

    Provides persistent storage with:
    - Network metadata (name, source, node/edge counts, hash)
    - Precomputed metrics by tier (1, 2, 3)
    - Thread-safe operations
    """

    DEFAULT_DB_PATH = "data/database/networks.db"

    def __init__(self, db_path: str = None):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database file (default: data/database/networks.db)
        """
        self.db_path = Path(db_path or self.DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self.initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Networks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS networks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                source_file TEXT,
                node_count INTEGER,
                edge_count INTEGER,
                network_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Precomputed metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS precomputed_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network_id INTEGER NOT NULL,
                metric_tier INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                computation_time_ms INTEGER,
                FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE,
                UNIQUE(network_id, metric_tier)
            )
        ''')

        # Create indexes for fast lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_network_hash ON networks(network_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_network ON precomputed_metrics(network_id)')

        # HuggingFace Discovery tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovered_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- HuggingFace metadata
                hf_id TEXT UNIQUE NOT NULL,
                hf_author TEXT,
                name TEXT,
                description TEXT,
                tags TEXT,
                license TEXT,

                -- Size metrics
                num_rows INTEGER,
                download_size_bytes INTEGER,

                -- Discovery metadata
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                discovery_keywords TEXT,
                discovery_category TEXT,

                -- Scoring
                total_score REAL,
                structure_score REAL,
                size_score REAL,
                quality_score REAL,
                license_score REAL,
                feasibility_score REAL,
                recommendation TEXT,
                conversion_complexity TEXT,

                -- Approval workflow
                approval_status TEXT DEFAULT 'pending',
                approved_by TEXT,
                approved_at TIMESTAMP,
                rejection_reason TEXT,

                -- Processing
                processing_attempts INTEGER DEFAULT 0,
                last_processing_error TEXT,
                converted_network_id INTEGER,

                FOREIGN KEY (converted_network_id) REFERENCES networks(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discovery_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'running',
                keywords_searched TEXT,
                total_found INTEGER DEFAULT 0,
                high_potential INTEGER DEFAULT 0,
                medium_potential INTEGER DEFAULT 0,
                errors TEXT
            )
        ''')

        # Create indexes for discovery tables
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discovered_hf_id ON discovered_datasets(hf_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discovered_status ON discovered_datasets(approval_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discovered_score ON discovered_datasets(total_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discovered_category ON discovered_datasets(discovery_category)')

        conn.commit()
        logger.debug("Database schema initialized")

    @staticmethod
    def compute_network_hash(flow_matrix: np.ndarray, node_names: Optional[List[str]] = None) -> str:
        """
        Compute a unique hash for a network based on its structure.

        Args:
            flow_matrix: Square matrix of flows
            node_names: Optional list of node names

        Returns:
            SHA256 hash string (first 16 characters)
        """
        flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

        hasher = hashlib.sha256()
        hasher.update(str(flow_matrix.shape).encode())
        hasher.update(np.round(flow_matrix, decimals=10).tobytes())

        if node_names:
            names_str = ",".join(str(n) for n in node_names)
            hasher.update(names_str.encode())

        return hasher.hexdigest()[:16]

    def get_network_by_hash(self, network_hash: str) -> Optional[Dict[str, Any]]:
        """
        Find a network by its hash.

        Args:
            network_hash: Network hash string

        Returns:
            Network record dict or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM networks WHERE network_hash = ?',
            (network_hash,)
        )
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_network_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a network by its name.

        Args:
            name: Network name

        Returns:
            Network record dict or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM networks WHERE name = ?',
            (name,)
        )
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def save_network(self,
                     name: str,
                     source_file: str,
                     node_count: int,
                     edge_count: int,
                     network_hash: str) -> int:
        """
        Save or update a network record.

        Args:
            name: Network name
            source_file: Source file path
            node_count: Number of nodes
            edge_count: Number of edges
            network_hash: Unique network hash

        Returns:
            Network ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO networks (name, source_file, node_count, edge_count, network_hash)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(network_hash) DO UPDATE SET
                    name = excluded.name,
                    source_file = excluded.source_file,
                    node_count = excluded.node_count,
                    edge_count = excluded.edge_count,
                    updated_at = CURRENT_TIMESTAMP
            ''', (name, source_file, node_count, edge_count, network_hash))

            conn.commit()

            # Get the ID
            cursor.execute(
                'SELECT id FROM networks WHERE network_hash = ?',
                (network_hash,)
            )
            row = cursor.fetchone()
            network_id = row['id'] if row else cursor.lastrowid

            logger.debug(f"Saved network {name} with ID {network_id}")
            return network_id

        except sqlite3.IntegrityError as e:
            # Handle unique constraint on name
            cursor.execute('''
                UPDATE networks
                SET source_file = ?, node_count = ?, edge_count = ?,
                    network_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE name = ?
            ''', (source_file, node_count, edge_count, network_hash, name))
            conn.commit()

            cursor.execute('SELECT id FROM networks WHERE name = ?', (name,))
            row = cursor.fetchone()
            return row['id'] if row else 0

    def get_precomputed_metrics(self,
                                 network_id: int,
                                 tier: int = None) -> Optional[Dict[str, Any]]:
        """
        Get precomputed metrics for a network.

        Args:
            network_id: Network ID
            tier: Optional tier filter (1, 2, or 3). If None, returns all tiers merged.

        Returns:
            Dictionary of metrics or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if tier is not None:
            cursor.execute('''
                SELECT metrics_json, computation_time_ms, computed_at
                FROM precomputed_metrics
                WHERE network_id = ? AND metric_tier = ?
            ''', (network_id, tier))
            row = cursor.fetchone()

            if row:
                metrics = json.loads(row['metrics_json'])
                metrics['_computation_time_ms'] = row['computation_time_ms']
                metrics['_computed_at'] = row['computed_at']
                return metrics
            return None
        else:
            # Merge all tiers
            cursor.execute('''
                SELECT metric_tier, metrics_json, computation_time_ms, computed_at
                FROM precomputed_metrics
                WHERE network_id = ?
                ORDER BY metric_tier
            ''', (network_id,))
            rows = cursor.fetchall()

            if not rows:
                return None

            merged = {}
            for row in rows:
                tier_metrics = json.loads(row['metrics_json'])
                merged.update(tier_metrics)

            merged['_tiers_available'] = [row['metric_tier'] for row in rows]
            return merged

    def get_precomputed_metrics_by_hash(self,
                                         network_hash: str,
                                         tier: int = None) -> Optional[Dict[str, Any]]:
        """
        Get precomputed metrics by network hash.

        Args:
            network_hash: Network hash string
            tier: Optional tier filter

        Returns:
            Dictionary of metrics or None if not found
        """
        network = self.get_network_by_hash(network_hash)
        if network:
            return self.get_precomputed_metrics(network['id'], tier)
        return None

    def save_precomputed_metrics(self,
                                  network_id: int,
                                  tier: int,
                                  metrics: Dict[str, Any],
                                  computation_time_ms: int = 0) -> None:
        """
        Save precomputed metrics for a network.

        Args:
            network_id: Network ID
            tier: Metric tier (1, 2, or 3)
            metrics: Dictionary of metric values
            computation_time_ms: Time taken to compute in milliseconds
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Convert numpy types to native Python types
        metrics_serializable = self._make_serializable(metrics)
        metrics_json = json.dumps(metrics_serializable)

        cursor.execute('''
            INSERT INTO precomputed_metrics (network_id, metric_tier, metrics_json, computation_time_ms)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(network_id, metric_tier) DO UPDATE SET
                metrics_json = excluded.metrics_json,
                computation_time_ms = excluded.computation_time_ms,
                computed_at = CURRENT_TIMESTAMP
        ''', (network_id, tier, metrics_json, computation_time_ms))

        conn.commit()
        logger.debug(f"Saved tier {tier} metrics for network {network_id}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def list_networks(self) -> List[Dict[str, Any]]:
        """
        List all networks in the database.

        Returns:
            List of network records
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT n.*,
                   GROUP_CONCAT(DISTINCT pm.metric_tier) as computed_tiers,
                   MAX(pm.computed_at) as last_computed
            FROM networks n
            LEFT JOIN precomputed_metrics pm ON n.id = pm.network_id
            GROUP BY n.id
            ORDER BY n.updated_at DESC
        ''')

        return [dict(row) for row in cursor.fetchall()]

    def delete_network(self, network_id: int) -> bool:
        """
        Delete a network and its precomputed metrics.

        Args:
            network_id: Network ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete metrics first (due to foreign key)
        cursor.execute(
            'DELETE FROM precomputed_metrics WHERE network_id = ?',
            (network_id,)
        )

        # Delete network
        cursor.execute(
            'DELETE FROM networks WHERE id = ?',
            (network_id,)
        )

        deleted = cursor.rowcount > 0
        conn.commit()

        if deleted:
            logger.debug(f"Deleted network {network_id}")

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with counts and size information
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) as count FROM networks')
        network_count = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM precomputed_metrics')
        metrics_count = cursor.fetchone()['count']

        cursor.execute('''
            SELECT COUNT(DISTINCT network_id) as count
            FROM precomputed_metrics
        ''')
        networks_with_metrics = cursor.fetchone()['count']

        # Database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            'network_count': network_count,
            'metrics_entries': metrics_count,
            'networks_with_metrics': networks_with_metrics,
            'database_size_bytes': db_size,
            'database_size_mb': db_size / (1024 * 1024),
            'database_path': str(self.db_path)
        }

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Singleton instance
_db_instance: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


def get_database_manager(db_path: str = None) -> DatabaseManager:
    """
    Get or create the singleton DatabaseManager instance.

    Args:
        db_path: Optional custom database path

    Returns:
        DatabaseManager singleton instance
    """
    global _db_instance

    with _db_lock:
        if _db_instance is None:
            _db_instance = DatabaseManager(db_path)
            logger.info("Created DatabaseManager singleton")
        return _db_instance
