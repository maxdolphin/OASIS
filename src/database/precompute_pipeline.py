"""
Precompute Pipeline for Network Metrics.

Integrates precomputation with data ingestion, providing automatic
metric computation when networks are loaded.
"""

import json
import time
import logging
import numpy as np
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable

from .db_manager import DatabaseManager, get_database_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecomputePipeline:
    """
    Integrates precomputation with data ingestion.

    Provides:
    - Automatic metric computation on network load
    - Cache-first retrieval with fallback to computation
    - Batch migration for existing networks
    """

    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the precompute pipeline.

        Args:
            db_manager: DatabaseManager instance (uses singleton if None)
        """
        self.db = db_manager or get_database_manager()
        self._vectorized_calculator = None

    def _get_vectorized_metrics(self, flow_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute vectorized metrics for a flow matrix.

        Args:
            flow_matrix: Square flow matrix

        Returns:
            Dictionary of computed metrics
        """
        try:
            from vectorized_metrics import get_all_vectorized_metrics
            return get_all_vectorized_metrics(flow_matrix)
        except ImportError:
            try:
                from src.vectorized_metrics import get_all_vectorized_metrics
                return get_all_vectorized_metrics(flow_matrix)
            except ImportError:
                logger.warning("Vectorized metrics not available, using fallback")
            return self._compute_basic_metrics(flow_matrix)

    def _compute_basic_metrics(self, flow_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute basic metrics without vectorized implementation.

        Args:
            flow_matrix: Square flow matrix

        Returns:
            Dictionary of basic metrics
        """
        n = flow_matrix.shape[0]
        total_flow = float(np.sum(flow_matrix))

        if total_flow == 0:
            return {
                'total_system_throughput': 0.0,
                'ascendency': 0.0,
                'development_capacity': 0.0,
                'reserve': 0.0,
                'relative_ascendency': 0.0,
                'robustness': 0.0
            }

        # Basic throughput
        tst = total_flow

        # Simple flow diversity using entropy
        p_matrix = flow_matrix / total_flow
        p_flat = p_matrix.flatten()
        p_nonzero = p_flat[p_flat > 0]
        flow_diversity = -np.sum(p_nonzero * np.log(p_nonzero)) if len(p_nonzero) > 0 else 0.0

        return {
            'total_system_throughput': float(tst),
            'flow_diversity': float(flow_diversity),
            'num_nodes': n,
            'num_edges': int(np.sum(flow_matrix > 0))
        }

    def on_network_loaded(self,
                          network_data: Dict[str, Any],
                          source_file: str = None) -> Dict[str, Any]:
        """
        Hook called when a network is loaded - triggers precomputation if needed.

        Args:
            network_data: Network data dictionary with 'flow_matrix', 'node_names', 'organization'
            source_file: Optional source file path

        Returns:
            Dictionary containing network_id and precomputed metrics (if available)
        """
        # Support multiple naming conventions
        raw_matrix = network_data.get('flow_matrix', network_data.get('flows', []))
        flow_matrix = np.asarray(raw_matrix, dtype=np.float64)
        node_names = network_data.get('node_names', network_data.get('nodes', []))
        org_name = network_data.get('organization', network_data.get('name', 'Unknown'))

        if flow_matrix.size == 0:
            logger.warning(f"Empty flow matrix for {org_name}")
            return {'error': 'Empty flow matrix'}

        n_nodes = flow_matrix.shape[0]
        n_edges = int(np.sum(flow_matrix > 0))

        # Compute network hash
        network_hash = self.db.compute_network_hash(flow_matrix, node_names)

        # Check if network already exists with metrics
        existing = self.db.get_network_by_hash(network_hash)
        if existing:
            metrics = self.db.get_precomputed_metrics(existing['id'], tier=2)
            if metrics:
                logger.debug(f"Cache hit for {org_name} (hash: {network_hash})")
                return {
                    'network_id': existing['id'],
                    'metrics': metrics,
                    'cached': True
                }

        # Save/update network record
        network_id = self.db.save_network(
            name=org_name,
            source_file=source_file or '',
            node_count=n_nodes,
            edge_count=n_edges,
            network_hash=network_hash
        )

        # Compute and store metrics
        start_time = time.time()
        metrics = self._get_vectorized_metrics(flow_matrix)
        computation_time_ms = int((time.time() - start_time) * 1000)

        self.db.save_precomputed_metrics(
            network_id=network_id,
            tier=2,
            metrics=metrics,
            computation_time_ms=computation_time_ms
        )

        logger.info(f"Precomputed metrics for {org_name} ({n_nodes} nodes) in {computation_time_ms}ms")

        return {
            'network_id': network_id,
            'metrics': metrics,
            'cached': False,
            'computation_time_ms': computation_time_ms
        }

    def get_or_compute_metrics(self,
                                flow_matrix: np.ndarray,
                                node_names: List[str] = None,
                                network_name: str = None) -> Dict[str, Any]:
        """
        Get metrics from DB or compute and store.

        Args:
            flow_matrix: Square flow matrix
            node_names: Optional node names
            network_name: Optional network name for storage

        Returns:
            Dictionary containing metrics and cache status
        """
        flow_matrix = np.asarray(flow_matrix, dtype=np.float64)
        network_hash = self.db.compute_network_hash(flow_matrix, node_names)

        # Check cache first
        existing = self.db.get_network_by_hash(network_hash)
        if existing:
            metrics = self.db.get_precomputed_metrics(existing['id'], tier=2)
            if metrics:
                return {
                    'metrics': metrics,
                    'cached': True,
                    'network_id': existing['id']
                }

        # Need to compute
        n_nodes = flow_matrix.shape[0]
        n_edges = int(np.sum(flow_matrix > 0))

        # Save network record
        network_id = self.db.save_network(
            name=network_name or f"network_{network_hash}",
            source_file='',
            node_count=n_nodes,
            edge_count=n_edges,
            network_hash=network_hash
        )

        # Compute metrics
        start_time = time.time()
        metrics = self._get_vectorized_metrics(flow_matrix)
        computation_time_ms = int((time.time() - start_time) * 1000)

        # Store metrics
        self.db.save_precomputed_metrics(
            network_id=network_id,
            tier=2,
            metrics=metrics,
            computation_time_ms=computation_time_ms
        )

        return {
            'metrics': metrics,
            'cached': False,
            'network_id': network_id,
            'computation_time_ms': computation_time_ms
        }

    def precompute_all_existing(self,
                                 progress_callback: Callable[[int, int, str], None] = None) -> Dict[str, Any]:
        """
        Migrate all existing networks to database with precomputed metrics.

        Args:
            progress_callback: Optional callback(current, total, network_name) for progress updates

        Returns:
            Summary dict with counts and errors
        """
        results = {
            'total': 0,
            'success': 0,
            'skipped': 0,
            'errors': [],
            'networks': []
        }

        # Find all network files
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

        # Collect all JSON files
        network_files = []
        for dir_path in network_dirs:
            if dir_path.exists():
                network_files.extend(dir_path.glob('*.json'))

        results['total'] = len(network_files)

        for idx, filepath in enumerate(network_files):
            network_name = filepath.stem

            if progress_callback:
                progress_callback(idx + 1, results['total'], network_name)

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract flow matrix and node names
                # Support multiple naming conventions: flow_matrix, flows, matrix
                flow_matrix = None
                node_names = None

                if 'flow_matrix' in data:
                    flow_matrix = np.array(data['flow_matrix'])
                    node_names = data.get('node_names', data.get('nodes', []))
                elif 'flows' in data:
                    flow_matrix = np.array(data['flows'])
                    node_names = data.get('nodes', data.get('node_names', []))
                elif 'matrix' in data:
                    flow_matrix = np.array(data['matrix'])
                    node_names = data.get('nodes', [])
                else:
                    logger.warning(f"No flow matrix found in {filepath}")
                    results['errors'].append({
                        'file': str(filepath),
                        'error': 'No flow matrix found'
                    })
                    continue

                if flow_matrix is None or flow_matrix.size == 0:
                    results['errors'].append({
                        'file': str(filepath),
                        'error': 'Empty flow matrix'
                    })
                    continue

                # Check if already computed
                network_hash = self.db.compute_network_hash(flow_matrix, node_names)
                existing = self.db.get_network_by_hash(network_hash)

                if existing:
                    metrics = self.db.get_precomputed_metrics(existing['id'], tier=2)
                    if metrics:
                        results['skipped'] += 1
                        results['networks'].append({
                            'name': network_name,
                            'status': 'skipped',
                            'cached': True
                        })
                        continue

                # Compute and store
                org_name = data.get('organization', data.get('name', network_name))
                result = self.on_network_loaded(
                    network_data={
                        'flow_matrix': flow_matrix,
                        'node_names': node_names,
                        'organization': org_name
                    },
                    source_file=str(filepath)
                )

                if 'error' in result:
                    results['errors'].append({
                        'file': str(filepath),
                        'error': result['error']
                    })
                else:
                    results['success'] += 1
                    results['networks'].append({
                        'name': org_name,
                        'status': 'computed',
                        'time_ms': result.get('computation_time_ms', 0),
                        'network_id': result.get('network_id')
                    })

            except json.JSONDecodeError as e:
                results['errors'].append({
                    'file': str(filepath),
                    'error': f'JSON decode error: {e}'
                })
            except Exception as e:
                results['errors'].append({
                    'file': str(filepath),
                    'error': str(e)
                })

        return results


# Singleton instance
_pipeline_instance: Optional[PrecomputePipeline] = None
_pipeline_lock = threading.Lock()


def get_precompute_pipeline(db_manager: DatabaseManager = None) -> PrecomputePipeline:
    """
    Get or create the singleton PrecomputePipeline instance.

    Args:
        db_manager: Optional DatabaseManager instance

    Returns:
        PrecomputePipeline singleton instance
    """
    global _pipeline_instance

    with _pipeline_lock:
        if _pipeline_instance is None:
            _pipeline_instance = PrecomputePipeline(db_manager)
            logger.info("Created PrecomputePipeline singleton")
        return _pipeline_instance
