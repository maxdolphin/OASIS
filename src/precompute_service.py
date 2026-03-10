"""
Precomputation Service for Large Network Analysis

This module provides background precomputation and caching services for
expensive network metrics, enabling efficient analysis of large networks
(200+ nodes).

Architecture:
- Tier 1: Always live (O(1)/O(n) metrics) - computed on demand
- Tier 2: Precompute + Cache (O(n²) metrics) - vectorized computation
- Tier 3: Background Job (O(n³+) metrics) - async computation

Caching Layers:
- Layer 1: In-memory cache (session duration)
- Layer 2: Disk cache (JSON, configurable TTL)
- Layer 3: Background job queue for expensive computations
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import time
import logging

from vectorized_metrics import (
    precompute_sums,
    get_all_vectorized_metrics,
    VectorizedMetricsCalculator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecomputeService:
    """
    Background precomputation service for expensive network metrics.

    This service manages caching and background computation of metrics
    to enable efficient analysis of large networks.
    """

    # Processing mode thresholds
    FULL_MODE_THRESHOLD = 50      # Full analysis for <= 50 nodes
    OPTIMIZED_THRESHOLD = 200     # Optimized analysis for <= 200 nodes
    SCALABLE_THRESHOLD = 1000     # Scalable analysis for <= 1000 nodes
    # Massive mode for > 1000 nodes

    # Default cache settings
    DEFAULT_CACHE_TTL_HOURS = 24
    DEFAULT_CACHE_DIR = ".cache/metrics"

    def __init__(self,
                 cache_dir: Optional[Path] = None,
                 cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
                 max_workers: int = 2):
        """
        Initialize the precomputation service.

        Args:
            cache_dir: Directory for disk cache (default: .cache/metrics)
            cache_ttl_hours: Time-to-live for cached results in hours
            max_workers: Maximum number of background worker threads
        """
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_workers = max_workers

        # In-memory cache for fast access
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_times: Dict[str, datetime] = {}

        # Background job management
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_jobs: Dict[str, Future] = {}
        self._job_results: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_cache_key(self, flow_matrix: np.ndarray, node_names: Optional[list] = None) -> str:
        """
        Generate a unique hash key from network structure.

        The key is based on:
        1. Matrix shape (number of nodes)
        2. Matrix values (flow magnitudes)
        3. Optional node names (if provided)

        Args:
            flow_matrix: Square matrix of flows
            node_names: Optional list of node names

        Returns:
            SHA256 hash string identifying this network
        """
        flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

        # Create hash from matrix content
        hasher = hashlib.sha256()

        # Add matrix shape
        hasher.update(str(flow_matrix.shape).encode())

        # Add matrix values (rounded to avoid floating point issues)
        matrix_bytes = np.round(flow_matrix, decimals=10).tobytes()
        hasher.update(matrix_bytes)

        # Add node names if provided
        if node_names:
            names_str = ",".join(str(n) for n in node_names)
            hasher.update(names_str.encode())

        return hasher.hexdigest()[:16]  # First 16 chars is sufficient

    def get_processing_mode(self, n_nodes: int) -> str:
        """
        Determine the appropriate processing mode based on network size.

        Args:
            n_nodes: Number of nodes in the network

        Returns:
            Processing mode string: 'full', 'optimized', 'scalable', or 'massive'
        """
        if n_nodes <= self.FULL_MODE_THRESHOLD:
            return 'full'
        elif n_nodes <= self.OPTIMIZED_THRESHOLD:
            return 'optimized'
        elif n_nodes <= self.SCALABLE_THRESHOLD:
            return 'scalable'
        else:
            return 'massive'

    def precompute_tier2(self, flow_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute Tier 2 metrics using vectorized operations.

        Tier 2 metrics are O(n²) and include:
        - flow_diversity (H)
        - ami (mutual information)
        - ascendency (A)
        - development_capacity (C)
        - reserve (Φ)
        - relative_ascendency (α)
        - robustness (R)
        - effective_flows/nodes/connectivity
        - number_of_roles

        Args:
            flow_matrix: Square matrix of flows

        Returns:
            Dictionary of computed metrics
        """
        flow_matrix = np.asarray(flow_matrix, dtype=np.float64)
        return get_all_vectorized_metrics(flow_matrix)

    def precompute_tier3_async(self,
                               flow_matrix: np.ndarray,
                               node_names: list,
                               calculator_class: Any = None) -> str:
        """
        Start background computation of Tier 3 (expensive) metrics.

        Tier 3 metrics include:
        - finn_cycling_index (cycle detection)
        - autocatalytic_index (cycle enumeration)
        - trophic_depth (path analysis)
        - marginal_contributions (sensitivity analysis)

        Args:
            flow_matrix: Square matrix of flows
            node_names: List of node names
            calculator_class: Optional calculator class to use

        Returns:
            Job ID string for tracking progress
        """
        cache_key = self.get_cache_key(flow_matrix, node_names)
        job_id = f"tier3_{cache_key}"

        # Check if job already running or completed
        with self._lock:
            if job_id in self._pending_jobs:
                logger.info(f"Job {job_id} already running")
                return job_id
            if job_id in self._job_results:
                logger.info(f"Job {job_id} already completed")
                return job_id

        # Submit background job
        future = self._executor.submit(
            self._compute_tier3_metrics,
            flow_matrix.copy(),
            node_names.copy(),
            cache_key
        )

        with self._lock:
            self._pending_jobs[job_id] = future

        logger.info(f"Started background job {job_id}")
        return job_id

    def _compute_tier3_metrics(self,
                               flow_matrix: np.ndarray,
                               node_names: list,
                               cache_key: str) -> Dict[str, Any]:
        """
        Internal method to compute Tier 3 metrics in background.

        Args:
            flow_matrix: Square matrix of flows
            node_names: List of node names
            cache_key: Cache key for storing results

        Returns:
            Dictionary of computed metrics
        """
        job_id = f"tier3_{cache_key}"
        results: Dict[str, Any] = {
            'status': 'computing',
            'progress': 0,
            'started_at': datetime.now().isoformat()
        }

        try:
            n_nodes = len(node_names)

            # Import UlanowiczCalculator here to avoid circular imports
            from ulanowicz_calculator import UlanowiczCalculator
            calc = UlanowiczCalculator(flow_matrix, node_names)

            # Finn Cycling Index (skip for large networks)
            results['progress'] = 25
            if n_nodes <= 15:
                try:
                    results['finn_cycling_index'] = calc.calculate_finn_cycling_index()
                except Exception as e:
                    logger.warning(f"FCI calculation failed: {e}")
                    results['finn_cycling_index'] = None
            else:
                results['finn_cycling_index'] = None
                results['finn_cycling_index_note'] = f"Skipped for {n_nodes} nodes (>15)"

            # Autocatalytic Index
            results['progress'] = 50
            if n_nodes <= 50:
                try:
                    autocatalytic = calc.calculate_autocatalytic_index()
                    results['autocatalytic_index'] = autocatalytic.get('autocatalytic_index', 0)
                    results['cycle_count'] = autocatalytic.get('count', 0)
                except Exception as e:
                    logger.warning(f"Autocatalytic calculation failed: {e}")
                    results['autocatalytic_index'] = None
            else:
                results['autocatalytic_index'] = None
                results['autocatalytic_index_note'] = f"Skipped for {n_nodes} nodes (>50)"

            # Trophic Depth
            results['progress'] = 75
            if n_nodes <= 30:
                try:
                    results['trophic_depth'] = calc.calculate_trophic_depth()
                except Exception as e:
                    logger.warning(f"Trophic depth calculation failed: {e}")
                    results['trophic_depth'] = None
            else:
                results['trophic_depth'] = None
                results['trophic_depth_note'] = f"Skipped for {n_nodes} nodes (>30)"

            # Network topology (if not too large)
            results['progress'] = 90
            if n_nodes <= 100:
                try:
                    topology = calc.calculate_network_topology_metrics()
                    results['topology_metrics'] = topology
                except Exception as e:
                    logger.warning(f"Topology calculation failed: {e}")
                    results['topology_metrics'] = None
            else:
                results['topology_metrics'] = None

            results['status'] = 'completed'
            results['progress'] = 100
            results['completed_at'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Tier 3 computation failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)

        # Store results
        with self._lock:
            self._job_results[job_id] = results
            if job_id in self._pending_jobs:
                del self._pending_jobs[job_id]

        # Save to disk cache
        self.save_to_cache(cache_key, results, tier='tier3')

        return results

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a background computation job.

        Args:
            job_id: Job ID returned by precompute_tier3_async

        Returns:
            Dictionary with job status and results if complete
        """
        with self._lock:
            if job_id in self._job_results:
                return self._job_results[job_id]
            elif job_id in self._pending_jobs:
                return {'status': 'running', 'progress': 0}
            else:
                return {'status': 'not_found'}

    def wait_for_job(self, job_id: str, timeout: float = 300) -> Dict[str, Any]:
        """
        Wait for a background job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Job results or timeout status
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            if status['status'] in ['completed', 'failed', 'not_found']:
                return status
            time.sleep(0.5)

        return {'status': 'timeout'}

    def load_cached(self,
                    cache_key: str,
                    tier: str = 'tier2') -> Optional[Dict[str, Any]]:
        """
        Load precomputed results from cache.

        Checks memory cache first, then disk cache.

        Args:
            cache_key: Network hash key
            tier: Cache tier ('tier2' or 'tier3')

        Returns:
            Cached results or None if not found/expired
        """
        full_key = f"{tier}_{cache_key}"

        # Check memory cache
        if full_key in self._memory_cache:
            cache_time = self._memory_cache_times.get(full_key)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                logger.debug(f"Memory cache hit: {full_key}")
                return self._memory_cache[full_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{full_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Check TTL
                cached_at = datetime.fromisoformat(data.get('cached_at', '1970-01-01'))
                if datetime.now() - cached_at < self.cache_ttl:
                    logger.debug(f"Disk cache hit: {full_key}")
                    # Populate memory cache
                    self._memory_cache[full_key] = data['results']
                    self._memory_cache_times[full_key] = cached_at
                    return data['results']
                else:
                    logger.debug(f"Disk cache expired: {full_key}")
                    cache_file.unlink()  # Remove expired cache
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")

        return None

    def save_to_cache(self,
                      cache_key: str,
                      results: Dict[str, Any],
                      tier: str = 'tier2') -> None:
        """
        Persist computed results to cache.

        Args:
            cache_key: Network hash key
            results: Computed metrics to cache
            tier: Cache tier ('tier2' or 'tier3')
        """
        full_key = f"{tier}_{cache_key}"
        cache_time = datetime.now()

        # Update memory cache
        self._memory_cache[full_key] = results
        self._memory_cache_times[full_key] = cache_time

        # Write to disk cache
        cache_file = self.cache_dir / f"{full_key}.json"

        # Convert numpy types to native Python types for JSON serialization
        serializable_results = self._make_serializable(results)

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'cached_at': cache_time.isoformat(),
                    'cache_key': cache_key,
                    'tier': tier,
                    'results': serializable_results
                }, f, indent=2)
            logger.debug(f"Saved to disk cache: {full_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {e}")

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

    def invalidate_cache(self, cache_key: Optional[str] = None) -> int:
        """
        Invalidate cached results.

        Args:
            cache_key: Specific key to invalidate, or None for all

        Returns:
            Number of cache entries invalidated
        """
        count = 0

        if cache_key:
            # Invalidate specific key
            for tier in ['tier2', 'tier3']:
                full_key = f"{tier}_{cache_key}"

                if full_key in self._memory_cache:
                    del self._memory_cache[full_key]
                    if full_key in self._memory_cache_times:
                        del self._memory_cache_times[full_key]
                    count += 1

                cache_file = self.cache_dir / f"{full_key}.json"
                if cache_file.exists():
                    cache_file.unlink()
                    count += 1
        else:
            # Invalidate all
            self._memory_cache.clear()
            self._memory_cache_times.clear()

            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                count += 1

        logger.info(f"Invalidated {count} cache entries")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        memory_entries = len(self._memory_cache)
        disk_files = list(self.cache_dir.glob("*.json"))
        disk_entries = len(disk_files)

        total_disk_size = sum(f.stat().st_size for f in disk_files)

        return {
            'memory_entries': memory_entries,
            'disk_entries': disk_entries,
            'total_disk_size_bytes': total_disk_size,
            'total_disk_size_mb': total_disk_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
            'pending_jobs': len(self._pending_jobs),
            'completed_jobs': len(self._job_results)
        }

    def compute_with_cache(self,
                           flow_matrix: np.ndarray,
                           node_names: Optional[list] = None,
                           force_recompute: bool = False) -> Tuple[Dict[str, Any], bool]:
        """
        Compute metrics with automatic caching.

        This is the main entry point for cached computation.

        Args:
            flow_matrix: Square matrix of flows
            node_names: Optional list of node names
            force_recompute: If True, ignore cache and recompute

        Returns:
            Tuple of (results dict, was_cached bool)
        """
        flow_matrix = np.asarray(flow_matrix, dtype=np.float64)
        cache_key = self.get_cache_key(flow_matrix, node_names)

        # Check cache unless forced recompute
        if not force_recompute:
            cached = self.load_cached(cache_key, tier='tier2')
            if cached is not None:
                return cached, True

        # Compute Tier 2 metrics
        results = self.precompute_tier2(flow_matrix)

        # Save to cache
        self.save_to_cache(cache_key, results, tier='tier2')

        return results, False

    def shutdown(self):
        """Shut down the executor and clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info("PrecomputeService shut down")


# Singleton instance for global access
_service_instance: Optional[PrecomputeService] = None
_service_lock = threading.Lock()


def get_precompute_service(cache_dir: Optional[Path] = None) -> PrecomputeService:
    """
    Get or create the singleton PrecomputeService instance.

    This function is thread-safe and suitable for use with Streamlit's
    @st.cache_resource decorator.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        PrecomputeService singleton instance
    """
    global _service_instance

    with _service_lock:
        if _service_instance is None:
            _service_instance = PrecomputeService(cache_dir=cache_dir)
            logger.info("Created PrecomputeService singleton")
        return _service_instance


def compute_metrics_cached(flow_matrix: np.ndarray,
                           node_names: Optional[list] = None,
                           include_tier3: bool = False) -> Dict[str, Any]:
    """
    Convenience function for cached metric computation.

    This function handles caching automatically and can optionally
    start background computation of Tier 3 metrics.

    Args:
        flow_matrix: Square matrix of flows
        node_names: Optional list of node names
        include_tier3: If True, start background Tier 3 computation

    Returns:
        Dictionary of computed metrics
    """
    service = get_precompute_service()

    # Get Tier 2 metrics (with caching)
    results, was_cached = service.compute_with_cache(flow_matrix, node_names)

    # Optionally start Tier 3 computation
    if include_tier3 and node_names:
        job_id = service.precompute_tier3_async(flow_matrix, node_names)
        results['tier3_job_id'] = job_id

    results['_was_cached'] = was_cached
    results['_cache_key'] = service.get_cache_key(flow_matrix, node_names)

    return results
