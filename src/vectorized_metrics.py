"""
Vectorized Metrics Module for Large Network Analysis

This module provides numpy-optimized implementations of Ulanowicz ecosystem metrics.
All functions use vectorized operations to achieve O(n²) complexity instead of O(n³)
from redundant loop calculations.

The implementations follow the same formulas as the original UlanowiczCalculator
but precompute row/column sums once and use numpy broadcasting for efficiency.

References:
- Ulanowicz et al. (2009) "Quantifying sustainability: Resilience, efficiency
  and the return of information theory"
- Zorach & Ulanowicz (2003) "Quantifying the Complexity of Flow Networks"
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings


def precompute_sums(flow_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Precompute row sums, column sums, and total system throughput.

    This is the key optimization - computing these O(n) once instead of
    O(n) times n² = O(n³) inside nested loops.

    Args:
        flow_matrix: Square matrix of flows between nodes

    Returns:
        Tuple of (row_sums, col_sums, tst):
        - row_sums: Output throughput for each node (sum along axis=1)
        - col_sums: Input throughput for each node (sum along axis=0)
        - tst: Total System Throughput (sum of all flows)
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)
    row_sums = np.sum(flow_matrix, axis=1)  # Output throughput (T_i.)
    col_sums = np.sum(flow_matrix, axis=0)  # Input throughput (T_.j)
    tst = np.sum(flow_matrix)               # Total System Throughput (T..)

    return row_sums, col_sums, tst


def vectorized_flow_diversity(flow_matrix: np.ndarray,
                               tst: Optional[float] = None) -> float:
    """
    Calculate Flow Diversity (H) using vectorized Shannon entropy.

    Flow diversity measures the evenness of flow distribution across all
    network connections. Higher diversity indicates more evenly distributed flows.

    Formula: H = -Σ(p_ij * ln(p_ij)) where p_ij = T_ij / TST

    Args:
        flow_matrix: Square matrix of flows between nodes
        tst: Precomputed total system throughput (computed if not provided)

    Returns:
        Flow Diversity value in nats (natural log units)
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    if tst is None:
        tst = np.sum(flow_matrix)

    if tst == 0:
        return 0.0

    # Compute probabilities for all non-zero flows
    p = flow_matrix / tst

    # Use numpy where to avoid log(0) - set 0 * log(0) = 0
    # This is vectorized: one operation on entire matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log(p), 0)
        entropy_terms = np.where(p > 0, p * log_p, 0)

    return -np.sum(entropy_terms)


def vectorized_ami(flow_matrix: np.ndarray,
                   row_sums: Optional[np.ndarray] = None,
                   col_sums: Optional[np.ndarray] = None,
                   tst: Optional[float] = None) -> float:
    """
    Calculate Average Mutual Information (AMI) using vectorized operations.

    AMI measures the degree of organization or constraint in the network.
    Higher AMI indicates more organized, less random flow patterns.

    Formula: AMI = Σ(T_ij * ln(T_ij * TST / (T_i. * T_.j))) / TST

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs (computed if not provided)
        col_sums: Precomputed input throughputs (computed if not provided)
        tst: Precomputed total system throughput (computed if not provided)

    Returns:
        Average Mutual Information value in nats
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    if row_sums is None or col_sums is None or tst is None:
        row_sums, col_sums, tst = precompute_sums(flow_matrix)

    if tst == 0:
        return 0.0

    # Create outer product matrix for denominator: T_i. * T_.j
    # This is the key vectorization: instead of computing row_sums[i] * col_sums[j]
    # in a nested loop, we compute all products at once
    outer_product = np.outer(row_sums, col_sums)

    # Compute ratio: T_ij * TST / (T_i. * T_.j)
    # Only for positions where flow > 0 and outer_product > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        # Create mask for valid positions
        valid_mask = (flow_matrix > 0) & (outer_product > 0)

        # Compute ratios
        ratios = np.zeros_like(flow_matrix)
        ratios[valid_mask] = (flow_matrix[valid_mask] * tst) / outer_product[valid_mask]

        # Compute log of ratios
        log_ratios = np.where(ratios > 0, np.log(ratios), 0)

        # Compute AMI terms: T_ij * log(ratio)
        ami_terms = np.where(valid_mask, flow_matrix * log_ratios, 0)

    return np.sum(ami_terms) / tst


def vectorized_ascendency(flow_matrix: np.ndarray,
                          row_sums: Optional[np.ndarray] = None,
                          col_sums: Optional[np.ndarray] = None,
                          tst: Optional[float] = None) -> float:
    """
    Calculate Ascendency (A) using vectorized operations.

    Ascendency is the scaled mutual constraint representing the system's
    organized power. From Ulanowicz et al. (2009) Eq. (12).

    Formula: A = Σ(T_ij * ln(T_ij * T·· / (T_i· * T_·j)))

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Ascendency value (flow-nats)
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    if row_sums is None or col_sums is None or tst is None:
        row_sums, col_sums, tst = precompute_sums(flow_matrix)

    if tst == 0:
        return 0.0

    # Outer product for denominator
    outer_product = np.outer(row_sums, col_sums)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = (flow_matrix > 0) & (outer_product > 0)

        ratios = np.zeros_like(flow_matrix)
        ratios[valid_mask] = (flow_matrix[valid_mask] * tst) / outer_product[valid_mask]

        log_ratios = np.where(ratios > 0, np.log(ratios), 0)
        ascendency_terms = np.where(valid_mask, flow_matrix * log_ratios, 0)

    return np.sum(ascendency_terms)


def vectorized_development_capacity(flow_matrix: np.ndarray,
                                     tst: Optional[float] = None) -> float:
    """
    Calculate Development Capacity (C) using vectorized operations.

    Development Capacity represents the scaled system indeterminacy -
    the capacity for system development and change.
    From Ulanowicz et al. (2009) Eq. (11).

    Formula: C = -Σ(T_ij * ln(T_ij / T··))

    Args:
        flow_matrix: Square matrix of flows between nodes
        tst: Precomputed total system throughput

    Returns:
        Development Capacity value (flow-nats)
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    if tst is None:
        tst = np.sum(flow_matrix)

    if tst == 0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = flow_matrix > 0

        # T_ij * ln(T_ij / T··)
        log_ratios = np.where(valid_mask, np.log(flow_matrix / tst), 0)
        capacity_terms = np.where(valid_mask, flow_matrix * log_ratios, 0)

    return -np.sum(capacity_terms)


def vectorized_reserve(flow_matrix: np.ndarray,
                       row_sums: Optional[np.ndarray] = None,
                       col_sums: Optional[np.ndarray] = None,
                       tst: Optional[float] = None) -> float:
    """
    Calculate Reserve (Φ) using the fundamental relationship.

    Reserve represents the system's flexibility and reserve capacity.
    From the fundamental relationship: C = A + Φ, therefore Φ = C - A

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Reserve value (flow-nats)
    """
    if row_sums is None or col_sums is None or tst is None:
        row_sums, col_sums, tst = precompute_sums(flow_matrix)

    capacity = vectorized_development_capacity(flow_matrix, tst)
    ascendency = vectorized_ascendency(flow_matrix, row_sums, col_sums, tst)

    return capacity - ascendency


def vectorized_relative_ascendency(flow_matrix: np.ndarray,
                                    row_sums: Optional[np.ndarray] = None,
                                    col_sums: Optional[np.ndarray] = None,
                                    tst: Optional[float] = None) -> float:
    """
    Calculate Relative Ascendency (α = A/C) using vectorized operations.

    Relative ascendency is the key sustainability metric representing
    the fraction of total system capacity that is realized as organization.

    Optimal range: 0.2 - 0.6 for sustainability
    Peak robustness: ~0.37

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Relative Ascendency (dimensionless ratio between 0 and 1)
    """
    if row_sums is None or col_sums is None or tst is None:
        row_sums, col_sums, tst = precompute_sums(flow_matrix)

    ascendency = vectorized_ascendency(flow_matrix, row_sums, col_sums, tst)
    capacity = vectorized_development_capacity(flow_matrix, tst)

    if capacity == 0:
        return 0.0

    return ascendency / capacity


def vectorized_effective_flows(flow_matrix: np.ndarray,
                                tst: Optional[float] = None) -> float:
    """
    Calculate effective number of flows (F) using vectorized operations.

    Based on Zorach & Ulanowicz (2003), the effective number of flows
    is the exponential of the flow diversity (Shannon entropy).

    Formula: F = exp(H) = exp(-Σ((T_ij/T··) * ln(T_ij/T··)))

    Args:
        flow_matrix: Square matrix of flows between nodes
        tst: Precomputed total system throughput

    Returns:
        Effective number of flows
    """
    flow_diversity = vectorized_flow_diversity(flow_matrix, tst)
    return np.exp(flow_diversity)


def vectorized_effective_nodes(flow_matrix: np.ndarray,
                                row_sums: Optional[np.ndarray] = None,
                                col_sums: Optional[np.ndarray] = None,
                                tst: Optional[float] = None) -> float:
    """
    Calculate effective number of nodes (N) using vectorized operations.

    Based on the weighted distribution of node throughputs.
    Formula: N = exp(0.5 * Σ((T_ij/T··) * ln(T··² / (T_i· * T_·j))))

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Effective number of nodes
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    if row_sums is None or col_sums is None or tst is None:
        row_sums, col_sums, tst = precompute_sums(flow_matrix)

    if tst == 0:
        return float(flow_matrix.shape[0])

    # Outer product: T_i· * T_·j
    outer_product = np.outer(row_sums, col_sums)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = (flow_matrix > 0) & (outer_product > 0)

        # Weight: T_ij / T··
        weights = flow_matrix / tst

        # Ratio: T··² / (T_i· * T_·j)
        ratios = np.zeros_like(flow_matrix)
        ratios[valid_mask] = (tst ** 2) / outer_product[valid_mask]

        # Log terms
        log_ratios = np.where(ratios > 0, np.log(ratios), 0)

        # Weighted sum
        sum_term = np.sum(np.where(valid_mask, weights * log_ratios, 0))

    return np.exp(0.5 * sum_term)


def vectorized_effective_connectivity(flow_matrix: np.ndarray,
                                       row_sums: Optional[np.ndarray] = None,
                                       col_sums: Optional[np.ndarray] = None,
                                       tst: Optional[float] = None) -> float:
    """
    Calculate effective connectivity (C) using vectorized operations.

    Based on Zorach & Ulanowicz (2003), effective connectivity is the number
    of effective flows per effective node.

    Formula: C = F / N  (Zorach & Ulanowicz 2003, p.72: "C ≡ F/N").
    This is the average number of flows per node, bounded below by 1.0 for a
    connected network (Ulanowicz 2004, p.334). The identities R = F/C² = N/C
    follow.

    NOTE (Track-1 correction): the previous form C = exp(0.5·Σ w·ln(Tij²/…))
    carries a positive exponent; the canonical form (Z-U 2003 Appendix p.76)
    has a NEGATIVE exponent. The positive form equals N/F (the reciprocal,
    always < 1) and violates the C ≥ 1 floor. Computing C = F/N directly keeps
    this vectorized path in exact agreement with the loop implementation and
    guarantees the identity block.

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Effective connectivity in flows per node (>= 1.0 for a connected net)
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    if row_sums is None or col_sums is None or tst is None:
        row_sums, col_sums, tst = precompute_sums(flow_matrix)

    if tst == 0:
        return 0.0

    eff_nodes = vectorized_effective_nodes(flow_matrix, row_sums, col_sums, tst)
    if eff_nodes <= 0:
        return 0.0

    eff_flows = vectorized_effective_flows(flow_matrix, tst)
    return eff_flows / eff_nodes


def vectorized_number_of_roles(flow_matrix: np.ndarray,
                                row_sums: Optional[np.ndarray] = None,
                                col_sums: Optional[np.ndarray] = None,
                                tst: Optional[float] = None) -> float:
    """
    Calculate number of functional roles (R) using vectorized AMI.

    From Zorach & Ulanowicz (2003): R = exp(AMI)

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Number of functional roles in the network
    """
    ami = vectorized_ami(flow_matrix, row_sums, col_sums, tst)
    return np.exp(ami)


def vectorized_robustness(flow_matrix: np.ndarray,
                          row_sums: Optional[np.ndarray] = None,
                          col_sums: Optional[np.ndarray] = None,
                          tst: Optional[float] = None) -> float:
    """
    Calculate Network Robustness (R) using vectorized operations.

    Robustness measures the system's ability to maintain functionality
    under stress or disturbance.

    Formula: R = -α * ln(α) where α = A/C
    Maximum at α = 1/e ≈ 0.368

    Args:
        flow_matrix: Square matrix of flows between nodes
        row_sums: Precomputed output throughputs
        col_sums: Precomputed input throughputs
        tst: Precomputed total system throughput

    Returns:
        Robustness value (max ~0.368 at optimal α)
    """
    alpha = vectorized_relative_ascendency(flow_matrix, row_sums, col_sums, tst)

    if alpha <= 0 or alpha >= 1:
        return 0.0

    return -alpha * np.log(alpha)


def get_all_vectorized_metrics(flow_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate all vectorized metrics efficiently in a single pass.

    This function precomputes shared values once and uses them across
    all metric calculations, maximizing efficiency.

    Args:
        flow_matrix: Square matrix of flows between nodes

    Returns:
        Dictionary containing all computed metrics
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)

    # Precompute shared values ONCE
    row_sums, col_sums, tst = precompute_sums(flow_matrix)

    # Compute all metrics using precomputed values
    flow_diversity = vectorized_flow_diversity(flow_matrix, tst)
    ami = vectorized_ami(flow_matrix, row_sums, col_sums, tst)
    ascendency = vectorized_ascendency(flow_matrix, row_sums, col_sums, tst)
    capacity = vectorized_development_capacity(flow_matrix, tst)
    reserve = capacity - ascendency  # Fundamental relationship

    # Relative metrics
    relative_ascendency = ascendency / capacity if capacity > 0 else 0.0

    # Robustness
    if 0 < relative_ascendency < 1:
        robustness = -relative_ascendency * np.log(relative_ascendency)
    else:
        robustness = 0.0

    # Effective metrics
    effective_flows = np.exp(flow_diversity)
    effective_nodes = vectorized_effective_nodes(flow_matrix, row_sums, col_sums, tst)
    effective_connectivity = vectorized_effective_connectivity(flow_matrix, row_sums, col_sums, tst)
    number_of_roles = np.exp(ami)

    return {
        'total_system_throughput': tst,
        'flow_diversity': flow_diversity,
        'average_mutual_information': ami,
        'ascendency': ascendency,
        'development_capacity': capacity,
        'reserve': reserve,
        'relative_ascendency': relative_ascendency,
        'robustness': robustness,
        'effective_flows': effective_flows,
        'effective_nodes': effective_nodes,
        'effective_connectivity': effective_connectivity,
        'number_of_roles': number_of_roles,
        # Derived ratios
        'overhead': reserve,  # Alias
        'overhead_ratio': reserve / capacity if capacity > 0 else 0.0,
        'reserve_ratio': reserve / capacity if capacity > 0 else 0.0,
        'network_efficiency': relative_ascendency,  # Alias
        'ascendency_ratio': relative_ascendency,  # Alias
    }


class VectorizedMetricsCalculator:
    """
    Convenience class that wraps vectorized metrics with precomputed sums.

    This class maintains state for efficient repeated calculations on the
    same flow matrix.
    """

    def __init__(self, flow_matrix: np.ndarray):
        """
        Initialize with flow matrix and precompute shared values.

        Args:
            flow_matrix: Square matrix of flows between nodes
        """
        self.flow_matrix = np.asarray(flow_matrix, dtype=np.float64)
        self.n_nodes = self.flow_matrix.shape[0]

        # Precompute once
        self.row_sums, self.col_sums, self.tst = precompute_sums(self.flow_matrix)

        # Cache for computed metrics
        self._cache: Dict[str, float] = {}

    def get_precomputed_sums(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return precomputed row sums, column sums, and TST."""
        return self.row_sums.copy(), self.col_sums.copy(), self.tst

    def calculate_tst(self) -> float:
        """Return precomputed Total System Throughput."""
        return self.tst

    def calculate_flow_diversity(self) -> float:
        """Calculate Flow Diversity using cached TST."""
        if 'flow_diversity' not in self._cache:
            self._cache['flow_diversity'] = vectorized_flow_diversity(
                self.flow_matrix, self.tst
            )
        return self._cache['flow_diversity']

    def calculate_ami(self) -> float:
        """Calculate Average Mutual Information using cached values."""
        if 'ami' not in self._cache:
            self._cache['ami'] = vectorized_ami(
                self.flow_matrix, self.row_sums, self.col_sums, self.tst
            )
        return self._cache['ami']

    def calculate_ascendency(self) -> float:
        """Calculate Ascendency using cached values."""
        if 'ascendency' not in self._cache:
            self._cache['ascendency'] = vectorized_ascendency(
                self.flow_matrix, self.row_sums, self.col_sums, self.tst
            )
        return self._cache['ascendency']

    def calculate_development_capacity(self) -> float:
        """Calculate Development Capacity using cached TST."""
        if 'development_capacity' not in self._cache:
            self._cache['development_capacity'] = vectorized_development_capacity(
                self.flow_matrix, self.tst
            )
        return self._cache['development_capacity']

    def calculate_reserve(self) -> float:
        """Calculate Reserve using fundamental relationship."""
        if 'reserve' not in self._cache:
            self._cache['reserve'] = (
                self.calculate_development_capacity() -
                self.calculate_ascendency()
            )
        return self._cache['reserve']

    def calculate_relative_ascendency(self) -> float:
        """Calculate Relative Ascendency (α = A/C)."""
        if 'relative_ascendency' not in self._cache:
            capacity = self.calculate_development_capacity()
            if capacity > 0:
                self._cache['relative_ascendency'] = (
                    self.calculate_ascendency() / capacity
                )
            else:
                self._cache['relative_ascendency'] = 0.0
        return self._cache['relative_ascendency']

    def calculate_robustness(self) -> float:
        """Calculate Network Robustness."""
        if 'robustness' not in self._cache:
            alpha = self.calculate_relative_ascendency()
            if 0 < alpha < 1:
                self._cache['robustness'] = -alpha * np.log(alpha)
            else:
                self._cache['robustness'] = 0.0
        return self._cache['robustness']

    def calculate_effective_flows(self) -> float:
        """Calculate Effective Number of Flows."""
        if 'effective_flows' not in self._cache:
            self._cache['effective_flows'] = vectorized_effective_flows(
                self.flow_matrix, self.tst
            )
        return self._cache['effective_flows']

    def calculate_effective_nodes(self) -> float:
        """Calculate Effective Number of Nodes."""
        if 'effective_nodes' not in self._cache:
            self._cache['effective_nodes'] = vectorized_effective_nodes(
                self.flow_matrix, self.row_sums, self.col_sums, self.tst
            )
        return self._cache['effective_nodes']

    def calculate_effective_connectivity(self) -> float:
        """Calculate Effective Connectivity."""
        if 'effective_connectivity' not in self._cache:
            self._cache['effective_connectivity'] = vectorized_effective_connectivity(
                self.flow_matrix, self.row_sums, self.col_sums, self.tst
            )
        return self._cache['effective_connectivity']

    def calculate_number_of_roles(self) -> float:
        """Calculate Number of Functional Roles."""
        if 'number_of_roles' not in self._cache:
            self._cache['number_of_roles'] = np.exp(self.calculate_ami())
        return self._cache['number_of_roles']

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics at once, using cached values."""
        return {
            'total_system_throughput': self.calculate_tst(),
            'flow_diversity': self.calculate_flow_diversity(),
            'average_mutual_information': self.calculate_ami(),
            'ascendency': self.calculate_ascendency(),
            'development_capacity': self.calculate_development_capacity(),
            'reserve': self.calculate_reserve(),
            'relative_ascendency': self.calculate_relative_ascendency(),
            'robustness': self.calculate_robustness(),
            'effective_flows': self.calculate_effective_flows(),
            'effective_nodes': self.calculate_effective_nodes(),
            'effective_connectivity': self.calculate_effective_connectivity(),
            'number_of_roles': self.calculate_number_of_roles(),
            # Aliases for compatibility
            'overhead': self.calculate_reserve(),
            'overhead_ratio': self.calculate_reserve() / self.calculate_development_capacity()
                            if self.calculate_development_capacity() > 0 else 0.0,
            'network_efficiency': self.calculate_relative_ascendency(),
        }

    def clear_cache(self):
        """Clear the computation cache."""
        self._cache.clear()
