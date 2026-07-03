"""
Ecosystem Flow Calculator Extension for Ulanowicz Analysis

This module extends the basic Ulanowicz calculator to handle complete ecosystem
flow networks including imports, exports, and respiration (dissipation).

Based on:
1. Ulanowicz et al. (2009) "Quantifying sustainability: Resilience, efficiency, 
   and the return of information theory" Ecological Complexity 6(1):27-36
2. Ulanowicz (2004) "Quantitative methods for ecological network analysis"
   Computational Biology and Chemistry 28:321-339

Key Concepts:
- IMPORTS (Z): External inputs into the system (energy, nutrients from environment)
- EXPORTS (Y): Outputs leaving the system (emigration, harvest, waste)
- RESPIRATION (R): Energy dissipated as heat (metabolic losses, entropy production)
- TST_extended = Internal flows + Imports + Exports + Respiration

In ecological terms:
- Respiration represents energy lost as heat during metabolic processes
- It's the thermodynamic cost of maintaining organization against entropy
- Higher respiration indicates more energy needed to maintain system structure
"""

import numpy as np
from typing import Dict, Optional, Tuple
import sys
sys.path.append('.')
from ulanowicz_calculator import UlanowiczCalculator


class EcosystemFlowCalculator(UlanowiczCalculator):
    """
    Extended Ulanowicz calculator for complete ecosystem flow analysis.
    
    Handles the full ecosystem flow network including boundary flows
    (imports/exports) and dissipative flows (respiration).
    """
    
    def __init__(self, 
                 flow_matrix: np.ndarray,
                 node_names: Optional[list] = None,
                 imports: Optional[np.ndarray] = None,
                 exports: Optional[np.ndarray] = None, 
                 respiration: Optional[np.ndarray] = None):
        """
        Initialize ecosystem flow calculator.
        
        Args:
            flow_matrix: Internal flows between compartments
            node_names: Names of compartments/nodes
            imports: External inputs to each compartment (Z vector)
            exports: Outputs from each compartment (Y vector)
            respiration: Dissipative losses from each compartment (R vector)
        """
        super().__init__(flow_matrix, node_names)
        
        # Initialize boundary flows
        n = self.n_nodes
        self.imports = imports if imports is not None else np.zeros(n)
        self.exports = exports if exports is not None else np.zeros(n)
        self.respiration = respiration if respiration is not None else np.zeros(n)
        
        # Ensure arrays are correct shape
        self.imports = np.asarray(self.imports).reshape(n)
        self.exports = np.asarray(self.exports).reshape(n)
        self.respiration = np.asarray(self.respiration).reshape(n)
        
        # Calculate extended throughput
        self._calculate_extended_throughput()
    
    def _calculate_extended_throughput(self):
        """Calculate throughput including boundary flows."""
        # Input throughput: internal inputs + imports
        self.input_throughput_extended = self.input_throughput + self.imports
        
        # Output throughput: internal outputs + exports + respiration
        self.output_throughput_extended = self.output_throughput + self.exports + self.respiration
        
        # Total throughput for each compartment
        self.total_throughput_extended = self.input_throughput_extended + self.output_throughput_extended
    
    def calculate_tst_extended(self) -> float:
        """
        Calculate Total System Throughput including boundary flows.
        
        TST_extended = Σ(internal flows) + Σ(imports) + Σ(exports) + Σ(respiration)
        
        This represents the total activity of the ecosystem including
        exchanges with the environment.
        
        Returns:
            Extended Total System Throughput
        """
        internal_tst = self.calculate_tst()
        total_imports = np.sum(self.imports)
        total_exports = np.sum(self.exports)
        total_respiration = np.sum(self.respiration)
        
        return internal_tst + total_imports + total_exports + total_respiration
    
    def calculate_finn_cycling_index(self) -> float:
        """
        Calculate Finn's Cycling Index (FCI) — canonical Leontief method.

        FCI is the fraction of total system throughput that is cycled, i.e.
        that revisits at least one compartment. Higher values indicate more
        material/energy recycling.

        Canonical method (Finn 1976; Ulanowicz 2004 §5 p.330; Fath 2019
        Principle 2 p.20):
          1. Column-normalize by throughflow to form the transition matrix G:
                 G[:, j] = T[:, j] / T_j_in
             where T_j_in is the total inflow to compartment j (internal column
             sum + imports if boundary flows are provided; internal column sum
             only otherwise). Zero-inflow columns are guarded to zero.
          2. Leontief structure matrix  S = (I - G)^-1  (Simon-Hawkins limit).
          3. Cycled throughflow  TSTc = Σ_i ((S[i,i] - 1) / S[i,i]) · T_i,
             where T_i is the total throughflow of compartment i. Each diagonal
             element s_ii is the expected number of visits to i, so
             (s_ii - 1)/s_ii is the fraction of i's throughflow that is cycled.
          4. FCI = TSTc / TST, where TST = Σ_i T_i is the total system
             throughflow — the SAME per-compartment throughflow used to weight
             TSTc. Numerator and denominator therefore share one consistent
             basis (Finn 1976; Ulanowicz 2004 §5).

        NOTE (Track-1 correction): the previous implementation normalized by the
        scalar TST (making G tiny so S ≈ I and cycling was crushed) and summed
        the off-diagonal of S. Both are departures from the canonical method and
        systematically under-estimate cycling (≈ 0.3-0.6× true FCI); a pure ring
        returned ≈ 0 instead of ≈ 1.

        NOTE (basis reconciliation): TSTc is weighted by the total throughflow
        T_i = internal inflow + imports, so the denominator must be the total
        system throughflow Σ_i T_i, NOT the internal-only flow sum
        (calculate_tst). Using the internal-only sum in the denominator while
        weighting the numerator by total throughflow biases FCI upward for
        networks that have both large imports and real cycling.

        Returns:
            Finn's Cycling Index (0-1)
        """
        n = self.n_nodes

        # Total throughflow of each compartment (T_i): receiving-side inflow
        # including imports where boundary flows are present; internal-only
        # matrices fall back to the internal column sum (imports = 0).
        col_sum = self.input_throughput          # internal inflow to j
        t_in = col_sum + self.imports            # total inflow (throughflow) to j
        # Compartment throughflow used for weighting TSTc: total input to i.
        throughflow = t_in

        # Denominator uses the SAME basis: total system throughflow Σ_i T_i.
        tst = float(np.sum(throughflow))
        if tst == 0:
            return 0.0

        # Column-normalized transition matrix G[:, j] = T[:, j] / T_j_in
        G = np.zeros((n, n), dtype=np.float64)
        for j in range(n):
            if t_in[j] > 0:
                G[:, j] = self.flow_matrix[:, j] / t_in[j]

        identity = np.eye(n)

        # A perfectly conservative internal structure (no leak to the boundary)
        # makes (I - G) singular: it is the limit of full recycling, where every
        # quantum returns to its compartment infinitely often (FCI -> 1). We
        # evaluate S as the limit of the Leontief inverse under a vanishing leak
        # so that a pure ring yields FCI -> 1 (Ulanowicz 2004 §5: 0.993 at 1%
        # leak, -> 1.0 at closure) rather than a division by a singular matrix.
        try:
            S = np.linalg.inv(identity - G)
        except np.linalg.LinAlgError:
            # Regularized limit: shrink G slightly toward zero (tiny leak).
            eps = 1e-9
            try:
                S = np.linalg.inv(identity - (1.0 - eps) * G)
            except np.linalg.LinAlgError:
                return 0.0

        # If the inverse is finite but ill-conditioned (near-closed system), the
        # diagonal blows up and (s_ii - 1)/s_ii -> 1, which is exactly the
        # full-cycling limit; the arithmetic below handles it directly.
        diag = np.diag(S)
        with np.errstate(divide='ignore', invalid='ignore'):
            cycled_fraction = np.where(np.isfinite(diag) & (diag > 0),
                                       (diag - 1.0) / diag, 1.0)
        cycled_fraction = np.clip(cycled_fraction, 0.0, 1.0)

        tst_c = float(np.sum(cycled_fraction * throughflow))
        fci = tst_c / tst
        return max(0.0, min(1.0, fci))
    
    def calculate_balance_metrics(self) -> Dict[str, float]:
        """
        Calculate compartment balance metrics.
        
        For each compartment: Input = Output (steady state assumption)
        Input = Internal_in + Imports
        Output = Internal_out + Exports + Respiration
        
        Returns:
            Dictionary with balance metrics
        """
        balance_metrics = {}
        
        for i in range(self.n_nodes):
            total_input = self.input_throughput[i] + self.imports[i]
            total_output = self.output_throughput[i] + self.exports[i] + self.respiration[i]
            
            balance_metrics[f'{self.node_names[i]}_input'] = total_input
            balance_metrics[f'{self.node_names[i]}_output'] = total_output
            balance_metrics[f'{self.node_names[i]}_balance'] = total_input - total_output
        
        # System-wide metrics
        balance_metrics['total_imports'] = np.sum(self.imports)
        balance_metrics['total_exports'] = np.sum(self.exports)
        balance_metrics['total_respiration'] = np.sum(self.respiration)
        balance_metrics['net_production'] = np.sum(self.imports) - np.sum(self.exports) - np.sum(self.respiration)
        
        return balance_metrics
    
    def calculate_respiratory_retention_ratio(self) -> float:
        """
        Respiratory retention ratio (system-wide dissipation retention).

        Formula: 1 - Σ(respiration) / (TST + Σ(imports)).

        This is a single system-wide scalar: one minus the dissipated
        (respired) fraction of total activity. It is a legitimate, bounded
        [0, 1] respiratory-retention / dissipation ratio.

        NOTE (Track-1 correction): this quantity was previously mislabeled
        "Lindeman efficiency". True Lindeman (1942) trophic efficiency is a
        BETWEEN-LEVEL transfer efficiency (the "~10% rule") obtained from the
        Lindeman spine [L] (Ulanowicz 2004 §4, Fig. 5) — a per-level ratio of
        successive throughflows along a virtual straight chain. The metric here
        is neither between-level nor derived from [L], so it is renamed.

        TODO: implement true between-level transfer efficiency via the Lindeman
        spine [L] (Lindeman 1942; Ulanowicz 2004 §4) if per-level efficiencies
        are required.

        Returns:
            Respiratory retention ratio in [0, 1].
        """
        tst = self.calculate_tst()
        if tst == 0:
            return 0

        total_respiration = np.sum(self.respiration)
        retention = 1 - (total_respiration / (tst + np.sum(self.imports)))

        return max(0, min(1, retention))

    def calculate_lindeman_efficiency(self) -> float:
        """Deprecated alias for :meth:`calculate_respiratory_retention_ratio`.

        WARNING: despite the historical name, this is a system-wide respiratory
        retention ratio, NOT Lindeman between-level transfer efficiency. Kept as
        a back-compat alias so existing consumers do not break.
        """
        return self.calculate_respiratory_retention_ratio()
    
    def get_ecosystem_metrics(self) -> Dict[str, float]:
        """
        Get complete ecosystem flow metrics.
        
        Returns:
            Dictionary with all ecosystem-specific metrics
        """
        # Get base metrics
        base_metrics = self.get_extended_metrics()
        
        # Add ecosystem-specific metrics
        eco_metrics = {
            'tst_internal': self.calculate_tst(),
            'tst_extended': self.calculate_tst_extended(),
            'total_imports': np.sum(self.imports),
            'total_exports': np.sum(self.exports),
            'total_respiration': np.sum(self.respiration),
            'finn_cycling_index': self.calculate_finn_cycling_index(),
            'respiratory_retention_ratio': self.calculate_respiratory_retention_ratio(),
            # Back-compat alias (mislabeled historically; see method docstring).
            'lindeman_efficiency': self.calculate_respiratory_retention_ratio(),
            'import_dependency': np.sum(self.imports) / self.calculate_tst_extended() if self.calculate_tst_extended() > 0 else 0,
            'export_ratio': np.sum(self.exports) / self.calculate_tst_extended() if self.calculate_tst_extended() > 0 else 0,
            'respiration_ratio': np.sum(self.respiration) / self.calculate_tst_extended() if self.calculate_tst_extended() > 0 else 0,
        }
        
        # Combine all metrics
        return {**base_metrics, **eco_metrics}
    
    def assess_ecosystem_health(self) -> Dict[str, str]:
        """
        Assess ecosystem health based on flow patterns.
        
        Returns:
            Dictionary with health assessments
        """
        metrics = self.get_ecosystem_metrics()
        
        assessments = {}
        
        # Energy efficiency assessment
        if metrics['respiration_ratio'] > 0.7:
            assessments['energy_efficiency'] = "LOW - High dissipative losses"
        elif metrics['respiration_ratio'] < 0.3:
            assessments['energy_efficiency'] = "HIGH - Efficient energy use"
        else:
            assessments['energy_efficiency'] = "MODERATE - Balanced energy dissipation"
        
        # Cycling assessment
        if metrics['finn_cycling_index'] < 0.1:
            assessments['nutrient_cycling'] = "LOW - Little internal recycling"
        elif metrics['finn_cycling_index'] > 0.5:
            assessments['nutrient_cycling'] = "HIGH - Strong internal cycling"
        else:
            assessments['nutrient_cycling'] = "MODERATE - Some recycling present"
        
        # Import dependency
        if metrics['import_dependency'] > 0.5:
            assessments['autonomy'] = "LOW - High external dependency"
        elif metrics['import_dependency'] < 0.2:
            assessments['autonomy'] = "HIGH - Self-sufficient system"
        else:
            assessments['autonomy'] = "MODERATE - Balanced autonomy"
        
        # Overall ecosystem health
        if metrics['is_viable'] and metrics['finn_cycling_index'] > 0.2 and metrics['respiration_ratio'] < 0.6:
            assessments['overall_health'] = "HEALTHY - Well-functioning ecosystem"
        elif not metrics['is_viable']:
            assessments['overall_health'] = "STRESSED - Outside viability window"
        else:
            assessments['overall_health'] = "TRANSITIONAL - System adapting"
        
        return assessments


def create_from_ecosystem_data(data: Dict) -> EcosystemFlowCalculator:
    """
    Create calculator from ecosystem data dictionary.
    
    Args:
        data: Dictionary with 'flows', 'nodes', and optionally 'metadata' with
              'exogenous_inputs', 'exogenous_outputs', 'dissipations'
    
    Returns:
        Configured EcosystemFlowCalculator
    """
    flow_matrix = np.array(data['flows'])
    node_names = data['nodes']
    
    # Extract boundary flows from metadata if available
    metadata = data.get('metadata', {})
    
    # Imports (exogenous inputs)
    imports = np.zeros(len(node_names))
    if 'exogenous_inputs' in metadata:
        for key, value in metadata['exogenous_inputs'].items():
            # Extract node index from key (e.g., 'to_plants' -> 'plants')
            node_name = key.replace('to_', '')
            if node_name in [n.lower() for n in node_names]:
                idx = [n.lower() for n in node_names].index(node_name)
                imports[idx] = value
    
    # Exports (exogenous outputs)
    exports = np.zeros(len(node_names))
    if 'exogenous_outputs' in metadata:
        for key, value in metadata['exogenous_outputs'].items():
            node_name = key.replace('from_', '')
            if node_name in [n.lower() for n in node_names]:
                idx = [n.lower() for n in node_names].index(node_name)
                exports[idx] = value
    
    # Respiration (dissipations)
    respiration = np.zeros(len(node_names))
    if 'dissipations' in metadata:
        for key, value in metadata['dissipations'].items():
            if key in [n.lower() for n in node_names]:
                idx = [n.lower() for n in node_names].index(key)
                respiration[idx] = value
    
    return EcosystemFlowCalculator(
        flow_matrix=flow_matrix,
        node_names=node_names,
        imports=imports,
        exports=exports,
        respiration=respiration
    )


if __name__ == "__main__":
    # Test with Cone Spring ecosystem data
    import json
    
    with open('data/ecosystem_samples/cone_spring_original.json', 'r') as f:
        cone_spring = json.load(f)
    
    # Create calculator with full ecosystem flows
    calc = create_from_ecosystem_data(cone_spring)
    
    print("Cone Spring Ecosystem Analysis")
    print("=" * 50)
    
    # Get metrics
    metrics = calc.get_ecosystem_metrics()
    
    print(f"\nFlow Components:")
    print(f"  Internal TST: {metrics['tst_internal']:.1f}")
    print(f"  Total Imports: {metrics['total_imports']:.1f}")
    print(f"  Total Exports: {metrics['total_exports']:.1f}")
    print(f"  Total Respiration: {metrics['total_respiration']:.1f}")
    print(f"  Extended TST: {metrics['tst_extended']:.1f}")
    
    print(f"\nSustainability Metrics:")
    print(f"  Relative Ascendency: {metrics['relative_ascendency']:.3f}")
    print(f"  Robustness: {metrics['robustness']:.3f}")
    print(f"  Finn Cycling Index: {metrics['finn_cycling_index']:.3f}")
    print(f"  Respiratory Retention Ratio: {metrics['respiratory_retention_ratio']:.3f}")
    
    print(f"\nEcosystem Health:")
    health = calc.assess_ecosystem_health()
    for aspect, assessment in health.items():
        print(f"  {aspect}: {assessment}")