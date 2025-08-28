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
        Calculate Finn's Cycling Index (FCI).
        
        FCI measures the fraction of total throughput involved in cycling.
        Higher values indicate more material/energy recycling.
        
        Based on: Finn, J.T. (1976) "Measures of ecosystem structure and function 
        derived from analysis of flows" J. Theor. Biol. 56:363-380
        
        Returns:
            Finn's Cycling Index (0-1)
        """
        # Create augmented matrix including boundary flows
        n = self.n_nodes
        augmented = np.zeros((n+2, n+2))
        
        # Internal flows
        augmented[:n, :n] = self.flow_matrix
        
        # Imports (from environment node n to compartments)
        augmented[n, :n] = self.imports
        
        # Exports and respiration (from compartments to sink node n+1)
        augmented[:n, n+1] = self.exports + self.respiration
        
        # Calculate cycling using matrix powers
        try:
            # Normalize by total throughput
            tst = self.calculate_tst_extended()
            if tst == 0:
                return 0
            
            # Calculate first-order cycling
            flow_norm = self.flow_matrix / tst
            identity = np.eye(n)
            
            # Leontief inverse for cycling calculation
            leontief = np.linalg.inv(identity - flow_norm)
            cycling = np.sum(leontief) - n  # Subtract diagonal
            
            fci = cycling / np.sum(leontief)
            return max(0, min(1, fci))  # Bound between 0 and 1
            
        except np.linalg.LinAlgError:
            return 0
    
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
    
    def calculate_lindeman_efficiency(self) -> float:
        """
        Calculate Lindeman trophic efficiency.
        
        Efficiency of energy transfer between trophic levels.
        Based on: Lindeman, R.L. (1942) "The trophic-dynamic aspect of ecology"
        
        Returns:
            Average trophic efficiency
        """
        # Simplified calculation based on respiration losses
        tst = self.calculate_tst()
        if tst == 0:
            return 0
        
        total_respiration = np.sum(self.respiration)
        efficiency = 1 - (total_respiration / (tst + np.sum(self.imports)))
        
        return max(0, min(1, efficiency))
    
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
            'lindeman_efficiency': self.calculate_lindeman_efficiency(),
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
    print(f"  Lindeman Efficiency: {metrics['lindeman_efficiency']:.3f}")
    
    print(f"\nEcosystem Health:")
    health = calc.assess_ecosystem_health()
    for aspect, assessment in health.items():
        print(f"  {aspect}: {assessment}")