"""
Energy Flow Dataset Processor

Specialized processor for energy flow datasets including power grids and energy networks.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .base_processor import BaseDatasetProcessor


class EnergyFlowProcessor(BaseDatasetProcessor):
    """Processor for energy flow datasets."""
    
    def __init__(self, dataset_name: str, source_info: Dict[str, Any]):
        super().__init__(dataset_name, source_info)
        
    def download_dataset(self) -> Optional[Path]:
        """
        Download energy dataset. 
        Note: For now, this creates sample data until Kaggle API is integrated.
        """
        # TODO: Implement Kaggle API download
        self.logger.info("Creating sample energy network data for demonstration")
        return None
        
    def explore_structure(self, data_path: Path) -> Dict[str, Any]:
        """Explore energy dataset structure."""
        if data_path and data_path.exists():
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
                return {
                    "file_type": "CSV",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample_data": df.head(3).to_dict()
                }
        return {"status": "using_sample_data"}
        
    def extract_nodes(self, data: Any) -> List[str]:
        """Extract energy network nodes."""
        if isinstance(data, pd.DataFrame):
            # Look for common energy network node columns
            node_columns = ['node', 'station', 'substation', 'generator', 'bus', 'from_node', 'to_node']
            
            nodes = set()
            for col in node_columns:
                if col in data.columns:
                    nodes.update(data[col].dropna().unique())
                if f'{col}_id' in data.columns:
                    nodes.update(data[f'{col}_id'].dropna().unique())
                    
            if nodes:
                return sorted(list(nodes))
                
        # Create sample European power grid nodes
        return self._create_sample_energy_nodes()
        
    def extract_flows(self, data: Any, nodes: List[str]) -> np.ndarray:
        """Extract energy flows between nodes."""
        n = len(nodes)
        flows = np.zeros((n, n))
        
        if isinstance(data, pd.DataFrame) and 'from_node' in data.columns and 'to_node' in data.columns:
            # Process real power flow data
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            flow_col = None
            for col in ['power_mw', 'flow', 'capacity', 'transmission']:
                if col in data.columns:
                    flow_col = col
                    break
                    
            if flow_col:
                for _, row in data.iterrows():
                    from_node = row.get('from_node')
                    to_node = row.get('to_node') 
                    flow_value = row.get(flow_col, 0)
                    
                    if from_node in node_to_idx and to_node in node_to_idx:
                        i, j = node_to_idx[from_node], node_to_idx[to_node]
                        flows[i, j] += flow_value
        else:
            # Create sample European power grid flows
            flows = self._create_sample_energy_flows(nodes)
            
        return flows
        
    def _create_sample_energy_nodes(self) -> List[str]:
        """Create sample European energy network nodes."""
        return [
            "Nuclear_Plant_France",
            "Wind_Farm_Germany", 
            "Solar_Farm_Spain",
            "Hydro_Plant_Norway",
            "Coal_Plant_Poland",
            "Gas_Plant_Netherlands",
            "Substation_Belgium",
            "Substation_Austria",
            "Distribution_Italy",
            "Distribution_Denmark"
        ]
        
    def _create_sample_energy_flows(self, nodes: List[str]) -> np.ndarray:
        """Create realistic sample energy flows."""
        n = len(nodes)
        flows = np.zeros((n, n))
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create flows based on realistic energy patterns
        generation_nodes = ["Nuclear_Plant_France", "Wind_Farm_Germany", "Solar_Farm_Spain", 
                          "Hydro_Plant_Norway", "Coal_Plant_Poland", "Gas_Plant_Netherlands"]
        
        transmission_nodes = ["Substation_Belgium", "Substation_Austria"]
        distribution_nodes = ["Distribution_Italy", "Distribution_Denmark"]
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Generation -> Transmission flows
        for gen_node in generation_nodes:
            if gen_node in node_to_idx:
                gen_idx = node_to_idx[gen_node]
                for trans_node in transmission_nodes:
                    if trans_node in node_to_idx:
                        trans_idx = node_to_idx[trans_node]
                        # Typical power plant output: 100-800 MW
                        flows[gen_idx, trans_idx] = np.random.uniform(150, 600)
                        
        # Transmission -> Distribution flows
        for trans_node in transmission_nodes:
            if trans_node in node_to_idx:
                trans_idx = node_to_idx[trans_node]
                for dist_node in distribution_nodes:
                    if dist_node in node_to_idx:
                        dist_idx = node_to_idx[dist_node]
                        # Transmission capacity: 200-400 MW
                        flows[trans_idx, dist_idx] = np.random.uniform(200, 400)
                        
        # Some direct generation -> distribution flows
        for gen_node in generation_nodes[:3]:  # Some generators feed directly
            if gen_node in node_to_idx:
                gen_idx = node_to_idx[gen_node]
                for dist_node in distribution_nodes:
                    if dist_node in node_to_idx:
                        dist_idx = node_to_idx[dist_node]
                        flows[gen_idx, dist_idx] = np.random.uniform(50, 200)
                        
        # Cross-border transmission flows
        cross_border_pairs = [
            ("Nuclear_Plant_France", "Gas_Plant_Netherlands"),
            ("Wind_Farm_Germany", "Coal_Plant_Poland"),
            ("Hydro_Plant_Norway", "Wind_Farm_Germany")
        ]
        
        for from_node, to_node in cross_border_pairs:
            if from_node in node_to_idx and to_node in node_to_idx:
                from_idx, to_idx = node_to_idx[from_node], node_to_idx[to_node]
                flows[from_idx, to_idx] = np.random.uniform(100, 300)
                
        return flows
        
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample energy data structure."""
        nodes = self._create_sample_energy_nodes()
        
        # Create sample data structure that mimics real energy datasets
        data_rows = []
        for i, from_node in enumerate(nodes):
            for j, to_node in enumerate(nodes):
                if i != j and np.random.random() > 0.6:  # ~40% connectivity
                    data_rows.append({
                        'from_node': from_node,
                        'to_node': to_node,
                        'power_mw': np.random.uniform(50, 500),
                        'capacity_mw': np.random.uniform(100, 800),
                        'voltage_kv': np.random.choice([110, 220, 400]),
                        'line_type': np.random.choice(['transmission', 'distribution'])
                    })
                    
        return pd.DataFrame(data_rows)
        
    def _get_flow_units(self) -> str:
        """Get energy flow units."""
        return "MW (Megawatts)"
        
    def _get_processing_notes(self) -> str:
        """Get energy-specific processing notes."""
        return ("Processed European power grid network with generation, transmission, "
                "and distribution nodes. Flows represent electrical power transmission "
                "between network components.")


class EuropeanPowerGridProcessor(EnergyFlowProcessor):
    """Specific processor for European Power Grid Network dataset."""
    
    def __init__(self):
        source_info = {
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/pythonafroz/european-power-grid-network-dataset",
            "description": "European power grid network data with energy flow information for network analysis",
            "type": "Energy Flow",
            "license": "Dataset-specific license"
        }
        super().__init__("European Power Grid Network", source_info)
        
        
class SmartGridProcessor(EnergyFlowProcessor):
    """Specific processor for Smart Grid monitoring datasets."""
    
    def __init__(self):
        source_info = {
            "source": "Kaggle", 
            "url": "https://www.kaggle.com/datasets/ziya07/smart-grid-real-time-load-monitoring-dataset",
            "description": "Real-time smart grid load monitoring data for energy management analysis",
            "type": "Energy Flow",
            "license": "Dataset-specific license"
        }
        super().__init__("Smart Grid Real-Time Monitoring", source_info)