"""
Supply Chain Flow Dataset Processor

Specialized processor for supply chain datasets including logistics and distribution networks.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .base_processor import BaseDatasetProcessor


class SupplyChainProcessor(BaseDatasetProcessor):
    """Processor for supply chain flow datasets."""
    
    def __init__(self, dataset_name: str, source_info: Dict[str, Any]):
        super().__init__(dataset_name, source_info)
        
    def download_dataset(self) -> Optional[Path]:
        """
        Download supply chain dataset. 
        Note: For now, this creates sample data until Kaggle API is integrated.
        """
        # TODO: Implement Kaggle API download
        self.logger.info("Creating sample supply chain data for demonstration")
        return None
        
    def explore_structure(self, data_path: Path) -> Dict[str, Any]:
        """Explore supply chain dataset structure."""
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
        """Extract supply chain network nodes."""
        if isinstance(data, pd.DataFrame):
            # Look for common supply chain node columns
            node_columns = ['supplier', 'warehouse', 'customer', 'facility', 'origin', 'destination']
            
            nodes = set()
            for col in node_columns:
                if col in data.columns:
                    nodes.update(data[col].dropna().unique())
                if f'{col}_id' in data.columns:
                    nodes.update(data[f'{col}_id'].dropna().unique())
                if f'{col}_name' in data.columns:
                    nodes.update(data[f'{col}_name'].dropna().unique())
                    
            if nodes:
                return sorted(list(nodes))
                
        # Create sample supply chain nodes
        return self._create_sample_supply_chain_nodes()
        
    def extract_flows(self, data: Any, nodes: List[str]) -> np.ndarray:
        """Extract supply chain flows between nodes."""
        n = len(nodes)
        flows = np.zeros((n, n))
        
        if isinstance(data, pd.DataFrame):
            # Process real supply chain data
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Look for flow-related columns
            flow_cols = ['quantity', 'volume', 'units', 'orders', 'shipments']
            flow_col = None
            for col in flow_cols:
                if col in data.columns:
                    flow_col = col
                    break
                    
            # Look for origin/destination columns
            origin_col = None
            dest_col = None
            for col in ['origin', 'supplier', 'from']:
                if col in data.columns or f'{col}_id' in data.columns:
                    origin_col = col if col in data.columns else f'{col}_id'
                    break
                    
            for col in ['destination', 'customer', 'to', 'warehouse']:
                if col in data.columns or f'{col}_id' in data.columns:
                    dest_col = col if col in data.columns else f'{col}_id'
                    break
                    
            if flow_col and origin_col and dest_col:
                for _, row in data.iterrows():
                    origin = row.get(origin_col)
                    destination = row.get(dest_col)
                    flow_value = row.get(flow_col, 0)
                    
                    if origin in node_to_idx and destination in node_to_idx:
                        i, j = node_to_idx[origin], node_to_idx[destination]
                        flows[i, j] += flow_value
        else:
            # Create sample supply chain flows
            flows = self._create_sample_supply_chain_flows(nodes)
            
        return flows
        
    def _create_sample_supply_chain_nodes(self) -> List[str]:
        """Create sample supply chain network nodes."""
        return [
            "Raw_Materials_Supplier_Asia",
            "Raw_Materials_Supplier_Europe", 
            "Component_Manufacturer_China",
            "Component_Manufacturer_Germany",
            "Assembly_Plant_Mexico",
            "Assembly_Plant_USA",
            "Distribution_Center_East",
            "Distribution_Center_West",
            "Regional_Warehouse_North",
            "Regional_Warehouse_South",
            "Retail_Stores_Urban",
            "Retail_Stores_Suburban"
        ]
        
    def _create_sample_supply_chain_flows(self, nodes: List[str]) -> np.ndarray:
        """Create realistic sample supply chain flows."""
        n = len(nodes)
        flows = np.zeros((n, n))
        
        # Set random seed for reproducibility
        np.random.seed(123)
        
        # Create flows based on realistic supply chain patterns
        suppliers = ["Raw_Materials_Supplier_Asia", "Raw_Materials_Supplier_Europe"]
        manufacturers = ["Component_Manufacturer_China", "Component_Manufacturer_Germany"]
        assembly_plants = ["Assembly_Plant_Mexico", "Assembly_Plant_USA"]
        distribution_centers = ["Distribution_Center_East", "Distribution_Center_West"]
        warehouses = ["Regional_Warehouse_North", "Regional_Warehouse_South"]
        retail = ["Retail_Stores_Urban", "Retail_Stores_Suburban"]
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Suppliers → Manufacturers flows
        for supplier in suppliers:
            if supplier in node_to_idx:
                supplier_idx = node_to_idx[supplier]
                for manufacturer in manufacturers:
                    if manufacturer in node_to_idx:
                        mfg_idx = node_to_idx[manufacturer]
                        # Raw material flows: 1000-5000 units
                        flows[supplier_idx, mfg_idx] = np.random.uniform(1000, 5000)
                        
        # Manufacturers → Assembly Plants flows
        for manufacturer in manufacturers:
            if manufacturer in node_to_idx:
                mfg_idx = node_to_idx[manufacturer]
                for plant in assembly_plants:
                    if plant in node_to_idx:
                        plant_idx = node_to_idx[plant]
                        # Component flows: 800-3000 units
                        flows[mfg_idx, plant_idx] = np.random.uniform(800, 3000)
                        
        # Assembly Plants → Distribution Centers flows
        for plant in assembly_plants:
            if plant in node_to_idx:
                plant_idx = node_to_idx[plant]
                for dc in distribution_centers:
                    if dc in node_to_idx:
                        dc_idx = node_to_idx[dc]
                        # Finished product flows: 500-2000 units
                        flows[plant_idx, dc_idx] = np.random.uniform(500, 2000)
                        
        # Distribution Centers → Warehouses flows
        for dc in distribution_centers:
            if dc in node_to_idx:
                dc_idx = node_to_idx[dc]
                for warehouse in warehouses:
                    if warehouse in node_to_idx:
                        wh_idx = node_to_idx[warehouse]
                        # Distribution flows: 300-1500 units
                        flows[dc_idx, wh_idx] = np.random.uniform(300, 1500)
                        
        # Warehouses → Retail flows
        for warehouse in warehouses:
            if warehouse in node_to_idx:
                wh_idx = node_to_idx[warehouse]
                for store in retail:
                    if store in node_to_idx:
                        store_idx = node_to_idx[store]
                        # Retail flows: 200-800 units
                        flows[wh_idx, store_idx] = np.random.uniform(200, 800)
                        
        # Some direct flows (bypass certain levels)
        direct_flows = [
            ("Raw_Materials_Supplier_Asia", "Assembly_Plant_Mexico"),
            ("Component_Manufacturer_China", "Distribution_Center_West"),
            ("Distribution_Center_East", "Retail_Stores_Urban")
        ]
        
        for from_node, to_node in direct_flows:
            if from_node in node_to_idx and to_node in node_to_idx:
                from_idx, to_idx = node_to_idx[from_node], node_to_idx[to_node]
                flows[from_idx, to_idx] = np.random.uniform(100, 500)
                
        return flows
        
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample supply chain data structure."""
        nodes = self._create_sample_supply_chain_nodes()
        
        # Create sample data structure that mimics real supply chain datasets
        data_rows = []
        for i, origin in enumerate(nodes):
            for j, destination in enumerate(nodes):
                if i != j and np.random.random() > 0.7:  # ~30% connectivity
                    data_rows.append({
                        'supplier': origin,
                        'customer': destination,
                        'quantity': np.random.uniform(100, 2000),
                        'orders': np.random.randint(10, 200),
                        'lead_time': np.random.uniform(1, 30),
                        'cost': np.random.uniform(1000, 50000)
                    })
                    
        return pd.DataFrame(data_rows)
        
    def _get_flow_units(self) -> str:
        """Get supply chain flow units."""
        return "Units (Products/Components)"
        
    def _get_processing_notes(self) -> str:
        """Get supply chain-specific processing notes."""
        return ("Processed multi-tier supply chain network with suppliers, manufacturers, "
                "assembly plants, distribution centers, warehouses, and retail outlets. "
                "Flows represent product/component movement through the supply network.")


class DataCoSupplyChainProcessor(SupplyChainProcessor):
    """Specific processor for DataCo Smart Supply Chain dataset."""
    
    def __init__(self):
        source_info = {
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
            "description": "Comprehensive supply chain network with delivery performance, customer segments, and flow pathways",
            "type": "Supply Chain Flow",
            "license": "Dataset-specific license"
        }
        super().__init__("DataCo Smart Supply Chain", source_info)


class LogisticsSupplyChainProcessor(SupplyChainProcessor):
    """Specific processor for Logistics and Supply Chain dataset."""
    
    def __init__(self):
        source_info = {
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset", 
            "description": "Recent logistics and supply chain dataset focusing on modern distribution networks",
            "type": "Supply Chain Flow",
            "license": "Dataset-specific license"
        }
        super().__init__("Logistics and Supply Chain Network", source_info)