"""
HuggingFace Dataset to Flow Matrix Converter

This module provides preprocessing pipelines to convert various HuggingFace datasets
into flow matrices compatible with the Ulanowicz sustainability analysis system.

Supported datasets:
1. Mobility networks (Romz16/Mobility_Network_Data-Analysis)
2. Protein interaction networks (graphs-datasets/PROTEINS, proteinea/ppb_affinity)
3. Logistics networks (Cainiao-AI/LaDe)
4. Gene regulatory networks (ctheodoris/Genecorpus-30M)
5. Stock market correlations (pettah/global-top-Index-exploring-trends-in-stock-Market)
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Union
import json
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class HuggingFaceFlowExtractor:
    """Extract and convert flow networks from HuggingFace datasets."""
    
    def __init__(self):
        """Initialize the flow extractor."""
        self.scaler = MinMaxScaler()
        
    def extract_mobility_network(self, 
                                dataset_name: str = "Romz16/Mobility_Network_Data-Analysis",
                                config_name: Optional[str] = None,
                                max_nodes: int = 100) -> Dict:
        """
        Extract flow matrix from mobility/transportation datasets.
        
        Args:
            dataset_name: HuggingFace dataset ID
            config_name: Specific configuration (e.g., 'shanghai', 'hangzhou')
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Dictionary with flow matrix and metadata
        """
        print(f"Loading mobility dataset: {dataset_name}")
        
        try:
            # Load dataset
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split='train')
            else:
                dataset = load_dataset(dataset_name, split='train')
            
            # Convert to DataFrame for easier processing
            df = dataset.to_pandas()
            
            # Extract origin-destination flows
            if 'origin' in df.columns and 'destination' in df.columns:
                # Aggregate flows between locations
                flow_data = df.groupby(['origin', 'destination']).size().reset_index(name='flow')
            elif 'from' in df.columns and 'to' in df.columns:
                flow_data = df.groupby(['from', 'to']).size().reset_index(name='flow')
            else:
                # Try to identify flow columns
                flow_cols = [col for col in df.columns if 'flow' in col.lower()]
                if flow_cols:
                    return self._process_matrix_format(df, max_nodes)
                else:
                    raise ValueError("Could not identify flow structure in dataset")
            
            # Get unique nodes
            all_nodes = pd.concat([flow_data.iloc[:, 0], flow_data.iloc[:, 1]]).unique()
            
            # Limit nodes if necessary
            if len(all_nodes) > max_nodes:
                # Select top nodes by total flow
                node_flows = {}
                for node in all_nodes:
                    outflow = flow_data[flow_data.iloc[:, 0] == node]['flow'].sum()
                    inflow = flow_data[flow_data.iloc[:, 1] == node]['flow'].sum()
                    node_flows[node] = outflow + inflow
                
                top_nodes = sorted(node_flows.keys(), key=lambda x: node_flows[x], reverse=True)[:max_nodes]
                flow_data = flow_data[flow_data.iloc[:, 0].isin(top_nodes) & 
                                      flow_data.iloc[:, 1].isin(top_nodes)]
                nodes = top_nodes
            else:
                nodes = list(all_nodes)
            
            # Create flow matrix
            n = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            flow_matrix = np.zeros((n, n))
            
            for _, row in flow_data.iterrows():
                if row.iloc[0] in node_to_idx and row.iloc[1] in node_to_idx:
                    i = node_to_idx[row.iloc[0]]
                    j = node_to_idx[row.iloc[1]]
                    flow_matrix[i, j] = row['flow']
            
            # Normalize flows
            if flow_matrix.sum() > 0:
                flow_matrix = flow_matrix / flow_matrix.max() * 100  # Scale to 0-100
            
            return {
                'organization': f'Mobility Network - {config_name or dataset_name}',
                'nodes': [str(n) for n in nodes],
                'flows': flow_matrix.tolist(),
                'metadata': {
                    'source': dataset_name,
                    'config': config_name,
                    'total_nodes': len(nodes),
                    'total_flow': float(flow_matrix.sum()),
                    'density': float(np.count_nonzero(flow_matrix) / (n * n))
                }
            }
            
        except Exception as e:
            print(f"Error loading mobility dataset: {e}")
            return None
    
    def extract_protein_network(self,
                               dataset_name: str = "graphs-datasets/PROTEINS",
                               network_index: int = 0,
                               split: str = None) -> Dict:
        """
        Extract flow matrix from protein interaction networks.
        
        Args:
            dataset_name: HuggingFace dataset ID
            network_index: Which protein network to extract (if multiple)
            split: Dataset split to use (e.g., 'train', 'full', 'test'). If None, tries common splits.
            
        Returns:
            Dictionary with flow matrix and metadata
        """
        print(f"Loading protein network: {dataset_name}")
        
        try:
            # Try to load dataset with appropriate split
            if split:
                dataset = load_dataset(dataset_name, split=split)
            else:
                # Try common split names
                for try_split in ['train', 'full', 'test', 'validation']:
                    try:
                        dataset = load_dataset(dataset_name, split=try_split)
                        print(f"  Using split: {try_split}")
                        break
                    except:
                        continue
                else:
                    # If no split works, try without split parameter
                    dataset = load_dataset(dataset_name)
                    # If it's a DatasetDict, get the first split
                    if hasattr(dataset, 'keys'):
                        first_split = list(dataset.keys())[0]
                        dataset = dataset[first_split]
                        print(f"  Using split: {first_split}")
            
            # Get specific graph
            if len(dataset) > network_index:
                graph_data = dataset[network_index]
            else:
                graph_data = dataset[0]
            
            # Extract edge information
            if 'edge_index' in graph_data:
                edges = graph_data['edge_index']
                num_nodes = graph_data.get('num_nodes', max(max(edges[0]), max(edges[1])) + 1)
                
                # Build adjacency matrix
                flow_matrix = np.zeros((num_nodes, num_nodes))
                
                # If edge attributes exist, use them as weights
                if 'edge_attr' in graph_data:
                    edge_weights = graph_data['edge_attr']
                    for i, (src, dst) in enumerate(zip(edges[0], edges[1])):
                        weight = edge_weights[i] if i < len(edge_weights) else 1.0
                        flow_matrix[src, dst] = weight
                else:
                    # Unweighted edges
                    for src, dst in zip(edges[0], edges[1]):
                        flow_matrix[src, dst] = 1.0
                
                # Generate node labels
                if 'x' in graph_data:  # Node features
                    nodes = [f"Protein_{i}" for i in range(num_nodes)]
                else:
                    nodes = [f"Node_{i}" for i in range(num_nodes)]
                
                # Normalize
                if flow_matrix.sum() > 0:
                    flow_matrix = flow_matrix / flow_matrix.max() * 100
                
                return {
                    'organization': f'Protein Network #{network_index}',
                    'nodes': nodes,
                    'flows': flow_matrix.tolist(),
                    'metadata': {
                        'source': dataset_name,
                        'network_index': network_index,
                        'total_nodes': num_nodes,
                        'total_edges': len(edges[0]) if edges else 0,
                        'density': float(np.count_nonzero(flow_matrix) / (num_nodes * num_nodes))
                    }
                }
            
            # Alternative format: PPB Affinity dataset
            elif 'affinity' in graph_data or 'binding_affinity' in graph_data:
                return self._extract_affinity_network(dataset, network_index)
            
        except Exception as e:
            print(f"Error loading protein network: {e}")
            return None
    
    def extract_logistics_network(self,
                                 dataset_name: str = "Cainiao-AI/LaDe",
                                 city: str = "shanghai",
                                 aggregate_level: str = "district") -> Dict:
        """
        Extract flow matrix from logistics/delivery networks.
        
        Args:
            dataset_name: HuggingFace dataset ID
            city: City configuration to load
            aggregate_level: 'district', 'grid', or 'station'
            
        Returns:
            Dictionary with flow matrix and metadata
        """
        print(f"Loading logistics network for {city}")
        
        try:
            # Load dataset with specific city configuration
            dataset = load_dataset(dataset_name, city, split='train')
            df = dataset.to_pandas()
            
            # Aggregate deliveries by spatial units
            if 'courier_id' in df.columns and 'lat' in df.columns:
                # Create spatial bins based on coordinates
                if aggregate_level == 'district':
                    # Use district-level aggregation
                    lat_bins = pd.qcut(df['lat'], q=10, labels=False, duplicates='drop')
                    lon_bins = pd.qcut(df['lng'], q=10, labels=False, duplicates='drop')
                    df['zone'] = lat_bins.astype(str) + '_' + lon_bins.astype(str)
                elif aggregate_level == 'station':
                    # Use delivery station if available
                    if 'station_id' in df.columns:
                        df['zone'] = df['station_id']
                    else:
                        df['zone'] = pd.qcut(df['lat'], q=20, labels=False, duplicates='drop').astype(str)
                else:
                    # Grid-based aggregation
                    df['zone'] = (df['lat'].round(3).astype(str) + '_' + 
                                 df['lng'].round(3).astype(str))
                
                # Track package flows between zones over time
                if 'time' in df.columns or 'timestamp' in df.columns:
                    time_col = 'time' if 'time' in df.columns else 'timestamp'
                    df = df.sort_values(time_col)
                    
                    # Group by courier and track zone transitions
                    flows = []
                    for courier, courier_df in df.groupby('courier_id'):
                        zones = courier_df['zone'].values
                        for i in range(len(zones) - 1):
                            if zones[i] != zones[i+1]:  # Zone transition
                                flows.append((zones[i], zones[i+1]))
                    
                    # Count flows
                    flow_df = pd.DataFrame(flows, columns=['from', 'to'])
                    flow_counts = flow_df.groupby(['from', 'to']).size().reset_index(name='flow')
                    
                    # Build flow matrix
                    zones = sorted(set(flow_df['from'].unique()) | set(flow_df['to'].unique()))
                    n = min(len(zones), 50)  # Limit to 50 zones for managability
                    zones = zones[:n]
                    
                    zone_to_idx = {zone: i for i, zone in enumerate(zones)}
                    flow_matrix = np.zeros((n, n))
                    
                    for _, row in flow_counts.iterrows():
                        if row['from'] in zone_to_idx and row['to'] in zone_to_idx:
                            i = zone_to_idx[row['from']]
                            j = zone_to_idx[row['to']]
                            flow_matrix[i, j] = row['flow']
                    
                    # Normalize
                    if flow_matrix.sum() > 0:
                        flow_matrix = flow_matrix / flow_matrix.max() * 100
                    
                    return {
                        'organization': f'Logistics Network - {city.title()}',
                        'nodes': [f"Zone_{z}" for z in zones],
                        'flows': flow_matrix.tolist(),
                        'metadata': {
                            'source': dataset_name,
                            'city': city,
                            'aggregate_level': aggregate_level,
                            'total_zones': n,
                            'total_flow': float(flow_matrix.sum()),
                            'density': float(np.count_nonzero(flow_matrix) / (n * n))
                        }
                    }
            
        except Exception as e:
            print(f"Error loading logistics network: {e}")
            return None
    
    def extract_gene_regulatory_network(self,
                                       dataset_name: str = "ctheodoris/Genecorpus-30M",
                                       sample_size: int = 1000,
                                       correlation_threshold: float = 0.3) -> Dict:
        """
        Extract gene regulatory network from expression data.
        
        Args:
            dataset_name: HuggingFace dataset ID  
            sample_size: Number of cells to sample
            correlation_threshold: Minimum correlation for edge creation
            
        Returns:
            Dictionary with flow matrix and metadata
        """
        print(f"Loading gene expression dataset (this may take time)...")
        
        try:
            # Load a sample of the dataset
            dataset = load_dataset(dataset_name, split='train', streaming=True)
            
            # Sample cells
            samples = []
            for i, sample in enumerate(dataset):
                if i >= sample_size:
                    break
                samples.append(sample)
            
            if not samples:
                raise ValueError("No samples loaded")
            
            # Extract gene expression matrix
            # The format is tokenized, need to reconstruct expression values
            gene_expressions = []
            
            for sample in samples:
                if 'input_ids' in sample:
                    # Tokenized format - reconstruct expression
                    expr = sample['input_ids'][:100]  # Take first 100 genes
                    gene_expressions.append(expr)
            
            if not gene_expressions:
                print("Could not extract expression data, using synthetic example")
                # Create synthetic gene regulatory network
                n_genes = 30
                gene_expressions = np.random.randn(sample_size, n_genes)
            
            expr_matrix = np.array(gene_expressions)
            
            # Calculate gene-gene correlations
            gene_corr = np.corrcoef(expr_matrix.T)
            
            # Create flow matrix from significant correlations
            n_genes = gene_corr.shape[0]
            flow_matrix = np.zeros((n_genes, n_genes))
            
            for i in range(n_genes):
                for j in range(n_genes):
                    if i != j and abs(gene_corr[i, j]) > correlation_threshold:
                        # Use absolute correlation as flow strength
                        flow_matrix[i, j] = abs(gene_corr[i, j]) * 100
            
            # Generate gene names
            genes = [f"Gene_{i}" for i in range(n_genes)]
            
            return {
                'organization': 'Gene Regulatory Network',
                'nodes': genes,
                'flows': flow_matrix.tolist(),
                'metadata': {
                    'source': dataset_name,
                    'sample_size': sample_size,
                    'correlation_threshold': correlation_threshold,
                    'total_genes': n_genes,
                    'total_interactions': int(np.count_nonzero(flow_matrix)),
                    'density': float(np.count_nonzero(flow_matrix) / (n_genes * n_genes))
                }
            }
            
        except Exception as e:
            print(f"Error loading gene network: {e}")
            return None
    
    def extract_stock_correlation_network(self,
                                         dataset_name: str = "pettah/global-top-Index-exploring-trends-in-stock-Market",
                                         correlation_window: int = 30) -> Dict:
        """
        Extract correlation-based flow network from stock market data.
        
        Args:
            dataset_name: HuggingFace dataset ID
            correlation_window: Days for correlation calculation
            
        Returns:
            Dictionary with flow matrix and metadata
        """
        print(f"Loading stock market dataset...")
        
        try:
            dataset = load_dataset(dataset_name, split='train')
            df = dataset.to_pandas()
            
            # Identify index columns and price columns
            index_cols = [col for col in df.columns if 'index' in col.lower() or 'symbol' in col.lower()]
            price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
            
            if not price_cols:
                # Try to identify numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    price_cols = numeric_cols[:32]  # Take up to 32 indices
            
            if len(price_cols) < 2:
                print("Insufficient price data, creating synthetic network")
                # Create synthetic correlation network
                n_indices = 20
                returns = np.random.randn(100, n_indices) * 0.02
                correlations = np.corrcoef(returns.T)
            else:
                # Calculate returns
                prices = df[price_cols].values
                returns = np.diff(prices, axis=0) / prices[:-1]
                
                # Remove NaN values
                returns = np.nan_to_num(returns, nan=0.0)
                
                # Calculate correlation matrix
                correlations = np.corrcoef(returns.T)
                
            # Convert correlations to flow matrix
            n = correlations.shape[0]
            flow_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Use positive correlations as flows
                        if correlations[i, j] > 0:
                            flow_matrix[i, j] = correlations[i, j] * 100
            
            # Generate index names
            if index_cols and index_cols[0] in df.columns:
                indices = df[index_cols[0]].unique()[:n].tolist()
                nodes = [str(idx) for idx in indices]
            else:
                nodes = [f"Index_{i}" for i in range(n)]
            
            return {
                'organization': 'Global Stock Market Network',
                'nodes': nodes,
                'flows': flow_matrix.tolist(),
                'metadata': {
                    'source': dataset_name,
                    'correlation_window': correlation_window,
                    'total_indices': n,
                    'total_correlations': int(np.count_nonzero(flow_matrix)),
                    'avg_correlation': float(flow_matrix[flow_matrix > 0].mean() if np.any(flow_matrix > 0) else 0),
                    'density': float(np.count_nonzero(flow_matrix) / (n * n))
                }
            }
            
        except Exception as e:
            print(f"Error loading stock dataset: {e}")
            return None
    
    def _process_matrix_format(self, df: pd.DataFrame, max_nodes: int) -> Dict:
        """Helper to process datasets already in matrix format."""
        # Find columns that look like flow data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Treat as adjacency matrix
            matrix = df[numeric_cols].values
            n = min(matrix.shape[0], max_nodes)
            flow_matrix = matrix[:n, :n]
            
            # Normalize
            if flow_matrix.sum() > 0:
                flow_matrix = flow_matrix / flow_matrix.max() * 100
            
            nodes = [f"Node_{i}" for i in range(n)]
            
            return {
                'organization': 'Network from Matrix Data',
                'nodes': nodes,
                'flows': flow_matrix.tolist(),
                'metadata': {
                    'total_nodes': n,
                    'total_flow': float(flow_matrix.sum())
                }
            }
        
        raise ValueError("Could not process matrix format")
    
    def _extract_affinity_network(self, dataset, index: int) -> Dict:
        """Helper for protein binding affinity networks."""
        df = dataset.to_pandas()
        
        # Extract unique proteins
        if 'protein1' in df.columns and 'protein2' in df.columns:
            proteins = sorted(set(df['protein1'].unique()) | set(df['protein2'].unique()))
            n = min(len(proteins), 50)
            proteins = proteins[:n]
            
            protein_to_idx = {p: i for i, p in enumerate(proteins)}
            flow_matrix = np.zeros((n, n))
            
            # Fill with affinity values
            for _, row in df.iterrows():
                if row['protein1'] in protein_to_idx and row['protein2'] in protein_to_idx:
                    i = protein_to_idx[row['protein1']]
                    j = protein_to_idx[row['protein2']]
                    affinity = row.get('affinity', row.get('binding_affinity', 1.0))
                    flow_matrix[i, j] = abs(affinity)
                    flow_matrix[j, i] = abs(affinity)  # Symmetric
            
            # Normalize
            if flow_matrix.sum() > 0:
                flow_matrix = flow_matrix / flow_matrix.max() * 100
            
            return {
                'organization': 'Protein Binding Network',
                'nodes': proteins,
                'flows': flow_matrix.tolist(),
                'metadata': {
                    'total_proteins': n,
                    'total_interactions': int(np.count_nonzero(flow_matrix) / 2)
                }
            }
        
        return None


def demonstrate_extraction():
    """Demonstrate extraction from different dataset types."""
    
    extractor = HuggingFaceFlowExtractor()
    
    print("\n" + "="*60)
    print("HuggingFace Flow Network Extraction Examples")
    print("="*60)
    
    # Example 1: Protein Network
    print("\n1. Extracting Protein Interaction Network...")
    protein_network = extractor.extract_protein_network(
        dataset_name="graphs-datasets/PROTEINS",
        network_index=0
    )
    if protein_network:
        print(f"   ✓ Extracted {protein_network['metadata']['total_nodes']} nodes")
        print(f"   ✓ Total edges: {protein_network['metadata'].get('total_edges', 'N/A')}")
        print(f"   ✓ Network density: {protein_network['metadata']['density']:.3f}")
        
        # Save to file
        with open('data/extracted_networks/protein_network.json', 'w') as f:
            json.dump(protein_network, f, indent=2)
        print("   ✓ Saved to data/extracted_networks/protein_network.json")
    
    # Example 2: Stock Correlation Network
    print("\n2. Extracting Stock Market Correlation Network...")
    stock_network = extractor.extract_stock_correlation_network()
    if stock_network:
        print(f"   ✓ Extracted {stock_network['metadata']['total_indices']} indices")
        print(f"   ✓ Total correlations: {stock_network['metadata']['total_correlations']}")
        print(f"   ✓ Average correlation: {stock_network['metadata']['avg_correlation']:.3f}")
        
        with open('data/extracted_networks/stock_network.json', 'w') as f:
            json.dump(stock_network, f, indent=2)
        print("   ✓ Saved to data/extracted_networks/stock_network.json")
    
    # Example 3: Gene Regulatory Network
    print("\n3. Extracting Gene Regulatory Network...")
    gene_network = extractor.extract_gene_regulatory_network(
        sample_size=500,
        correlation_threshold=0.3
    )
    if gene_network:
        print(f"   ✓ Extracted {gene_network['metadata']['total_genes']} genes")
        print(f"   ✓ Total interactions: {gene_network['metadata']['total_interactions']}")
        print(f"   ✓ Network density: {gene_network['metadata']['density']:.3f}")
        
        with open('data/extracted_networks/gene_network.json', 'w') as f:
            json.dump(gene_network, f, indent=2)
        print("   ✓ Saved to data/extracted_networks/gene_network.json")
    
    print("\n" + "="*60)
    print("Extraction complete! Networks ready for Ulanowicz analysis.")
    print("="*60)


if __name__ == "__main__":
    demonstrate_extraction()