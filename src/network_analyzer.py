"""
Advanced Network Analysis Module
=================================

Implements comprehensive network science metrics for organizational flow networks.
Based on modern network science principles and complex systems theory.

Author: Network Science Expert
Date: 2024
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import stats
from networkx.algorithms import community
import pandas as pd


class AdvancedNetworkAnalyzer:
    """
    Advanced network analysis for organizational flow networks.
    
    Provides comprehensive metrics including:
    - Centrality measures (degree, betweenness, eigenvector, closeness)
    - Community detection and modularity
    - Small world analysis
    - Robustness metrics
    - Flow-specific measures
    """
    
    def __init__(self, flow_matrix: np.ndarray, node_names: List[str]):
        """
        Initialize the network analyzer.
        
        Args:
            flow_matrix: Square matrix of flows between nodes
            node_names: List of node names
        """
        self.flow_matrix = np.array(flow_matrix, dtype=float)
        self.node_names = node_names
        self.n_nodes = len(node_names)
        
        # Create NetworkX graph
        self.G = self._create_graph()
        
        # Cache for expensive computations
        self._cache = {}
    
    def _create_graph(self) -> nx.DiGraph:
        """Create NetworkX directed graph from flow matrix."""
        G = nx.DiGraph()
        
        # Add nodes with names
        for i, name in enumerate(self.node_names):
            G.add_node(i, name=name)
        
        # Add edges with weights
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.flow_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=self.flow_matrix[i, j])
        
        return G
    
    def calculate_centralities(self) -> Dict[str, Dict]:
        """
        Calculate comprehensive centrality metrics.
        
        Returns:
            Dictionary with various centrality measures
        """
        if 'centralities' in self._cache:
            return self._cache['centralities']
        
        centralities = {}
        
        # Degree centrality (normalized)
        centralities['in_degree'] = nx.in_degree_centrality(self.G)
        centralities['out_degree'] = nx.out_degree_centrality(self.G)
        centralities['total_degree'] = nx.degree_centrality(self.G)
        
        # Betweenness centrality (identifies bridges/brokers)
        try:
            centralities['betweenness'] = nx.betweenness_centrality(
                self.G, weight='weight', normalized=True
            )
        except:
            centralities['betweenness'] = {i: 0 for i in range(self.n_nodes)}
        
        # Eigenvector centrality (influence measure)
        try:
            centralities['eigenvector'] = nx.eigenvector_centrality(
                self.G, max_iter=1000, weight='weight'
            )
        except:
            # Fall back to degree if eigenvector fails
            centralities['eigenvector'] = centralities['total_degree']
        
        # Closeness centrality (accessibility)
        try:
            centralities['closeness'] = nx.closeness_centrality(
                self.G, distance='weight'
            )
        except:
            centralities['closeness'] = {i: 0 for i in range(self.n_nodes)}
        
        # PageRank (Google's algorithm, variant of eigenvector)
        try:
            centralities['pagerank'] = nx.pagerank(
                self.G, weight='weight', alpha=0.85
            )
        except:
            centralities['pagerank'] = {i: 1/self.n_nodes for i in range(self.n_nodes)}
        
        # Katz centrality (considers all paths)
        try:
            centralities['katz'] = nx.katz_centrality(
                self.G, weight='weight', alpha=0.1, normalized=True
            )
        except:
            centralities['katz'] = centralities['total_degree']
        
        self._cache['centralities'] = centralities
        return centralities
    
    def detect_communities(self) -> Dict[str, Any]:
        """
        Detect communities using multiple algorithms.
        
        Returns:
            Dictionary with community assignments and modularity scores
        """
        if 'communities' in self._cache:
            return self._cache['communities']
        
        results = {}
        
        # Convert to undirected for community detection
        G_undirected = self.G.to_undirected()
        
        # Louvain algorithm (fast and effective)
        try:
            louvain_communities = community.louvain_communities(
                G_undirected, weight='weight', seed=42
            )
            results['louvain'] = {
                'communities': louvain_communities,
                'modularity': community.modularity(
                    G_undirected, louvain_communities, weight='weight'
                ),
                'num_communities': len(louvain_communities)
            }
        except:
            results['louvain'] = {'communities': [], 'modularity': 0, 'num_communities': 0}
        
        # Label propagation (fast, non-deterministic)
        try:
            label_prop = community.label_propagation_communities(G_undirected)
            label_prop_list = list(label_prop)
            results['label_propagation'] = {
                'communities': label_prop_list,
                'modularity': community.modularity(
                    G_undirected, label_prop_list, weight='weight'
                ) if label_prop_list else 0,
                'num_communities': len(label_prop_list)
            }
        except:
            results['label_propagation'] = {'communities': [], 'modularity': 0, 'num_communities': 0}
        
        # Greedy modularity optimization
        try:
            greedy_communities = community.greedy_modularity_communities(
                G_undirected, weight='weight'
            )
            results['greedy_modularity'] = {
                'communities': list(greedy_communities),
                'modularity': community.modularity(
                    G_undirected, greedy_communities, weight='weight'
                ),
                'num_communities': len(list(greedy_communities))
            }
        except:
            results['greedy_modularity'] = {'communities': [], 'modularity': 0, 'num_communities': 0}
        
        self._cache['communities'] = results
        return results
    
    def calculate_small_world_metrics(self) -> Dict[str, float]:
        """
        Calculate small world metrics.
        
        Small world networks have high clustering and short path lengths.
        
        Returns:
            Dictionary with small world metrics
        """
        if 'small_world' in self._cache:
            return self._cache['small_world']
        
        metrics = {}
        
        # Convert to undirected for analysis
        G_undirected = self.G.to_undirected()
        
        # Actual metrics
        try:
            actual_clustering = nx.average_clustering(G_undirected, weight='weight')
        except:
            actual_clustering = 0
        
        try:
            if nx.is_connected(G_undirected):
                actual_path_length = nx.average_shortest_path_length(G_undirected)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                subgraph = G_undirected.subgraph(largest_cc)
                actual_path_length = nx.average_shortest_path_length(subgraph)
        except:
            actual_path_length = float('inf')
        
        # Random graph comparison (Erdős-Rényi)
        n = self.n_nodes
        m = G_undirected.number_of_edges()
        p = 2 * m / (n * (n - 1)) if n > 1 else 0
        
        # Theoretical random graph values
        random_clustering = p
        random_path_length = np.log(n) / np.log(nx.average_degree_connectivity(G_undirected).get(1, 2)) if n > 1 else 1
        
        # Small world index (sigma)
        if random_clustering > 0 and random_path_length > 0 and actual_path_length < float('inf'):
            C_ratio = actual_clustering / random_clustering if random_clustering > 0 else 1
            L_ratio = actual_path_length / random_path_length if random_path_length > 0 else 1
            sigma = C_ratio / L_ratio if L_ratio > 0 else 0
        else:
            sigma = 0
        
        # Omega small world metric (alternative measure)
        # Range: -1 (lattice) to 0 (small world) to 1 (random)
        if actual_path_length < float('inf'):
            omega = (random_path_length / actual_path_length) - (actual_clustering / random_clustering) \
                    if random_clustering > 0 else 0
        else:
            omega = 0
        
        metrics = {
            'clustering_coefficient': actual_clustering,
            'average_path_length': actual_path_length,
            'random_clustering': random_clustering,
            'random_path_length': random_path_length,
            'small_world_sigma': sigma,  # > 1 indicates small world
            'small_world_omega': omega,  # Close to 0 indicates small world
            'is_small_world': sigma > 1
        }
        
        self._cache['small_world'] = metrics
        return metrics
    
    def calculate_assortativity(self) -> Dict[str, float]:
        """
        Calculate assortativity metrics.
        
        Assortativity measures the preference for nodes to connect to similar nodes.
        
        Returns:
            Dictionary with assortativity metrics
        """
        metrics = {}
        
        # Degree assortativity (do hubs connect to hubs?)
        try:
            metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(
                self.G, weight='weight'
            )
        except:
            metrics['degree_assortativity'] = 0
        
        # In-degree and out-degree assortativity for directed graphs
        try:
            metrics['in_degree_assortativity'] = nx.degree_assortativity_coefficient(
                self.G, x='in', y='in', weight='weight'
            )
            metrics['out_degree_assortativity'] = nx.degree_assortativity_coefficient(
                self.G, x='out', y='out', weight='weight'
            )
        except:
            metrics['in_degree_assortativity'] = 0
            metrics['out_degree_assortativity'] = 0
        
        return metrics
    
    def calculate_rich_club_coefficient(self, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate rich club coefficient.
        
        Measures the tendency of high-degree nodes to connect to each other.
        
        Args:
            k: Degree threshold (default: top 10% of nodes)
        
        Returns:
            Dictionary with rich club metrics
        """
        # Convert to undirected
        G_undirected = self.G.to_undirected()
        
        # Default k to top 10% degree
        if k is None:
            degrees = dict(G_undirected.degree())
            if degrees:
                k = int(np.percentile(list(degrees.values()), 90))
            else:
                k = 1
        
        try:
            # Calculate rich club coefficient
            rc = nx.rich_club_coefficient(G_undirected, normalized=False)
            
            # Get the coefficient at threshold k
            rc_at_k = rc.get(k, 0) if rc else 0
            
            return {
                'rich_club_coefficient': rc_at_k,
                'threshold_k': k,
                'full_spectrum': rc
            }
        except:
            return {
                'rich_club_coefficient': 0,
                'threshold_k': k,
                'full_spectrum': {}
            }
    
    def calculate_robustness_metrics(self, num_simulations: int = 10) -> Dict[str, Any]:
        """
        Calculate network robustness metrics.
        
        Args:
            num_simulations: Number of attack simulations
        
        Returns:
            Dictionary with robustness metrics
        """
        metrics = {}
        
        # Original giant component size
        if nx.is_weakly_connected(self.G):
            original_gcc_size = self.n_nodes
        else:
            original_gcc_size = len(max(nx.weakly_connected_components(self.G), key=len))
        
        # Random failure simulation
        random_robustness = []
        for _ in range(num_simulations):
            G_copy = self.G.copy()
            removed = 0
            gcc_sizes = [original_gcc_size]
            
            while G_copy.number_of_nodes() > 1:
                # Remove random node
                if G_copy.nodes():
                    node = np.random.choice(list(G_copy.nodes()))
                    G_copy.remove_node(node)
                    removed += 1
                    
                    # Measure giant component
                    if G_copy.nodes():
                        gcc = max(nx.weakly_connected_components(G_copy), key=len)
                        gcc_sizes.append(len(gcc))
                    else:
                        gcc_sizes.append(0)
                else:
                    break
            
            # Calculate area under curve (robustness)
            random_robustness.append(np.mean(gcc_sizes) / original_gcc_size)
        
        metrics['random_failure_robustness'] = np.mean(random_robustness)
        
        # Targeted attack (remove high-degree nodes)
        G_copy = self.G.copy()
        gcc_sizes = [original_gcc_size]
        
        while G_copy.number_of_nodes() > 1:
            # Remove highest degree node
            if G_copy.nodes():
                degrees = dict(G_copy.degree())
                if degrees:
                    max_node = max(degrees, key=degrees.get)
                    G_copy.remove_node(max_node)
                    
                    # Measure giant component
                    if G_copy.nodes():
                        if nx.weakly_connected_components(G_copy):
                            gcc = max(nx.weakly_connected_components(G_copy), key=len)
                            gcc_sizes.append(len(gcc))
                        else:
                            gcc_sizes.append(0)
                    else:
                        gcc_sizes.append(0)
                else:
                    break
            else:
                break
        
        metrics['targeted_attack_robustness'] = np.mean(gcc_sizes) / original_gcc_size if original_gcc_size > 0 else 0
        
        # Percolation threshold estimate
        avg_degree = 2 * self.G.number_of_edges() / self.n_nodes if self.n_nodes > 0 else 0
        metrics['percolation_threshold'] = 1 / avg_degree if avg_degree > 0 else 1
        
        # Redundancy (alternative paths)
        path_redundancy = []
        for i in range(min(10, self.n_nodes)):
            for j in range(min(10, self.n_nodes)):
                if i != j:
                    try:
                        paths = list(nx.all_simple_paths(self.G, i, j, cutoff=3))
                        if len(paths) > 1:
                            path_redundancy.append(len(paths))
                    except:
                        pass
        
        metrics['path_redundancy'] = np.mean(path_redundancy) if path_redundancy else 0
        
        return metrics
    
    def calculate_flow_metrics(self) -> Dict[str, float]:
        """
        Calculate flow-specific network metrics.
        
        Returns:
            Dictionary with flow metrics
        """
        metrics = {}
        
        # Flow concentration (Gini coefficient)
        flows = self.flow_matrix[self.flow_matrix > 0]
        if len(flows) > 0:
            sorted_flows = np.sort(flows)
            n = len(sorted_flows)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_flows)) / (n * np.sum(sorted_flows)) - (n + 1) / n
            metrics['flow_gini_coefficient'] = gini
        else:
            metrics['flow_gini_coefficient'] = 0
        
        # Flow heterogeneity
        if len(flows) > 0:
            metrics['flow_heterogeneity'] = np.std(flows) / np.mean(flows)
        else:
            metrics['flow_heterogeneity'] = 0
        
        # Throughput efficiency
        total_flow = np.sum(self.flow_matrix)
        max_possible_flow = self.n_nodes * (self.n_nodes - 1) * np.max(self.flow_matrix) if self.flow_matrix.size > 0 else 0
        metrics['throughput_efficiency'] = total_flow / max_possible_flow if max_possible_flow > 0 else 0
        
        # Flow reciprocity (bidirectional flows)
        reciprocal_flows = 0
        total_edges = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.flow_matrix[i, j] > 0 and self.flow_matrix[j, i] > 0:
                    reciprocal_flows += 1
                if self.flow_matrix[i, j] > 0 or self.flow_matrix[j, i] > 0:
                    total_edges += 1
        
        metrics['flow_reciprocity'] = reciprocal_flows / total_edges if total_edges > 0 else 0
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return all network metrics.
        
        Returns:
            Comprehensive dictionary of all network metrics
        """
        all_metrics = {}
        
        # Basic topology
        all_metrics['basic'] = {
            'num_nodes': self.n_nodes,
            'num_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'is_connected': nx.is_weakly_connected(self.G),
            'num_components': nx.number_weakly_connected_components(self.G)
        }
        
        # Centralities
        all_metrics['centralities'] = self.calculate_centralities()
        
        # Communities
        all_metrics['communities'] = self.detect_communities()
        
        # Small world
        all_metrics['small_world'] = self.calculate_small_world_metrics()
        
        # Assortativity
        all_metrics['assortativity'] = self.calculate_assortativity()
        
        # Rich club
        all_metrics['rich_club'] = self.calculate_rich_club_coefficient()
        
        # Robustness
        all_metrics['robustness'] = self.calculate_robustness_metrics()
        
        # Flow metrics
        all_metrics['flow'] = self.calculate_flow_metrics()
        
        return all_metrics
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary of key network metrics.
        
        Returns:
            Formatted text report
        """
        metrics = self.get_all_metrics()
        
        report = "=" * 60 + "\n"
        report += "NETWORK ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Basic info
        report += f"Network Size: {metrics['basic']['num_nodes']} nodes, {metrics['basic']['num_edges']} edges\n"
        report += f"Density: {metrics['basic']['density']:.3f}\n"
        report += f"Connected: {metrics['basic']['is_connected']}\n\n"
        
        # Small world
        sw = metrics['small_world']
        report += "SMALL WORLD PROPERTIES:\n"
        report += f"  Clustering: {sw['clustering_coefficient']:.3f} (random: {sw['random_clustering']:.3f})\n"
        report += f"  Path Length: {sw['average_path_length']:.2f} (random: {sw['random_path_length']:.2f})\n"
        report += f"  Small World σ: {sw['small_world_sigma']:.2f} {'✓ Small World' if sw['is_small_world'] else '✗ Not Small World'}\n\n"
        
        # Communities
        comm = metrics['communities']
        if 'louvain' in comm and comm['louvain']['modularity'] > 0:
            report += "COMMUNITY STRUCTURE:\n"
            report += f"  Number of Communities: {comm['louvain']['num_communities']}\n"
            report += f"  Modularity: {comm['louvain']['modularity']:.3f}\n\n"
        
        # Robustness
        rob = metrics['robustness']
        report += "ROBUSTNESS:\n"
        report += f"  Random Failure: {rob['random_failure_robustness']:.3f}\n"
        report += f"  Targeted Attack: {rob['targeted_attack_robustness']:.3f}\n"
        report += f"  Path Redundancy: {rob['path_redundancy']:.2f}\n\n"
        
        # Flow
        flow = metrics['flow']
        report += "FLOW CHARACTERISTICS:\n"
        report += f"  Flow Inequality (Gini): {flow['flow_gini_coefficient']:.3f}\n"
        report += f"  Flow Reciprocity: {flow['flow_reciprocity']:.3f}\n"
        report += f"  Throughput Efficiency: {flow['throughput_efficiency']:.3f}\n"
        
        report += "\n" + "=" * 60
        
        return report