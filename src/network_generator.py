"""
Visual Network Generator for Organizational Structures

This module provides tools for generating and visualizing different types of 
organizational networks with interactive controls for structure and flow patterns.

Based on NetworkX graph generators and Plotly interactive visualizations.
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import random


class OrganizationalNetworkGenerator:
    """
    Generator for different types of organizational network structures.
    
    Supports multiple network topologies commonly found in organizations:
    - Random (equal probability connections)
    - Scale-Free (few large hubs, many small nodes)
    - Small-World (clustered with shortcuts)
    - Hierarchical (management layers)
    - Community-Based (department clusters)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the network generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_network(self, 
                        network_type: str,
                        num_nodes: int,
                        density: float,
                        **kwargs) -> nx.DiGraph:
        """
        Generate a directed network of specified type and parameters.
        
        Args:
            network_type: Type of network structure
            num_nodes: Number of nodes in the network
            density: Target density (0.0 to 1.0)
            **kwargs: Additional parameters specific to network type
            
        Returns:
            NetworkX directed graph
        """
        generators = {
            'random': self._generate_random,
            'scale_free': self._generate_scale_free,
            'small_world': self._generate_small_world,
            'hierarchical': self._generate_hierarchical,
            'community': self._generate_community
        }
        
        if network_type not in generators:
            raise ValueError(f"Unknown network type: {network_type}")
        
        return generators[network_type](num_nodes, density, **kwargs)
    
    def _generate_random(self, num_nodes: int, density: float, **kwargs) -> nx.DiGraph:
        """Generate random (Erdős-Rényi) network."""
        # Convert density to probability for directed graph
        p = density
        G = nx.erdos_renyi_graph(num_nodes, p, directed=True)
        return G
    
    def _generate_scale_free(self, num_nodes: int, density: float, **kwargs) -> nx.DiGraph:
        """Generate scale-free (Barabási-Albert) network."""
        # Calculate m (edges to attach from new node)
        target_edges = int(density * num_nodes * (num_nodes - 1))
        m = max(1, min(num_nodes - 1, target_edges // num_nodes))
        
        G = nx.barabasi_albert_graph(num_nodes, m)
        return G.to_directed()
    
    def _generate_small_world(self, num_nodes: int, density: float, **kwargs) -> nx.DiGraph:
        """Generate small-world (Watts-Strogatz) network."""
        # Calculate k (mean degree)
        k = max(2, int(density * num_nodes))
        if k % 2 == 1:
            k += 1  # k must be even
        k = min(k, num_nodes - 1)
        
        # Rewiring probability
        p = kwargs.get('rewiring_prob', 0.3)
        
        G = nx.watts_strogatz_graph(num_nodes, k, p)
        return G.to_directed()
    
    def _generate_hierarchical(self, num_nodes: int, density: float, **kwargs) -> nx.DiGraph:
        """Generate hierarchical (tree-like) network."""
        # Create balanced tree and add some cross-connections
        r = kwargs.get('branching_factor', 2)
        
        # Calculate depth for approximately num_nodes
        depth = int(np.log(num_nodes) / np.log(r))
        G = nx.balanced_tree(r, depth)
        
        # Trim to exact number of nodes if needed
        if G.number_of_nodes() > num_nodes:
            nodes_to_remove = list(G.nodes())[num_nodes:]
            G.remove_nodes_from(nodes_to_remove)
        
        # Add some cross-connections to reach target density
        G = G.to_directed()
        current_density = nx.density(G)
        
        if current_density < density:
            nodes = list(G.nodes())
            target_edges = int(density * num_nodes * (num_nodes - 1)) - G.number_of_edges()
            
            for _ in range(target_edges):
                u, v = random.sample(nodes, 2)
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
        
        return G
    
    def _generate_community(self, num_nodes: int, density: float, **kwargs) -> nx.DiGraph:
        """Generate community-based network with department clusters."""
        num_communities = kwargs.get('num_communities', max(2, num_nodes // 5))
        
        # Divide nodes among communities
        nodes_per_community = num_nodes // num_communities
        communities = []
        
        node_id = 0
        for i in range(num_communities):
            if i == num_communities - 1:  # Last community gets remaining nodes
                community_size = num_nodes - node_id
            else:
                community_size = nodes_per_community
            
            communities.append(list(range(node_id, node_id + community_size)))
            node_id += community_size
        
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # Intra-community connections (higher probability)
        intra_density = min(0.8, density * 2)
        for community in communities:
            for u in community:
                for v in community:
                    if u != v and random.random() < intra_density:
                        G.add_edge(u, v)
        
        # Inter-community connections (lower probability)
        inter_density = max(0.1, density * 0.3)
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities):
                if i != j:
                    for u in comm1:
                        for v in comm2:
                            if random.random() < inter_density:
                                G.add_edge(u, v)
        
        return G
    
    def add_flow_weights(self, 
                        G: nx.DiGraph,
                        flow_min: float = 1.0,
                        flow_max: float = 50.0,
                        hub_amplification: float = 1.0) -> nx.DiGraph:
        """
        Add realistic flow weights to network edges.
        
        Args:
            G: NetworkX directed graph
            flow_min: Minimum flow value
            flow_max: Maximum flow value
            hub_amplification: Amplification factor for high-degree nodes
            
        Returns:
            Graph with weighted edges
        """
        G_weighted = G.copy()
        
        # Calculate node centralities for hub amplification
        centrality = nx.degree_centrality(G)
        
        for u, v in G_weighted.edges():
            # Base flow
            base_flow = random.uniform(flow_min, flow_max)
            
            # Amplify based on node centrality
            u_centrality = centrality.get(u, 0)
            v_centrality = centrality.get(v, 0)
            avg_centrality = (u_centrality + v_centrality) / 2
            
            amplified_flow = base_flow * (1 + avg_centrality * hub_amplification)
            
            G_weighted.edges[u, v]['weight'] = round(amplified_flow, 1)
        
        return G_weighted
    
    def create_plotly_visualization(self, 
                                   G: nx.DiGraph, 
                                   node_names: Optional[List[str]] = None,
                                   title: str = "Network Visualization") -> go.Figure:
        """
        Create interactive Plotly visualization of the network.
        
        Args:
            G: NetworkX directed graph
            node_names: Optional list of node names
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Calculate layout
        pos = nx.spring_layout(G, seed=self.seed)
        
        # Calculate node sizes based on degree centrality
        centrality = nx.degree_centrality(G)
        node_sizes = [20 + 40 * centrality.get(node, 0) for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G.edges[edge].get('weight', 1)
            edge_weights.append(weight)
        
        # Edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node label
            if node_names:
                label = node_names[node] if node < len(node_names) else f"Node {node}"
            else:
                label = f"Node {node}"
            
            node_text.append(label)
            
            # Node info for hover
            degree = G.degree(node)
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            node_info.append(f"{label}<br>"
                           f"Total Degree: {degree}<br>"
                           f"In-degree: {in_degree}<br>"
                           f"Out-degree: {out_degree}<br>"
                           f"Centrality: {centrality.get(node, 0):.3f}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=[centrality.get(node, 0) for node in G.nodes()],
                colorscale='Viridis',
                colorbar=dict(title="Node Centrality"),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text=title, font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def network_to_flow_matrix(self, G: nx.DiGraph) -> np.ndarray:
        """
        Convert NetworkX graph to flow matrix for Ulanowicz analysis.
        
        Args:
            G: NetworkX directed graph with edge weights
            
        Returns:
            Flow matrix as numpy array
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if G.has_edge(u, v):
                    matrix[i, j] = G.edges[u, v].get('weight', 1.0)
        
        return matrix


# Network type descriptions for UI
NETWORK_TYPES = {
    "random": {
        "name": "Random (Erdős-Rényi)",
        "description": "Equal probability connections - democratic structure",
        "characteristics": "Even distribution, no natural hierarchy",
        "use_cases": "Flat organizations, peer-to-peer networks"
    },
    "scale_free": {
        "name": "Scale-Free (Barabási-Albert)", 
        "description": "Few large hubs, many small nodes - typical corporate",
        "characteristics": "Power-law distribution, natural hierarchy",
        "use_cases": "Traditional corporations, hub-and-spoke models"
    },
    "small_world": {
        "name": "Small-World (Watts-Strogatz)",
        "description": "Clustered with shortcuts - efficient communication",
        "characteristics": "Local clustering, global connectivity",
        "use_cases": "Matrix organizations, cross-functional teams"
    },
    "hierarchical": {
        "name": "Hierarchical (Tree-like)",
        "description": "Management layers with some cross-connections",
        "characteristics": "Clear reporting lines, structured authority",
        "use_cases": "Military, government, traditional enterprises"
    },
    "community": {
        "name": "Community-Based (Departments)",
        "description": "Department clusters with inter-group connections",
        "characteristics": "Strong intra-group, weak inter-group ties",
        "use_cases": "Divisional structures, academic institutions"
    }
}