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
        Create interactive Plotly visualization of the network with enhanced features:
        - Node sizes proportional to total flow (in + out)
        - Directional arrows showing flow direction
        - Edge thickness and color based on flow intensity
        
        Args:
            G: NetworkX directed graph
            node_names: Optional list of node names
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Calculate layout
        pos = nx.spring_layout(G, seed=self.seed, k=1.5, iterations=50)
        
        # Calculate node metrics
        in_flow = {}
        out_flow = {}
        total_flow = {}
        
        for node in G.nodes():
            in_flow[node] = sum(G.edges[u, v].get('weight', 1) for u, v in G.in_edges(node))
            out_flow[node] = sum(G.edges[u, v].get('weight', 1) for u, v in G.out_edges(node))
            total_flow[node] = in_flow[node] + out_flow[node]
        
        # Normalize node sizes based on total flow
        max_flow = max(total_flow.values()) if total_flow else 1
        min_flow = min(total_flow.values()) if total_flow else 0
        flow_range = max_flow - min_flow if max_flow != min_flow else 1
        
        node_sizes = []
        for node in G.nodes():
            # Size from 40 to 100 based on total flow (larger for better visibility)
            normalized = (total_flow[node] - min_flow) / flow_range if flow_range > 0 else 0.5
            size = 40 + 60 * normalized
            node_sizes.append(size)
        
        # Get all edge weights for normalization
        all_weights = [G.edges[e].get('weight', 1) for e in G.edges()]
        max_weight = max(all_weights) if all_weights else 1
        min_weight = min(all_weights) if all_weights else 0
        weight_range = max_weight - min_weight if max_weight != min_weight else 1
        
        # Create edge traces with varying thickness and color
        edge_traces = []
        edge_annotations = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G.edges[edge].get('weight', 1)
            
            # Normalize weight for visualization
            normalized_weight = (weight - min_weight) / weight_range if weight_range > 0 else 0.5
            
            # Edge width from 0.5 to 3 based on flow
            edge_width = 0.5 + 2.5 * normalized_weight
            
            # Simple color scheme
            gray_value = int(200 - 100 * normalized_weight)
            edge_color = f'rgba({gray_value}, {gray_value}, {gray_value}, 0.7)'
            
            # Create edge line
            edge_trace = go.Scatter(
                x=[x0, x1, None], 
                y=[y0, y1, None],
                line=dict(
                    width=edge_width, 
                    color=edge_color
                ),
                hoverinfo='text',
                hovertext=f'Flow: {weight:.1f}<br>From: Node {edge[0]}<br>To: Node {edge[1]}',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
            # Add arrow annotation using Plotly's annotation system
            # Position arrow at 80% along the edge
            arrow_pos = 0.8
            arrow_x = x0 + arrow_pos * (x1 - x0)
            arrow_y = y0 + arrow_pos * (y1 - y0)
            
            # Create arrow that points from current position toward end
            arrow_end_x = arrow_x + 0.1 * (x1 - x0)
            arrow_end_y = arrow_y + 0.1 * (y1 - y0)
            
            arrow_annotation = dict(
                ax=arrow_x,
                ay=arrow_y,
                x=arrow_end_x,
                y=arrow_end_y,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=max(1.5, edge_width * 0.8),
                arrowcolor=edge_color,
            )
            edge_annotations.append(arrow_annotation)
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node label
            if node_names:
                label = node_names[node] if node < len(node_names) else f"Node {node}"
            else:
                label = f"Node {node}"
            
            # Shorten label for display if too long
            display_label = label[:12] + "..." if len(label) > 15 else label
            node_text.append(display_label)
            
            # Node info for hover with flow details
            node_info.append(
                f"<b>{label}</b><br>"
                f"<br><b>Flow Metrics:</b><br>"
                f"Inbound Flow: {in_flow[node]:.1f}<br>"
                f"Outbound Flow: {out_flow[node]:.1f}<br>"
                f"Total Flow: {total_flow[node]:.1f}<br>"
                f"<br><b>Network Position:</b><br>"
                f"In-degree: {G.in_degree(node)}<br>"
                f"Out-degree: {G.out_degree(node)}<br>"
                f"Total Connections: {G.degree(node)}"
            )
            
            # Color based on flow balance (more inflow = blue, more outflow = red, balanced = green)
            if total_flow[node] > 0:
                balance = (out_flow[node] - in_flow[node]) / total_flow[node]
                # -1 = all inflow (blue), 0 = balanced (green), 1 = all outflow (red)
                node_colors.append(balance)
            else:
                node_colors.append(0)
        
        # Create custom colorscale with better contrast
        # Using more saturated, distinct colors
        custom_colorscale = [
            [0.0, '#1e3a8a'],  # Deep blue for receivers
            [0.25, '#3b82f6'], # Bright blue
            [0.5, '#10b981'],  # Green (balanced)
            [0.75, '#f59e0b'], # Orange
            [1.0, '#dc2626']   # Red for sources
        ]
        
        # Create text annotations separately for better control
        text_annotations = []
        for idx, (x, y) in enumerate(zip(node_x, node_y)):
            if node_text[idx]:  # Only add if there's text
                # Add text with styled background
                text_annotations.append(dict(
                    x=x,
                    y=y,
                    text=f"<b>{node_text[idx]}</b>",  # Bold text for better visibility
                    showarrow=False,
                    font=dict(
                        size=13,
                        color='white',
                        family='Arial Rounded MT, Arial, sans-serif'  # Try rounded font first
                    ),
                    bgcolor='rgba(20, 20, 20, 0.5)',  # 50% opaque background
                    borderpad=8,  # Good padding
                    bordercolor='rgba(255, 255, 255, 0.3)',  # More visible white border
                    borderwidth=1,  # Slightly thicker border
                    xanchor='center',
                    yanchor='middle',
                    align='center',
                    valign='middle',
                ))
        
        # Create node trace with markers only (text will be in annotations with backgrounds)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',  # Markers only, text will be in annotations
            # text=node_text,  # Removed - using annotations instead
            # textposition="middle center",  # Removed - using annotations instead
            # textfont=dict(  # Removed - using annotations instead
            #     size=14,
            #     color='white',
            #     family='Arial, sans-serif'
            # ),
            hoverinfo='text',
            hovertext=node_info,
            customdata=list(G.nodes()),  # Store node indices for click detection
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=custom_colorscale,
                colorbar=dict(
                    title="Flow Balance",
                    tickmode='array',
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=['Receiver', 'More In', 'Balanced', 'More Out', 'Source'],
                    len=0.5,
                    y=0.5,
                    thickness=15,
                    tickfont=dict(size=11)
                ),
                line=dict(width=3, color='white'),  # Thicker white border for contrast
                cmin=-1,
                cmax=1
            )
        )
        
        # Combine all edge traces and node trace
        all_traces = edge_traces + [node_trace]
        
        # Create figure with arrow and text annotations
        fig = go.Figure(data=all_traces,
                       layout=go.Layout(
                           title=dict(text=title, font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=edge_annotations + text_annotations + [dict(
                               text="Node size = Total flow | Color = Flow balance | Edge thickness = Flow volume | Arrows show flow direction | Use dropdown to highlight connections",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.5, y=-0.05,
                               xanchor="center", yanchor="bottom",
                               font=dict(color="#666", size=10)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='#f8f9fa',  # Slightly lighter background for better contrast
                           paper_bgcolor='white'
                       ))
        
        # Add interactive highlighting
        fig = self._add_interactive_highlighting(fig, G, pos, node_names, all_traces, 
                                                in_flow, out_flow, total_flow)
        
        return fig
    
    def _add_interactive_highlighting(self, fig, G, pos, node_names, all_traces, 
                                     in_flow, out_flow, total_flow):
        """Add interactive highlighting capability to the network visualization."""
        
        # Create edge connectivity map for efficient lookup
        edge_map = {}
        for i, edge in enumerate(G.edges()):
            if edge[0] not in edge_map:
                edge_map[edge[0]] = {'out': [], 'in': []}
            if edge[1] not in edge_map:
                edge_map[edge[1]] = {'out': [], 'in': []}
            edge_map[edge[0]]['out'].append(i)
            edge_map[edge[1]]['in'].append(i)
        
        # Store original edge colors
        original_data = []
        for trace in fig.data[:-1]:  # All edge traces except the last node trace
            original_data.append({
                'line_color': trace.line.color if hasattr(trace.line, 'color') else 'gray',
                'line_width': trace.line.width if hasattr(trace.line, 'width') else 1,
                'opacity': 1.0
            })
        
        # Create dropdown menu for node selection
        buttons = [dict(
            label='🔄 Clear Selection',
            method='update',
            args=[{
                'line.color': [od['line_color'] for od in original_data],
                'line.width': [od['line_width'] for od in original_data],
                'opacity': [1.0] * len(original_data)
            },
            {'title': dict(text=fig.layout.title.text, font=dict(size=16))}],
            args2=[None]  # Reset on second click
        )]
        
        # Add button for each node
        for node_idx in G.nodes():
            # Create node label
            if node_names and node_idx < len(node_names):
                node_label = node_names[node_idx][:20]  # Truncate long names
            else:
                node_label = f"Node {node_idx}"
            
            # Prepare edge colors and widths for highlighting
            edge_colors = []
            edge_widths = []
            edge_opacities = []
            
            edge_trace_idx = 0
            for edge in G.edges():
                is_inbound = (edge[1] == node_idx)  # Edge points TO this node
                is_outbound = (edge[0] == node_idx)  # Edge points FROM this node
                
                if is_inbound:
                    edge_colors.append('rgba(59, 130, 246, 0.9)')  # Bright blue for inbound
                    edge_widths.append(4)  # Thicker line
                    edge_opacities.append(1.0)
                elif is_outbound:
                    edge_colors.append('rgba(239, 68, 68, 0.9)')  # Bright red for outbound
                    edge_widths.append(4)  # Thicker line
                    edge_opacities.append(1.0)
                else:
                    # Not connected - make it very faint
                    edge_colors.append('rgba(200, 200, 200, 0.2)')  # Very light gray
                    edge_widths.append(0.5)  # Thin line
                    edge_opacities.append(0.2)
                
                edge_trace_idx += 1
            
            # Count connections for info
            in_edges = list(G.in_edges(node_idx))
            out_edges = list(G.out_edges(node_idx))
            
            buttons.append(dict(
                label=f"📍 {node_label}",
                method='update',
                args=[{
                    'line.color': edge_colors,
                    'line.width': edge_widths,
                    'opacity': edge_opacities
                },
                {'title': dict(
                    text=f"{fig.layout.title.text}<br><sub>Selected: {node_label} | " +
                         f"<span style='color:blue'>← {len(in_edges)} inbound</span> | " +
                         f"<span style='color:red'>→ {len(out_edges)} outbound</span></sub>",
                    font=dict(size=16)
                )}]
            ))
        
        # Add the dropdown menu to the figure with enhanced styling
        fig.update_layout(
            updatemenus=[
                dict(
                    type='dropdown',
                    showactive=True,
                    buttons=buttons,
                    x=0.02,
                    xanchor='left',
                    y=0.98,
                    yanchor='top',
                    bgcolor='#e0f2fe',  # Light blue background
                    bordercolor='#1e3a8a',  # Dark blue border
                    borderwidth=2,
                    font=dict(size=13, color='#1e3a8a', family='Arial, sans-serif'),
                    active=0  # Start with "Clear Selection" active
                )
            ]
        )
        
        # Add more prominent instruction text with better styling
        fig.add_annotation(
            text="<b>👆 Select node to highlight connections</b>",
            xref="paper", yref="paper",
            x=0.35, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, color="#1e3a8a"),
            bgcolor="rgba(255, 245, 157, 0.9)",  # Light yellow background for visibility
            bordercolor="#1e3a8a",
            borderwidth=1,
            borderpad=4
        )
        
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
    
    def flow_matrix_to_network(self, flow_matrix: np.ndarray, node_names: list = None) -> nx.DiGraph:
        """
        Convert flow matrix to NetworkX directed graph.
        
        Args:
            flow_matrix: Flow matrix as numpy array
            node_names: Optional list of node names
            
        Returns:
            NetworkX directed graph with edge weights
        """
        n = len(flow_matrix)
        if node_names is None:
            node_names = [f"Node_{i+1}" for i in range(n)]
        
        G = nx.DiGraph()
        
        # Add nodes
        for i, name in enumerate(node_names):
            G.add_node(i, label=name)
        
        # Add edges with weights from flow matrix
        for i in range(n):
            for j in range(n):
                if flow_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=flow_matrix[i, j])
        
        return G
    
    def visualize_directed_network(self, G: nx.DiGraph, title: str = "Network Diagram", show_arrows: bool = True) -> go.Figure:
        """
        Create an interactive directed network visualization.
        
        Args:
            G: NetworkX directed graph
            title: Title for the visualization
            show_arrows: Whether to show arrow annotations (not used, arrows always shown)
            
        Returns:
            Plotly Figure object
        """
        # Get node labels
        node_labels = []
        for node in G.nodes():
            if 'label' in G.nodes[node]:
                node_labels.append(G.nodes[node]['label'])
            elif 'name' in G.nodes[node]:
                node_labels.append(G.nodes[node]['name'])
            else:
                node_labels.append(f"Node {node}")
        
        # Use the existing visualization method (arrows are always shown)
        return self.create_plotly_visualization(G, node_labels, title=title)


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