"""
DBLP Co-authorship Network Extractor
Extracts a subset of DBLP collaboration network for flow analysis
"""

import json
import gzip
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import random

class DBLPExtractor:
    def __init__(self):
        self.authors = set()
        self.edges = defaultdict(int)
        
    def extract_from_snap(self, filepath: str, max_nodes: int = 50, min_degree: int = 10) -> Dict:
        """
        Extract DBLP network from SNAP edge list format
        """
        print(f"Reading DBLP network from {filepath}...")
        
        # First pass: count collaborations
        edge_counts = defaultdict(int)
        author_neighbors = defaultdict(set)
        
        import gzip
        opener = gzip.open if filepath.endswith('.gz') or '.txt' in filepath else open
        mode = 'rt' if filepath.endswith('.gz') or '.txt' in filepath else 'r'
        
        line_count = 0
        with opener(filepath, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    author1, author2 = int(parts[0]), int(parts[1])
                    if author1 != author2:  # Skip self-loops
                        edge_counts[(min(author1, author2), max(author1, author2))] += 1
                        author_neighbors[author1].add(author2)
                        author_neighbors[author2].add(author1)
                        
                line_count += 1
                # Sample first 100k edges
                if line_count > 100000:
                    break
        
        print(f"Found {len(author_neighbors)} authors in sample")
        
        # Find authors with sufficient connections
        eligible_authors = {auth: len(neighbors) 
                          for auth, neighbors in author_neighbors.items() 
                          if len(neighbors) >= min_degree}
        
        print(f"Found {len(eligible_authors)} authors with degree >= {min_degree}")
        
        # Select top connected authors
        top_authors = sorted(eligible_authors.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        selected = set([a for a, _ in top_authors])
        
        print(f"Selected {len(selected)} highly connected authors")
        
        # Create mapping to sequential IDs
        author_map = {auth: idx for idx, auth in enumerate(sorted(selected))}
        
        # Build flow matrix
        n = len(author_map)
        flows = np.zeros((n, n))
        
        for (a1, a2), count in edge_counts.items():
            if a1 in author_map and a2 in author_map:
                idx1, idx2 = author_map[a1], author_map[a2]
                # Bidirectional flow (collaboration)
                # Scale based on collaboration strength
                flow_value = min(count * 20, 100)  # Cap at 100
                flows[idx1][idx2] = flow_value
                flows[idx2][idx1] = flow_value
        
        # Create node names
        nodes = [f"Author_{i+1}" for i in range(n)]
        
        # Calculate metadata
        total_edges = np.count_nonzero(flows)
        density = total_edges / (n * (n - 1)) if n > 1 else 0
        
        return {
            "organization": "DBLP Co-authorship Network",
            "nodes": nodes,
            "flows": flows.tolist(),
            "metadata": {
                "source": "SNAP Stanford - DBLP Collaboration Network",
                "dataset_info": "Computer science research collaboration network",
                "description": "Co-authorship relationships between highly connected CS researchers",
                "units": "Number of joint publications (x10)",
                "network_type": "Academic/Research Collaboration",
                "total_nodes": n,
                "total_edges": total_edges,
                "density": density,
                "icon_suggestion": "🎓",
                "collaboration_context": {
                    "structure_type": "Academic collaboration network",
                    "node_represents": "Computer Science researchers",
                    "edge_represents": "Joint publications",
                    "min_degree": min_degree,
                    "classification": "Research ecosystem"
                }
            }
        }
    
    def create_synthetic_dblp(self, num_nodes: int = 30) -> Dict:
        """
        Create a synthetic DBLP-like network for testing
        """
        np.random.seed(42)
        
        # Create research groups with strong internal collaboration
        groups = 5
        nodes_per_group = num_nodes // groups
        
        nodes = []
        for g in range(groups):
            group_name = ["AI", "Database", "Security", "Networks", "Theory"][g]
            for i in range(nodes_per_group):
                nodes.append(f"{group_name}_Researcher_{i+1}")
        
        # Add extra nodes if needed
        while len(nodes) < num_nodes:
            nodes.append(f"Interdisciplinary_Researcher_{len(nodes)-groups*nodes_per_group+1}")
        
        # Create flow matrix with group structure
        flows = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                group_i = i // nodes_per_group
                group_j = j // nodes_per_group
                
                if group_i == group_j and group_i < groups:
                    # Same group: high collaboration
                    if random.random() < 0.7:
                        collab = random.randint(20, 100)
                        flows[i][j] = collab
                        flows[j][i] = collab
                elif abs(group_i - group_j) == 1 or (i >= groups * nodes_per_group or j >= groups * nodes_per_group):
                    # Adjacent groups or interdisciplinary: moderate collaboration
                    if random.random() < 0.3:
                        collab = random.randint(5, 30)
                        flows[i][j] = collab
                        flows[j][i] = collab
                else:
                    # Distant groups: rare collaboration
                    if random.random() < 0.1:
                        collab = random.randint(1, 10)
                        flows[i][j] = collab
                        flows[j][i] = collab
        
        total_edges = np.count_nonzero(flows)
        density = total_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "organization": "DBLP Co-authorship Network (Sample)",
            "nodes": nodes,
            "flows": flows.tolist(),
            "metadata": {
                "source": "Synthetic DBLP-like network based on real collaboration patterns",
                "dataset_info": "Computer science research collaboration network",
                "description": "Simulated co-authorship network with 5 research areas",
                "units": "Number of joint publications",
                "network_type": "Academic/Research Collaboration",
                "total_nodes": num_nodes,
                "total_edges": total_edges,
                "density": density,
                "icon_suggestion": "🎓",
                "collaboration_context": {
                    "structure_type": "Academic collaboration network",
                    "node_represents": "Computer Science researchers",
                    "edge_represents": "Joint publications",
                    "research_areas": ["AI", "Database", "Security", "Networks", "Theory"],
                    "classification": "Research ecosystem"
                }
            }
        }

if __name__ == "__main__":
    extractor = DBLPExtractor()
    
    # Try to process downloaded file
    import os
    if os.path.exists("/tmp/dblp_sample.txt"):
        try:
            network = extractor.extract_from_snap("/tmp/dblp_sample.txt", max_nodes=30, min_degree=15)
        except Exception as e:
            print(f"Could not process SNAP file: {e}")
            print("Creating synthetic DBLP network instead...")
            network = extractor.create_synthetic_dblp(30)
    else:
        print("Creating synthetic DBLP network...")
        network = extractor.create_synthetic_dblp(30)
    
    # Save to JSON
    output_path = "data/ecosystem_samples/dblp_coauthorship_network.json"
    with open(output_path, 'w') as f:
        json.dump(network, f, indent=2)
    
    print(f"\nNetwork saved to {output_path}")
    print(f"Nodes: {network['metadata']['total_nodes']}")
    print(f"Edges: {network['metadata']['total_edges']}")
    print(f"Density: {network['metadata']['density']:.3f}")