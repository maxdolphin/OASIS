"""
Pharmaceutical and Manufacturing Network Extractor
Processes ogbg-molhiv and AIDS datasets into flow networks
"""

import json
import numpy as np
from typing import Dict, List, Optional
import random

class PharmaManufacturingExtractor:
    
    def extract_ogbg_molhiv(self, sample_size: int = 50) -> Dict:
        """
        Extract ogbg-molhiv as pharmaceutical development network
        """
        try:
            from ogb.graphproppred import GraphPropPredDataset
            print("Loading ogbg-molhiv dataset from OGB...")
            
            dataset = GraphPropPredDataset(name="ogbg-molhiv")
            
            # Get a representative molecule
            graph, label = dataset[0]
            num_nodes = graph['num_nodes']
            edge_index = graph['edge_index']
            
            print(f"Loaded molecule with {num_nodes} atoms")
            
            # For large molecules, sample or aggregate
            if num_nodes > sample_size:
                # Sample nodes
                sampled_nodes = random.sample(range(num_nodes), sample_size)
                num_nodes = sample_size
            else:
                sampled_nodes = list(range(num_nodes))
            
            # Create adjacency matrix
            flows = np.zeros((num_nodes, num_nodes))
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0][i], edge_index[1][i]
                if src in sampled_nodes and dst in sampled_nodes:
                    src_idx = sampled_nodes.index(src)
                    dst_idx = sampled_nodes.index(dst)
                    # Bidirectional chemical bond strength
                    flow_value = random.randint(40, 100)
                    flows[src_idx][dst_idx] = flow_value
                    flows[dst_idx][src_idx] = flow_value
            
            nodes = self._generate_pharma_nodes(num_nodes)
            
        except Exception as e:
            print(f"Could not load ogbg-molhiv: {e}")
            print("Creating synthetic pharmaceutical network...")
            return self.create_synthetic_pharma_network(sample_size)
        
        total_edges = np.count_nonzero(flows)
        density = total_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "organization": "Pharmaceutical Development Network",
            "nodes": nodes,
            "flows": flows.tolist(),
            "metadata": {
                "source": "OGB ogbg-molhiv - HIV inhibitor screening",
                "dataset_info": "Pharmaceutical R&D and drug development pipeline",
                "description": "Drug discovery network from research to production",
                "units": "Development resources allocated (%)",
                "network_type": "Pharmaceutical/R&D",
                "total_nodes": num_nodes,
                "total_edges": total_edges,
                "density": density,
                "icon_suggestion": "💰",
                "pharma_context": {
                    "structure_type": "Drug development pipeline",
                    "node_represents": "Research stages, labs, production facilities",
                    "edge_represents": "Resource flows and dependencies",
                    "classification": "Pharmaceutical ecosystem"
                }
            }
        }
    
    def extract_aids_dataset(self, max_nodes: int = 100) -> Dict:
        """
        Extract AIDS dataset as manufacturing production network
        """
        try:
            from datasets import load_dataset
            print("Loading AIDS dataset from HuggingFace...")
            
            # Try to load AIDS dataset
            dataset = load_dataset("graphs-datasets/AIDS", split="full")
            
            # Get first graph
            graph = dataset[0]
            num_nodes = graph['num_nodes']
            edge_index = graph['edge_index']
            
            print(f"Loaded graph with {num_nodes} nodes")
            
            # Limit size if too large
            if num_nodes > max_nodes:
                num_nodes = max_nodes
            
            # Create adjacency matrix
            flows = np.zeros((num_nodes, num_nodes))
            
            for i in range(len(edge_index[0])):
                src, dst = edge_index[0][i], edge_index[1][i]
                if src < num_nodes and dst < num_nodes:
                    # Production flow volumes
                    flow_value = random.randint(100, 500)
                    flows[src][dst] = flow_value
            
            nodes = self._generate_manufacturing_nodes(num_nodes)
            
        except Exception as e:
            print(f"Could not load AIDS dataset: {e}")
            print("Creating synthetic manufacturing network...")
            return self.create_synthetic_manufacturing_network(max_nodes)
        
        total_edges = np.count_nonzero(flows)
        density = total_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "organization": "Manufacturing Production Network",
            "nodes": nodes,
            "flows": flows.tolist(),
            "metadata": {
                "source": "AIDS dataset adapted as manufacturing network",
                "dataset_info": "Large-scale industrial production network",
                "description": "Manufacturing facilities with production flows",
                "units": "Units produced per day",
                "network_type": "Manufacturing/Industrial",
                "total_nodes": num_nodes,
                "total_edges": total_edges,
                "density": density,
                "icon_suggestion": "🏭",
                "manufacturing_context": {
                    "structure_type": "Multi-facility production network",
                    "node_represents": "Production lines, assembly stations, warehouses",
                    "edge_represents": "Material and product flows",
                    "classification": "Industrial ecosystem"
                }
            }
        }
    
    def _generate_pharma_nodes(self, num_nodes: int) -> List[str]:
        """Generate pharmaceutical development node names"""
        stages = [
            "Research_Lab", "Preclinical_Test", "Clinical_Trial_Phase",
            "Regulatory_Review", "Manufacturing_Facility", "Quality_Control",
            "Distribution_Hub", "Hospital_Network", "Pharmacy_Chain"
        ]
        
        nodes = []
        for i in range(num_nodes):
            stage = stages[i % len(stages)]
            nodes.append(f"{stage}_{i+1}")
        
        return nodes
    
    def _generate_manufacturing_nodes(self, num_nodes: int) -> List[str]:
        """Generate manufacturing facility node names"""
        facilities = [
            "Raw_Material_Storage", "Processing_Unit", "Assembly_Line",
            "Quality_Station", "Packaging_Unit", "Warehouse",
            "Distribution_Center", "Shipping_Hub"
        ]
        
        nodes = []
        for i in range(num_nodes):
            facility = facilities[i % len(facilities)]
            nodes.append(f"{facility}_{i+1}")
        
        return nodes
    
    def create_synthetic_pharma_network(self, num_nodes: int = 35) -> Dict:
        """Create synthetic pharmaceutical development network"""
        np.random.seed(43)
        
        # Define pharmaceutical pipeline stages
        stages = {
            "research": list(range(0, 8)),           # Research labs
            "preclinical": list(range(8, 14)),       # Preclinical testing
            "clinical": list(range(14, 20)),         # Clinical trials
            "regulatory": list(range(20, 24)),       # Regulatory approval
            "manufacturing": list(range(24, 30)),    # Manufacturing
            "distribution": list(range(30, num_nodes)) # Distribution
        }
        
        nodes = []
        for stage_name, indices in stages.items():
            for i, idx in enumerate(indices):
                if stage_name == "research":
                    nodes.append(f"Research_Lab_{i+1}")
                elif stage_name == "preclinical":
                    nodes.append(f"Preclinical_Test_{i+1}")
                elif stage_name == "clinical":
                    nodes.append(f"Clinical_Trial_Site_{i+1}")
                elif stage_name == "regulatory":
                    nodes.append(f"Regulatory_Office_{i+1}")
                elif stage_name == "manufacturing":
                    nodes.append(f"Pharma_Factory_{i+1}")
                else:
                    nodes.append(f"Distribution_Hub_{i+1}")
        
        # Ensure exact node count
        nodes = nodes[:num_nodes]
        while len(nodes) < num_nodes:
            nodes.append(f"Support_Facility_{len(nodes)+1}")
        
        # Create flow matrix with pipeline logic
        flows = np.zeros((num_nodes, num_nodes))
        
        # Research -> Preclinical
        for r in stages["research"]:
            for p in stages["preclinical"]:
                if random.random() < 0.4:
                    flows[r][p] = random.randint(30, 80)
        
        # Preclinical -> Clinical
        for p in stages["preclinical"]:
            for c in stages["clinical"]:
                if random.random() < 0.5:
                    flows[p][c] = random.randint(40, 90)
        
        # Clinical -> Regulatory
        for c in stages["clinical"]:
            for r in stages["regulatory"]:
                if random.random() < 0.6:
                    flows[c][r] = random.randint(50, 100)
        
        # Regulatory -> Manufacturing
        for r in stages["regulatory"]:
            for m in stages["manufacturing"]:
                if random.random() < 0.7:
                    flows[r][m] = random.randint(60, 120)
        
        # Manufacturing -> Distribution
        for m in stages["manufacturing"]:
            for d in stages["distribution"]:
                if random.random() < 0.8:
                    flows[m][d] = random.randint(80, 150)
        
        # Add feedback loops (clinical data back to research)
        for c in stages["clinical"][:2]:
            for r in stages["research"][:2]:
                if random.random() < 0.3:
                    flows[c][r] = random.randint(10, 30)
        
        total_edges = np.count_nonzero(flows)
        density = total_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "organization": "Pharmaceutical Development Network",
            "nodes": nodes,
            "flows": flows.tolist(),
            "metadata": {
                "source": "Synthetic pharmaceutical R&D pipeline",
                "dataset_info": "Drug development from research to market",
                "description": "End-to-end pharmaceutical development pipeline",
                "units": "Development resources allocated (%)",
                "network_type": "Pharmaceutical/R&D",
                "total_nodes": num_nodes,
                "total_edges": total_edges,
                "density": density,
                "icon_suggestion": "💰",
                "pharma_context": {
                    "structure_type": "6-stage drug development pipeline",
                    "stages": ["Research", "Preclinical", "Clinical", "Regulatory", "Manufacturing", "Distribution"],
                    "node_represents": "Development facilities and sites",
                    "edge_represents": "Resource and information flows",
                    "classification": "Pharmaceutical ecosystem"
                }
            }
        }
    
    def create_synthetic_manufacturing_network(self, num_nodes: int = 60) -> Dict:
        """Create synthetic manufacturing production network"""
        np.random.seed(44)
        
        # Define manufacturing tiers
        tiers = {
            "suppliers": list(range(0, 10)),
            "preprocessing": list(range(10, 20)),
            "manufacturing": list(range(20, 35)),
            "assembly": list(range(35, 45)),
            "quality": list(range(45, 52)),
            "distribution": list(range(52, num_nodes))
        }
        
        nodes = []
        for tier_name, indices in tiers.items():
            for i, idx in enumerate(indices):
                if tier_name == "suppliers":
                    nodes.append(f"Supplier_Warehouse_{i+1}")
                elif tier_name == "preprocessing":
                    nodes.append(f"Processing_Station_{i+1}")
                elif tier_name == "manufacturing":
                    nodes.append(f"Production_Line_{i+1}")
                elif tier_name == "assembly":
                    nodes.append(f"Assembly_Station_{i+1}")
                elif tier_name == "quality":
                    nodes.append(f"QC_Station_{i+1}")
                else:
                    nodes.append(f"Distribution_Hub_{i+1}")
        
        # Ensure exact node count
        nodes = nodes[:num_nodes]
        while len(nodes) < num_nodes:
            nodes.append(f"Facility_{len(nodes)+1}")
        
        # Create flow matrix
        flows = np.zeros((num_nodes, num_nodes))
        
        # Suppliers -> Preprocessing
        for s in tiers["suppliers"]:
            for p in tiers["preprocessing"]:
                if random.random() < 0.5:
                    flows[s][p] = random.randint(100, 300)
        
        # Preprocessing -> Manufacturing
        for p in tiers["preprocessing"]:
            for m in tiers["manufacturing"]:
                if random.random() < 0.6:
                    flows[p][m] = random.randint(150, 350)
        
        # Manufacturing -> Assembly
        for m in tiers["manufacturing"]:
            for a in tiers["assembly"]:
                if random.random() < 0.7:
                    flows[m][a] = random.randint(200, 400)
        
        # Assembly -> Quality
        for a in tiers["assembly"]:
            for q in tiers["quality"]:
                if random.random() < 0.8:
                    flows[a][q] = random.randint(180, 380)
        
        # Quality -> Distribution
        for q in tiers["quality"]:
            for d in tiers["distribution"]:
                if random.random() < 0.75:
                    flows[q][d] = random.randint(150, 350)
        
        # Add rework loops (quality back to manufacturing)
        for q in tiers["quality"][:3]:
            for m in tiers["manufacturing"][:3]:
                if random.random() < 0.2:
                    flows[q][m] = random.randint(20, 80)
        
        # Add express lanes (direct manufacturing to distribution)
        for m in tiers["manufacturing"][:3]:
            for d in tiers["distribution"][:2]:
                if random.random() < 0.15:
                    flows[m][d] = random.randint(50, 150)
        
        total_edges = np.count_nonzero(flows)
        density = total_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "organization": "Manufacturing Production Network",
            "nodes": nodes,
            "flows": flows.tolist(),
            "metadata": {
                "source": "Synthetic large-scale manufacturing network",
                "dataset_info": "Industrial production and assembly network",
                "description": "Multi-tier manufacturing with quality control loops",
                "units": "Units produced per day",
                "network_type": "Manufacturing/Industrial",
                "total_nodes": num_nodes,
                "total_edges": total_edges,
                "density": density,
                "icon_suggestion": "🏭",
                "manufacturing_context": {
                    "structure_type": "6-tier production network",
                    "tiers": ["Suppliers", "Preprocessing", "Manufacturing", "Assembly", "Quality Control", "Distribution"],
                    "node_represents": "Production facilities and stations",
                    "edge_represents": "Material and product flows",
                    "includes_rework": True,
                    "classification": "Industrial ecosystem"
                }
            }
        }

if __name__ == "__main__":
    extractor = PharmaManufacturingExtractor()
    
    # Extract pharmaceutical network
    print("=" * 50)
    print("Extracting Pharmaceutical Development Network...")
    pharma_network = extractor.extract_ogbg_molhiv(sample_size=35)
    
    output_path = "data/ecosystem_samples/pharma_development_network.json"
    with open(output_path, 'w') as f:
        json.dump(pharma_network, f, indent=2)
    
    print(f"Pharma network saved to {output_path}")
    print(f"Nodes: {pharma_network['metadata']['total_nodes']}")
    print(f"Edges: {pharma_network['metadata']['total_edges']}")
    print(f"Density: {pharma_network['metadata']['density']:.3f}")
    
    # Extract manufacturing network
    print("\n" + "=" * 50)
    print("Extracting Manufacturing Production Network...")
    manufacturing_network = extractor.extract_aids_dataset(max_nodes=60)
    
    output_path = "data/ecosystem_samples/manufacturing_network.json"
    with open(output_path, 'w') as f:
        json.dump(manufacturing_network, f, indent=2)
    
    print(f"Manufacturing network saved to {output_path}")
    print(f"Nodes: {manufacturing_network['metadata']['total_nodes']}")
    print(f"Edges: {manufacturing_network['metadata']['total_edges']}")
    print(f"Density: {manufacturing_network['metadata']['density']:.3f}")