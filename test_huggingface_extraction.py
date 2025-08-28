"""
Test HuggingFace dataset extraction with PROTEINS dataset
"""

import sys
import json
import numpy as np
sys.path.append('src')

from huggingface_flow_extractor import HuggingFaceFlowExtractor
from ulanowicz_calculator import UlanowiczCalculator

def test_protein_network_extraction():
    """Test extraction and analysis of protein network."""
    
    print("="*60)
    print("Testing HuggingFace Protein Network Extraction")
    print("="*60)
    
    # Initialize extractor
    extractor = HuggingFaceFlowExtractor()
    
    # Extract protein network
    print("\n1. Extracting protein interaction network from HuggingFace...")
    try:
        protein_network = extractor.extract_protein_network(
            dataset_name="graphs-datasets/PROTEINS",
            network_index=0
        )
        
        if protein_network:
            print(f"✓ Successfully extracted network!")
            print(f"  - Nodes: {protein_network['metadata']['total_nodes']}")
            print(f"  - Edges: {protein_network['metadata'].get('total_edges', 'N/A')}")
            print(f"  - Density: {protein_network['metadata']['density']:.3f}")
            
            # Save extracted network
            output_file = 'data/extracted_networks/test_protein_network.json'
            with open(output_file, 'w') as f:
                json.dump(protein_network, f, indent=2)
            print(f"✓ Saved to {output_file}")
            
            # Validate format
            print("\n2. Validating flow matrix format...")
            flows = np.array(protein_network['flows'])
            nodes = protein_network['nodes']
            
            print(f"  - Flow matrix shape: {flows.shape}")
            print(f"  - Is square matrix: {flows.shape[0] == flows.shape[1]}")
            print(f"  - Number of nodes matches: {len(nodes) == flows.shape[0]}")
            print(f"  - Non-negative values: {np.all(flows >= 0)}")
            print(f"  - Total flow: {flows.sum():.2f}")
            
            # Run Ulanowicz analysis
            print("\n3. Running Ulanowicz sustainability analysis...")
            calculator = UlanowiczCalculator(flows, nodes)
            metrics = calculator.get_extended_metrics()
            
            print("\nKey Sustainability Metrics:")
            print(f"  - Total System Throughput: {metrics['total_system_throughput']:.2f}")
            print(f"  - Development Capacity: {metrics['development_capacity']:.2f}")
            print(f"  - Ascendency: {metrics['ascendency']:.2f}")
            print(f"  - Relative Ascendency (α): {metrics['relative_ascendency']:.3f}")
            print(f"  - Robustness: {metrics['robustness']:.3f}")
            print(f"  - Network Efficiency: {metrics['network_efficiency']:.3f}")
            print(f"  - Is Viable: {metrics['is_viable']}")
            
            # Assess sustainability
            assessment = calculator.assess_sustainability()
            print(f"\nSustainability Assessment: {assessment}")
            
            # Check window of viability
            if metrics['relative_ascendency'] < 0.2:
                print("  → System is too chaotic (α < 0.2)")
            elif metrics['relative_ascendency'] > 0.6:
                print("  → System is too rigid (α > 0.6)")
            else:
                print("  → System is within window of viability (0.2 ≤ α ≤ 0.6)")
            
            print("\n✓ Test completed successfully!")
            return True
            
        else:
            print("✗ Failed to extract network")
            return False
            
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_protein_network_extraction()
    if success:
        print("\n" + "="*60)
        print("SUCCESS: HuggingFace dataset can be used with Ulanowicz analysis!")
        print("The extracted network is ready for use in the main application.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("FAILED: Issues encountered during extraction or analysis")
        print("="*60)