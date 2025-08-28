"""
Test Tier 1 HuggingFace datasets for flow network extraction
Using existing extraction code - no modifications
Each dataset will get a unique icon for menu display
"""

import sys
import json
import numpy as np
sys.path.append('src')

from huggingface_flow_extractor import HuggingFaceFlowExtractor
from ulanowicz_calculator import UlanowiczCalculator

# Define unique icons for each dataset type
DATASET_ICONS = {
    'enzyme': '🧪',      # Enzyme networks - test tube
    'molecular': '💊',   # AIDS molecular compounds - pill
    'mobility': '🚇',    # Transportation/mobility - metro
    'traffic': '📡',     # Network traffic - satellite dish
    'protein': '🧬'      # Already used for proteins
}

def test_enzymes_dataset():
    """Test ENZYMES dataset extraction."""
    
    print("\n" + "="*60)
    print("Testing ENZYMES Dataset")
    print("="*60)
    
    extractor = HuggingFaceFlowExtractor()
    
    try:
        # Use same method as PROTEINS - should work identically
        enzyme_network = extractor.extract_protein_network(
            dataset_name="graphs-datasets/ENZYMES",
            network_index=0
        )
        
        if enzyme_network:
            print(f"✓ Successfully extracted ENZYMES network!")
            print(f"  - Nodes: {enzyme_network['metadata']['total_nodes']}")
            print(f"  - Edges: {enzyme_network['metadata'].get('total_edges', 'N/A')}")
            print(f"  - Density: {enzyme_network['metadata']['density']:.3f}")
            
            # Quick viability check
            flows = np.array(enzyme_network['flows'])
            calc = UlanowiczCalculator(flows, enzyme_network['nodes'])
            metrics = calc.get_sustainability_metrics()
            
            print(f"\nSustainability Metrics:")
            print(f"  - Relative Ascendency: {metrics['relative_ascendency']:.3f}")
            print(f"  - Is Viable: {metrics['is_viable']}")
            print(f"  - Robustness: {calc.calculate_robustness():.3f}")
            
            # Update for app integration (no icon in data, just name)
            enzyme_network['organization'] = 'Enzyme Catalytic Network'
            
            # Add proper metadata
            enzyme_network['metadata'].update({
                'source': 'HuggingFace: graphs-datasets/ENZYMES',
                'dataset_info': 'Enzyme protein catalytic networks',
                'description': 'Graph representation of enzyme structures where nodes represent active sites and edges represent catalytic relationships',
                'units': 'Interaction strength (normalized 0-100)',
                'network_type': 'Biological/Enzymatic',
                'icon_suggestion': DATASET_ICONS['enzyme'],
                'menu_display': f"{DATASET_ICONS['enzyme']} Enzyme Catalytic Network"
            })
            
            # Save for use in app
            with open('data/extracted_networks/enzyme_network_test.json', 'w') as f:
                json.dump(enzyme_network, f, indent=2)
            
            return True
        else:
            print("✗ Failed to extract ENZYMES network")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_aids_dataset():
    """Test AIDS molecular graphs dataset."""
    
    print("\n" + "="*60)
    print("Testing AIDS Dataset")
    print("="*60)
    
    extractor = HuggingFaceFlowExtractor()
    
    try:
        # Use same method with correct split for AIDS dataset
        aids_network = extractor.extract_protein_network(
            dataset_name="graphs-datasets/AIDS",
            network_index=0,
            split='full'  # AIDS dataset uses 'full' split
        )
        
        if aids_network:
            print(f"✓ Successfully extracted AIDS molecular network!")
            print(f"  - Nodes: {aids_network['metadata']['total_nodes']}")
            print(f"  - Edges: {aids_network['metadata'].get('total_edges', 'N/A')}")
            print(f"  - Density: {aids_network['metadata']['density']:.3f}")
            
            # Quick viability check
            flows = np.array(aids_network['flows'])
            calc = UlanowiczCalculator(flows, aids_network['nodes'])
            metrics = calc.get_sustainability_metrics()
            
            print(f"\nSustainability Metrics:")
            print(f"  - Relative Ascendency: {metrics['relative_ascendency']:.3f}")
            print(f"  - Is Viable: {metrics['is_viable']}")
            print(f"  - Robustness: {calc.calculate_robustness():.3f}")
            
            # Update for app integration (no icon in data, just name)
            aids_network['organization'] = 'Molecular Compound Network'
            
            # Add proper metadata
            aids_network['metadata'].update({
                'source': 'HuggingFace: graphs-datasets/AIDS',
                'dataset_info': 'Anti-HIV molecular compounds',
                'description': 'Chemical compound interaction networks for AIDS drug discovery, nodes are atoms/functional groups, edges are chemical bonds',
                'units': 'Bond strength (normalized 0-100)',
                'network_type': 'Chemical/Pharmaceutical',
                'icon_suggestion': DATASET_ICONS['molecular'],
                'menu_display': f"{DATASET_ICONS['molecular']} Molecular Compound Network"
            })
            
            # Save for use in app
            with open('data/extracted_networks/aids_network_test.json', 'w') as f:
                json.dump(aids_network, f, indent=2)
            
            return True
        else:
            print("✗ Failed to extract AIDS network")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_mobility_dataset():
    """Test mobility network dataset."""
    
    print("\n" + "="*60)
    print("Testing Mobility Network Dataset")
    print("="*60)
    
    extractor = HuggingFaceFlowExtractor()
    
    try:
        # Test with Shanghai configuration first
        mobility_network = extractor.extract_mobility_network(
            dataset_name="Romz16/Mobility_Network_Data-Analysis",
            config_name="shanghai",
            max_nodes=50  # Limit for testing
        )
        
        if mobility_network:
            print(f"✓ Successfully extracted Mobility network!")
            print(f"  - Nodes: {mobility_network['metadata']['total_nodes']}")
            print(f"  - Total Flow: {mobility_network['metadata']['total_flow']:.2f}")
            print(f"  - Density: {mobility_network['metadata']['density']:.3f}")
            
            # Quick viability check
            flows = np.array(mobility_network['flows'])
            calc = UlanowiczCalculator(flows, mobility_network['nodes'])
            metrics = calc.get_sustainability_metrics()
            
            print(f"\nSustainability Metrics:")
            print(f"  - Relative Ascendency: {metrics['relative_ascendency']:.3f}")
            print(f"  - Is Viable: {metrics['is_viable']}")
            print(f"  - Robustness: {calc.calculate_robustness():.3f}")
            
            # Update for app integration (no icon in data, just name)
            mobility_network['organization'] = 'Urban Mobility Network'
            
            # Add proper metadata
            mobility_network['metadata'].update({
                'source': 'HuggingFace: Romz16/Mobility_Network_Data-Analysis',
                'dataset_info': 'Urban transportation flows',
                'description': 'Origin-destination flows in urban transportation networks, capturing movement patterns between city zones',
                'units': 'Trip frequency (normalized 0-100)',
                'network_type': 'Transportation/Urban',
                'icon_suggestion': DATASET_ICONS['mobility'],
                'menu_display': f"{DATASET_ICONS['mobility']} Urban Mobility Network"
            })
            
            # Save for use in app
            with open('data/extracted_networks/mobility_network_test.json', 'w') as f:
                json.dump(mobility_network, f, indent=2)
            
            return True
        else:
            print("✗ Failed to extract Mobility network")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: This dataset might need authentication or different config")
        return False

def main():
    """Test all Tier 1 datasets."""
    
    print("="*60)
    print("TIER 1 DATASET TESTING")
    print("Testing datasets that should work with existing extraction code")
    print("="*60)
    
    print("\nIcon assignments for menu display:")
    print("  🧬 Protein Structure Network (existing)")
    print("  🧪 Enzyme Catalytic Network (new)")
    print("  💊 Molecular Compound Network (new)")
    print("  🚇 Urban Mobility Network (new)")
    print("  📡 Network Traffic (future)")
    
    results = {
        'ENZYMES': test_enzymes_dataset(),
        'AIDS': test_aids_dataset(),
        'Mobility': test_mobility_dataset()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for dataset, success in results.items():
        status = "✓ Ready" if success else "✗ Needs adjustment"
        print(f"  {dataset}: {status}")
    
    successful = [k for k, v in results.items() if v]
    if successful:
        print(f"\n{len(successful)} datasets ready for integration!")
        print("Files saved in data/extracted_networks/ for review")
        print("\nTo add to app:")
        print("1. Review extracted JSON files")
        print("2. Copy to data/ecosystem_samples/")
        print("3. Icons will appear automatically in menu")
    else:
        print("\nSome datasets need code adjustments - please review errors")

if __name__ == "__main__":
    main()