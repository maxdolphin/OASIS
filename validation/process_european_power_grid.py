#!/usr/bin/env python3
"""
Process European Power Grid Dataset

Script to process the European Power Grid Network dataset into our standard format.
This creates a ready-to-use flow matrix for immediate analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.processors.energy_processor import EuropeanPowerGridProcessor


def main():
    """Process the European Power Grid dataset."""
    print("🔌 Processing European Power Grid Network Dataset")
    print("=" * 60)
    
    # Create processor
    processor = EuropeanPowerGridProcessor()
    
    # Set output directory
    output_dir = project_root / "data" / "real_world_datasets" / "energy"
    
    # Process the dataset
    result = processor.process_dataset(output_dir)
    
    if result:
        print(f"✅ Successfully processed dataset!")
        print(f"📁 Saved to: {result}")
        print("\n📊 Dataset ready for analysis through the web interface!")
        print("   Navigate to 'Use Sample Data' → '🌍 Real Life Data' → 'European Power Grid Network'")
    else:
        print("❌ Failed to process dataset")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())