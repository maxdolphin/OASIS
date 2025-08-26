#!/usr/bin/env python3
"""
Process All High-Priority Real-World Datasets

Script to process all high-priority datasets into our standard format.
This creates ready-to-use flow matrices for immediate analysis.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.processors.energy_processor import EuropeanPowerGridProcessor, SmartGridProcessor
from validation.processors.supply_chain_processor import DataCoSupplyChainProcessor, LogisticsSupplyChainProcessor
from validation.processors.financial_processor import PaySimFinancialProcessor, BankingTransactionsProcessor
from validation.processors.official_processor import OECDInputOutputProcessor, EurostatMaterialFlowProcessor, WTOTradeProcessor


def create_directories():
    """Create all necessary directories for processed datasets."""
    base_dir = project_root / "data" / "real_world_datasets"
    
    directories = [
        base_dir / "energy" / "metadata",
        base_dir / "supply_chain" / "metadata", 
        base_dir / "financial" / "metadata",
        base_dir / "trade_materials" / "metadata"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def get_all_processors() -> List[Dict[str, Any]]:
    """Get all processor configurations."""
    return [
        # Energy Flow Datasets (High Priority)
        {
            "processor": EuropeanPowerGridProcessor(),
            "category": "energy",
            "priority": "HIGH",
            "description": "Continental power grid network"
        },
        {
            "processor": SmartGridProcessor(),
            "category": "energy", 
            "priority": "MEDIUM",
            "description": "Real-time smart grid monitoring"
        },
        
        # Supply Chain Datasets (High Priority) 
        {
            "processor": DataCoSupplyChainProcessor(),
            "category": "supply_chain",
            "priority": "HIGH", 
            "description": "Multi-tier supply chain network"
        },
        {
            "processor": LogisticsSupplyChainProcessor(),
            "category": "supply_chain",
            "priority": "MEDIUM",
            "description": "Modern logistics network"
        },
        
        # Financial Flow Datasets (High Priority)
        {
            "processor": PaySimFinancialProcessor(),
            "category": "financial",
            "priority": "HIGH",
            "description": "Mobile money transaction network"
        },
        {
            "processor": BankingTransactionsProcessor(),
            "category": "financial",
            "priority": "MEDIUM", 
            "description": "Banking transaction network"
        },
        
        # Official Data Sources (Research Priority)
        {
            "processor": OECDInputOutputProcessor(),
            "category": "trade_materials",
            "priority": "RESEARCH",
            "description": "International economic flows"
        },
        {
            "processor": EurostatMaterialFlowProcessor(),
            "category": "trade_materials",
            "priority": "RESEARCH",
            "description": "EU material flows"
        },
        {
            "processor": WTOTradeProcessor(), 
            "category": "trade_materials",
            "priority": "RESEARCH",
            "description": "Global trade network"
        }
    ]


def process_dataset(config: Dict[str, Any], base_dir: Path) -> bool:
    """Process a single dataset."""
    processor = config["processor"]
    category = config["category"]
    priority = config["priority"]
    description = config["description"]
    
    print(f"\n🔄 Processing {processor.dataset_name}")
    print(f"   Category: {category}")
    print(f"   Priority: {priority}")
    print(f"   Description: {description}")
    print("-" * 60)
    
    # Set output directory
    output_dir = base_dir / category
    
    # Process the dataset
    try:
        result = processor.process_dataset(output_dir)
        
        if result:
            print(f"✅ Successfully processed: {processor.dataset_name}")
            print(f"📁 Saved to: {result}")
            return True
        else:
            print(f"❌ Failed to process: {processor.dataset_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {processor.dataset_name}: {e}")
        return False


def main():
    """Process all high-priority datasets."""
    print("🌍 Processing All High-Priority Real-World Datasets")
    print("=" * 70)
    
    # Create directories
    print("\n📁 Creating directory structure...")
    create_directories()
    
    # Get all processors
    processors = get_all_processors()
    base_dir = project_root / "data" / "real_world_datasets"
    
    # Process datasets by priority
    priorities = ["HIGH", "MEDIUM", "RESEARCH"]
    results = {"success": 0, "failed": 0, "total": len(processors)}
    
    for priority in priorities:
        priority_processors = [p for p in processors if p["priority"] == priority]
        
        if priority_processors:
            print(f"\n🎯 Processing {priority} Priority Datasets ({len(priority_processors)} datasets)")
            print("=" * 70)
            
            for config in priority_processors:
                success = process_dataset(config, base_dir)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📊 Total: {results['total']}")
    print(f"📈 Success Rate: {results['success']/results['total']*100:.1f}%")
    
    if results["success"] > 0:
        print(f"\n🚀 {results['success']} datasets ready for analysis!")
        print("   Navigate to 'Use Sample Data' → '🌍 Real Life Data' to access them")
        
    print("\n📚 Next Steps:")
    print("   1. Test processed datasets in the web interface")
    print("   2. Run analysis on real-world networks")
    print("   3. Compare results across different domains")
    print("   4. Validate calculations with published benchmarks")
    
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())