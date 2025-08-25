#!/usr/bin/env python3
"""
MetricsTracker - Persistent tracking system for published vs calculated metrics
Version 2.1.4
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ulanowicz_calculator import UlanowiczCalculator


class MetricsTracker:
    """Track and compare published metrics from papers with our calculations."""
    
    def __init__(self, database_path: str = "validation/metrics_database"):
        """Initialize the metrics tracker."""
        self.database_path = Path(database_path)
        self.networks_path = self.database_path / "networks"
        self.reports_path = self.database_path / "reports"
        self.registry_path = self.database_path / "master_registry.json"
        
        # Load registry
        self.registry = self._load_registry()
        self.networks = {}
        
    def _load_registry(self) -> Dict:
        """Load the master registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"networks": []}
    
    def load_network(self, network_id: str) -> Dict:
        """Load a network's metrics from database."""
        network_file = self.networks_path / f"{network_id}.json"
        if network_file.exists():
            with open(network_file, 'r') as f:
                network_data = json.load(f)
                self.networks[network_id] = network_data
                return network_data
        else:
            raise FileNotFoundError(f"Network {network_id} not found in database")
    
    def load_all_networks(self) -> Dict[str, Dict]:
        """Load all networks from the database."""
        for network_info in self.registry.get('networks', []):
            network_id = network_info['id']
            try:
                self.load_network(network_id)
            except FileNotFoundError:
                print(f"Warning: Network {network_id} in registry but file not found")
        return self.networks
    
    def calculate_metrics(self, network_id: str) -> Dict:
        """Calculate all metrics for a network using our implementation."""
        if network_id not in self.networks:
            self.load_network(network_id)
        
        network = self.networks[network_id]
        flows = np.array(network['network_data']['flows'])
        nodes = network['network_data']['nodes']
        
        # Calculate using our implementation
        calculator = UlanowiczCalculator(flows, node_names=nodes)
        basic_metrics = calculator.get_sustainability_metrics()
        extended_metrics = calculator.get_extended_metrics()
        
        # Update calculated metrics
        calculated = {
            'tst': {
                'value': basic_metrics['total_system_throughput'],
                'unit': network['network_data']['units'],
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            },
            'development_capacity': {
                'value': basic_metrics['development_capacity'],
                'unit': 'flow-nats',
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            },
            'ascendency': {
                'value': basic_metrics['ascendency'],
                'unit': 'flow-nats',
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            },
            'reserve': {
                'value': basic_metrics['reserve'],
                'unit': 'flow-nats',
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            },
            'relative_ascendency': {
                'value': basic_metrics['relative_ascendency'],
                'unit': 'dimensionless',
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            },
            'robustness': {
                'value': extended_metrics['robustness'],
                'unit': 'dimensionless',
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            }
        }
        
        network['calculated_metrics'].update(calculated)
        return calculated
    
    def compare_metrics(self, network_id: str) -> Dict:
        """Compare published vs calculated metrics for a network."""
        if network_id not in self.networks:
            self.load_network(network_id)
        
        network = self.networks[network_id]
        published = network['published_metrics']
        calculated = network['calculated_metrics']
        
        comparison = {}
        
        # Compare each metric
        for metric in ['tst', 'ascendency', 'reserve', 'relative_ascendency']:
            pub_val = published[metric].get('value')
            calc_val = calculated[metric].get('value')
            
            if pub_val is not None and calc_val is not None:
                error_abs = abs(calc_val - pub_val)
                error_pct = (error_abs / pub_val * 100) if pub_val != 0 else 0
                
                # Determine status
                if error_pct < 1:
                    status = "EXCELLENT"
                elif error_pct < 5:
                    status = "GOOD"
                elif error_pct < 15:
                    status = "MODERATE"
                elif error_pct < 30:
                    status = "DISCREPANCY"
                else:
                    status = "MAJOR_DISCREPANCY"
                
                comparison[f"{metric}_match"] = {
                    'published': pub_val,
                    'calculated': calc_val,
                    'error_abs': error_abs,
                    'error_pct': error_pct,
                    'status': status
                }
        
        # Check fundamental relationship
        C = calculated['development_capacity']['value']
        A = calculated['ascendency']['value']
        Phi = calculated['reserve']['value']
        
        if C and A and Phi:
            c_check = A + Phi
            error = abs(c_check - C) / C * 100 if C != 0 else 0
            comparison['fundamental_check'] = {
                'C_equals_A_plus_Phi': bool(error < 0.01),  # Convert numpy bool to Python bool
                'calculated_C': float(C),  # Convert to float for JSON
                'calculated_A_plus_Phi': float(c_check),
                'error': float(error),
                'status': 'PERFECT' if error < 0.01 else 'ERROR'
            }
        
        network['comparison'] = comparison
        return comparison
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table for all networks."""
        data = []
        
        for network_id, network in self.networks.items():
            row = {
                'Network': network['network_name'],
                'Source': network['source']['paper']
            }
            
            # Add published metrics
            for metric in ['tst', 'ascendency', 'reserve', 'relative_ascendency']:
                pub_val = network['published_metrics'][metric].get('value')
                row[f'{metric}_published'] = pub_val if pub_val is not None else 'N/A'
            
            # Add calculated metrics
            for metric in ['tst', 'ascendency', 'reserve', 'relative_ascendency']:
                calc_val = network['calculated_metrics'][metric].get('value')
                row[f'{metric}_calculated'] = calc_val
            
            # Add comparison status
            if 'comparison' in network:
                for metric in ['tst', 'ascendency', 'reserve', 'relative_ascendency']:
                    match_key = f"{metric}_match"
                    if match_key in network['comparison']:
                        row[f'{metric}_status'] = network['comparison'][match_key]['status']
                        row[f'{metric}_error_%'] = network['comparison'][match_key]['error_pct']
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_network(self, network_id: str):
        """Save network metrics back to database."""
        if network_id in self.networks:
            network_file = self.networks_path / f"{network_id}.json"
            with open(network_file, 'w') as f:
                json.dump(self.networks[network_id], f, indent=2)
    
    def generate_report(self, output_file: str = None):
        """Generate a comprehensive validation report."""
        report = {
            'title': 'Metrics Validation Report',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': '2.1.4',
            'summary': {},
            'networks': []
        }
        
        total_networks = len(self.networks)
        excellent_matches = 0
        discrepancies = 0
        
        for network_id, network in self.networks.items():
            network_report = {
                'id': network_id,
                'name': network['network_name'],
                'source': network['source']['paper'],
                'metrics_comparison': {}
            }
            
            if 'comparison' in network:
                for key, value in network['comparison'].items():
                    if 'status' in value:
                        network_report['metrics_comparison'][key] = value
                        if value['status'] == 'EXCELLENT':
                            excellent_matches += 1
                        elif 'DISCREPANCY' in value['status']:
                            discrepancies += 1
            
            report['networks'].append(network_report)
        
        report['summary'] = {
            'total_networks': total_networks,
            'excellent_matches': excellent_matches,
            'discrepancies': discrepancies,
            'validation_rate': f"{(excellent_matches / (excellent_matches + discrepancies) * 100):.1f}%" if (excellent_matches + discrepancies) > 0 else "N/A"
        }
        
        # Save report
        if output_file:
            report_path = self.reports_path / output_file
        else:
            report_path = self.reports_path / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print a summary of all networks and their validation status."""
        print("\n" + "="*80)
        print("METRICS TRACKING SUMMARY")
        print("="*80)
        
        df = self.generate_comparison_table()
        
        for _, row in df.iterrows():
            print(f"\n{row['Network']}")
            print("-" * 40)
            
            for metric in ['tst', 'relative_ascendency']:
                pub = row.get(f'{metric}_published', 'N/A')
                calc = row.get(f'{metric}_calculated', 'N/A')
                status = row.get(f'{metric}_status', 'N/A')
                error = row.get(f'{metric}_error_%', 0)
                
                if pub != 'N/A' and calc != 'N/A':
                    print(f"{metric.upper():>5}: Published={pub:<10} Calculated={calc:<10.3f} "
                          f"Error={error:>6.1f}% Status={status}")
                else:
                    print(f"{metric.upper():>5}: Published={pub:<10} Calculated={calc:<10.3f}")
        
        print("\n" + "="*80)


def main():
    """Test the MetricsTracker system."""
    tracker = MetricsTracker()
    
    # Load all networks
    print("Loading networks...")
    tracker.load_all_networks()
    print(f"Loaded {len(tracker.networks)} networks")
    
    # Calculate metrics for each network
    print("\nCalculating metrics...")
    for network_id in tracker.networks.keys():
        print(f"  Calculating for {network_id}...")
        tracker.calculate_metrics(network_id)
        tracker.compare_metrics(network_id)
        tracker.save_network(network_id)
    
    # Generate report
    print("\nGenerating report...")
    report = tracker.generate_report()
    
    # Print summary
    tracker.print_summary()
    
    print(f"\nValidation report saved to: {tracker.reports_path}")


if __name__ == "__main__":
    main()