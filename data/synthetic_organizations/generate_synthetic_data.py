#!/usr/bin/env python3
"""
Synthetic Data Generator for Organizational Communication Networks

This script generates realistic synthetic data for organizational communication
patterns including email exchanges, document sharing, and combined flows.
"""

import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class OrganizationalDataGenerator:
    """Generator for synthetic organizational communication data."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        self.output_dir = Path(__file__).parent
    
    def generate_organization_structure(self, 
                                      org_name: str,
                                      org_type: str,
                                      departments: List[str],
                                      employees_per_dept: Dict[str, int]) -> Dict:
        """Generate organizational structure data."""
        
        structure = {
            "organization_name": org_name,
            "organization_type": org_type,
            "total_employees": sum(employees_per_dept.values()),
            "departments": {},
            "communication_patterns": {
                "high_frequency": [],
                "medium_frequency": [],
                "low_frequency": []
            }
        }
        
        # Generate department details
        for dept in departments:
            emp_count = employees_per_dept.get(dept, 3)
            employees = [f"{dept}_{i+1}" for i in range(emp_count)]
            
            structure["departments"][dept] = {
                "employees": employees,
                "role": self._generate_dept_role(dept),
                "external_collaborators": self._generate_external_collaborators(dept)
            }
        
        # Generate communication patterns
        structure["communication_patterns"] = self._generate_communication_patterns(departments)
        
        return structure
    
    def generate_email_flow_matrix(self, 
                                 departments: List[str],
                                 communication_intensity: str = "medium") -> Tuple[np.ndarray, Dict]:
        """Generate realistic email flow matrix between departments."""
        
        n_depts = len(departments)
        flow_matrix = np.zeros((n_depts, n_depts))
        
        # Base flow parameters based on intensity
        intensity_params = {
            "low": {"base": 15, "variance": 10, "peak_multiplier": 2.0},
            "medium": {"base": 30, "variance": 20, "peak_multiplier": 2.5},
            "high": {"base": 50, "variance": 30, "peak_multiplier": 3.0}
        }
        
        params = intensity_params[communication_intensity]
        
        # Generate flows based on department relationships
        for i, from_dept in enumerate(departments):
            for j, to_dept in enumerate(departments):
                if i != j:  # No self-loops
                    base_flow = params["base"]
                    
                    # Adjust based on department relationships
                    relationship_multiplier = self._get_dept_relationship_strength(from_dept, to_dept)
                    
                    # Add random variance
                    variance = np.random.normal(0, params["variance"])
                    
                    flow = max(1, base_flow * relationship_multiplier + variance)
                    flow_matrix[i, j] = round(flow, 1)
        
        # Generate metadata
        metadata = {
            "generation_method": "synthetic_realistic_patterns",
            "intensity": communication_intensity,
            "assumptions": self._generate_email_assumptions(),
            "flow_characteristics": self._analyze_flow_characteristics(flow_matrix, departments)
        }
        
        return flow_matrix, metadata
    
    def generate_document_flow_matrix(self, 
                                    departments: List[str],
                                    formality_level: str = "medium") -> Tuple[np.ndarray, Dict]:
        """Generate document sharing flow matrix (typically lower volume, higher weight)."""
        
        n_depts = len(departments)
        flow_matrix = np.zeros((n_depts, n_depts))
        
        # Document flows are typically 20-40% of email volume but more important
        formality_params = {
            "low": {"base": 8, "variance": 5, "admin_multiplier": 1.5},
            "medium": {"base": 15, "variance": 8, "admin_multiplier": 2.0},
            "high": {"base": 25, "variance": 12, "admin_multiplier": 3.0}
        }
        
        params = formality_params[formality_level]
        
        for i, from_dept in enumerate(departments):
            for j, to_dept in enumerate(departments):
                if i != j:
                    base_flow = params["base"]
                    
                    # Administrative departments generate more documents
                    if from_dept in ["Executive", "HR", "Finance", "Operations"]:
                        base_flow *= params["admin_multiplier"]
                    
                    # Relationship strength (similar but different from email)
                    relationship_multiplier = self._get_dept_document_relationship(from_dept, to_dept)
                    
                    variance = np.random.normal(0, params["variance"])
                    flow = max(1, base_flow * relationship_multiplier + variance)
                    flow_matrix[i, j] = round(flow, 1)
        
        # Generate document type metadata
        metadata = {
            "generation_method": "document_flow_modeling",
            "formality_level": formality_level,
            "document_types": self._generate_document_types(departments),
            "assumptions": self._generate_document_assumptions()
        }
        
        return flow_matrix, metadata
    
    def generate_combined_flows(self, 
                              email_matrix: np.ndarray,
                              document_matrix: np.ndarray,
                              departments: List[str],
                              email_weight: float = 0.6,
                              document_weight: float = 1.4) -> Tuple[np.ndarray, Dict]:
        """Combine email and document flows with appropriate weighting."""
        
        combined_matrix = (email_matrix * email_weight) + (document_matrix * document_weight)
        
        metadata = {
            "weighting_scheme": {
                "email_weight": email_weight,
                "document_weight": document_weight,
                "rationale": "Documents carry more strategic/formal information weight than emails"
            },
            "flow_analysis": self._analyze_combined_flows(combined_matrix, departments),
            "network_properties": self._calculate_network_properties(combined_matrix)
        }
        
        return combined_matrix, metadata
    
    def _get_dept_relationship_strength(self, from_dept: str, to_dept: str) -> float:
        """Get relationship strength multiplier between departments for email."""
        
        # High-collaboration pairs
        high_collab = [
            ("Product", "Engineering"), ("Engineering", "Product"),
            ("Sales", "Customer_Success"), ("Customer_Success", "Sales"),
            ("Sales", "Marketing"), ("Marketing", "Sales"),
            ("Executive", "Product"), ("Product", "Executive"),
            ("Executive", "Sales"), ("Sales", "Executive")
        ]
        
        # Medium-collaboration pairs
        medium_collab = [
            ("Engineering", "Data_Science"), ("Data_Science", "Engineering"),
            ("Product", "Marketing"), ("Marketing", "Product"),
            ("Operations", "Engineering"), ("Engineering", "Operations"),
            ("Finance", "Executive"), ("Executive", "Finance")
        ]
        
        if (from_dept, to_dept) in high_collab:
            return np.random.uniform(2.0, 3.5)
        elif (from_dept, to_dept) in medium_collab:
            return np.random.uniform(1.2, 2.0)
        else:
            return np.random.uniform(0.3, 1.0)
    
    def _get_dept_document_relationship(self, from_dept: str, to_dept: str) -> float:
        """Get relationship strength for document sharing (different patterns than email)."""
        
        # Documents flow more from administrative/strategic departments
        if from_dept == "Executive":
            return np.random.uniform(1.5, 2.5)
        elif from_dept in ["HR", "Finance", "Operations"]:
            return np.random.uniform(1.0, 2.0)
        elif from_dept == "Product" and to_dept == "Engineering":
            return np.random.uniform(2.5, 3.5)
        elif from_dept == "Engineering" and to_dept == "Product":
            return np.random.uniform(2.0, 3.0)
        elif from_dept == "Sales" and to_dept == "Customer_Success":
            return np.random.uniform(2.0, 3.0)
        else:
            return np.random.uniform(0.2, 1.2)
    
    def _generate_dept_role(self, dept: str) -> str:
        """Generate role description for department."""
        roles = {
            "Executive": "Strategic leadership and decision making",
            "Product": "Product strategy and design",
            "Engineering": "Software development and technical implementation",
            "Data_Science": "Data analysis and machine learning",
            "Sales": "Revenue generation and client relationships",
            "Marketing": "Brand awareness and lead generation",
            "Customer_Success": "Customer support and retention",
            "Operations": "Business operations and infrastructure",
            "HR": "Human resources and talent management",
            "Finance": "Financial planning and analysis"
        }
        return roles.get(dept, f"{dept} operations and management")
    
    def _generate_external_collaborators(self, dept: str) -> List[str]:
        """Generate external collaborators for department."""
        collaborators = {
            "Executive": ["Board_Member", "Investor_Rep", "Strategic_Advisor"],
            "Product": ["UX_Consultant", "Market_Researcher", "Industry_Expert"],
            "Engineering": ["Tech_Consultant", "Security_Auditor", "Open_Source_Contributor"],
            "Data_Science": ["AI_Consultant", "Data_Vendor", "Research_Partner"],
            "Sales": ["Channel_Partner", "Sales_Consultant", "Industry_Contact"],
            "Marketing": ["PR_Agency", "Digital_Agency", "Content_Creator"],
            "Customer_Success": ["Support_Contractor", "Training_Partner", "Customer_Advocate"],
            "Operations": ["IT_Vendor", "Legal_Counsel", "Compliance_Consultant"],
            "HR": ["Recruiting_Agency", "Benefits_Provider", "Training_Provider"],
            "Finance": ["Accountant", "Tax_Advisor", "Financial_Consultant"]
        }
        base_list = collaborators.get(dept, ["External_Partner", "Consultant"])
        return random.sample(base_list, min(len(base_list), 3))
    
    def _generate_communication_patterns(self, departments: List[str]) -> Dict:
        """Generate communication frequency patterns."""
        patterns = {"high_frequency": [], "medium_frequency": [], "low_frequency": []}
        
        # Define some high-frequency pairs
        high_freq = [
            ("Executive", "Product"), ("Product", "Engineering"),
            ("Sales", "Customer_Success"), ("Marketing", "Sales")
        ]
        
        patterns["high_frequency"] = [pair for pair in high_freq if pair[0] in departments and pair[1] in departments]
        
        # Medium frequency - remaining important pairs
        for dept in departments:
            if dept == "Executive":
                for other in ["Sales", "Engineering", "Finance"]:
                    if other in departments and (dept, other) not in patterns["high_frequency"]:
                        patterns["medium_frequency"].append((dept, other))
        
        return patterns
    
    def _generate_email_assumptions(self) -> List[str]:
        """Generate assumptions for email flow generation."""
        return [
            "Higher flows between departments with frequent collaboration",
            "Executive has moderate outgoing communication to all departments", 
            "Product-Engineering has highest bidirectional flow",
            "Sales-Customer_Success has high communication due to customer handoffs",
            "Marketing-Sales has high communication for lead management"
        ]
    
    def _generate_document_assumptions(self) -> List[str]:
        """Generate assumptions for document flow generation."""
        return [
            "Document sharing is less frequent but more formal than emails",
            "Executive generates many policy/strategy documents distributed widely",
            "Administrative departments (HR, Finance, Ops) generate procedural documents",
            "Product-Engineering exchange technical specifications frequently"
        ]
    
    def _generate_document_types(self, departments: List[str]) -> Dict:
        """Generate document types for each department."""
        doc_types = {}
        
        type_mapping = {
            "Executive": {
                "outgoing": ["Strategic plans", "Board reports", "Policy documents"],
                "incoming": ["Department reports", "Financial summaries", "Project updates"]
            },
            "Product": {
                "outgoing": ["Product requirements", "User stories", "Wireframes"],
                "incoming": ["Market research", "User feedback", "Technical constraints"]
            },
            "Engineering": {
                "outgoing": ["Technical specifications", "Architecture docs", "Code reviews"],
                "incoming": ["Product requirements", "Bug reports", "User stories"]
            },
            # Add more as needed...
        }
        
        for dept in departments:
            if dept in type_mapping:
                doc_types[dept] = type_mapping[dept]
            else:
                doc_types[dept] = {
                    "outgoing": [f"{dept} reports", f"{dept} documentation"],
                    "incoming": ["Requirements", "Feedback", "Updates"]
                }
        
        return doc_types
    
    def _analyze_flow_characteristics(self, matrix: np.ndarray, departments: List[str]) -> Dict:
        """Analyze characteristics of flow matrix."""
        analysis = {
            "total_flows": np.sum(matrix),
            "average_flow": np.mean(matrix[matrix > 0]),
            "max_flow": np.max(matrix),
            "min_positive_flow": np.min(matrix[matrix > 0])
        }
        
        # Find peak flows
        peak_indices = np.unravel_index(np.argsort(matrix.ravel())[-3:], matrix.shape)
        analysis["peak_flows"] = []
        
        for i, j in zip(peak_indices[0], peak_indices[1]):
            if matrix[i, j] > 0:
                analysis["peak_flows"].append({
                    "from": departments[i],
                    "to": departments[j], 
                    "value": matrix[i, j]
                })
        
        return analysis
    
    def _analyze_combined_flows(self, matrix: np.ndarray, departments: List[str]) -> Dict:
        """Analyze combined flow matrix."""
        analysis = {}
        
        # Find strongest connections
        flat_indices = np.argsort(matrix.ravel())[-5:]
        indices = np.unravel_index(flat_indices, matrix.shape)
        
        analysis["strongest_connections"] = []
        for i, j in zip(indices[0], indices[1]):
            if matrix[i, j] > 0:
                analysis["strongest_connections"].append({
                    "from": departments[i],
                    "to": departments[j],
                    "value": round(matrix[i, j], 1)
                })
        
        # Calculate department totals (hubs)
        outflows = np.sum(matrix, axis=1)
        analysis["key_hubs"] = []
        
        for i, dept in enumerate(departments):
            analysis["key_hubs"].append({
                "department": dept,
                "total_outflow": round(outflows[i], 1)
            })
        
        analysis["key_hubs"].sort(key=lambda x: x["total_outflow"], reverse=True)
        
        return analysis
    
    def _calculate_network_properties(self, matrix: np.ndarray) -> Dict:
        """Calculate basic network properties."""
        return {
            "total_system_throughput": round(np.sum(matrix), 1),
            "average_flow_per_connection": round(np.mean(matrix[matrix > 0]), 1),
            "network_density": round(np.count_nonzero(matrix) / (matrix.size - matrix.shape[0]), 3),
            "flow_variance": round(np.var(matrix[matrix > 0]), 1)
        }
    
    def save_organization_data(self, 
                             org_name: str,
                             org_type: str = "Technology Company",
                             departments: Optional[List[str]] = None,
                             communication_intensity: str = "medium",
                             formality_level: str = "medium"):
        """Generate and save complete organizational data set."""
        
        if departments is None:
            departments = [
                "Executive", "Product", "Engineering", "Data_Science", "Sales",
                "Marketing", "Customer_Success", "Operations", "HR", "Finance"
            ]
        
        # Generate employees per department
        employees_per_dept = {dept: np.random.randint(2, 8) for dept in departments}
        
        print(f"Generating synthetic data for {org_name}...")
        
        # 1. Organization structure
        org_structure = self.generate_organization_structure(
            org_name, org_type, departments, employees_per_dept
        )
        
        structure_file = self.output_dir / "organizational_structures" / f"{org_name.lower().replace(' ', '_')}.json"
        with open(structure_file, 'w') as f:
            json.dump(org_structure, f, indent=2)
        print(f"✅ Saved organizational structure: {structure_file}")
        
        # 2. Email flows
        email_matrix, email_metadata = self.generate_email_flow_matrix(
            departments, communication_intensity
        )
        
        email_data = {
            "organization": org_name,
            "data_type": "email_exchange_flows",
            "period": "monthly_average",
            "units": "emails_per_month",
            "nodes": departments,
            "flows": email_matrix.tolist(),
            "metadata": email_metadata
        }
        
        email_file = self.output_dir / "email_flows" / f"{org_name.lower().replace(' ', '_')}_email_matrix.json"
        with open(email_file, 'w') as f:
            json.dump(email_data, f, indent=2)
        print(f"✅ Saved email flows: {email_file}")
        
        # 3. Document flows
        doc_matrix, doc_metadata = self.generate_document_flow_matrix(
            departments, formality_level
        )
        
        doc_data = {
            "organization": org_name,
            "data_type": "document_sharing_flows",
            "period": "monthly_average", 
            "units": "documents_shared_per_month",
            "nodes": departments,
            "flows": doc_matrix.tolist(),
            "metadata": doc_metadata
        }
        
        doc_file = self.output_dir / "document_flows" / f"{org_name.lower().replace(' ', '_')}_document_matrix.json"
        with open(doc_file, 'w') as f:
            json.dump(doc_data, f, indent=2)
        print(f"✅ Saved document flows: {doc_file}")
        
        # 4. Combined flows
        combined_matrix, combined_metadata = self.generate_combined_flows(
            email_matrix, doc_matrix, departments
        )
        
        combined_data = {
            "organization": org_name,
            "data_type": "combined_communication_flows",
            "period": "monthly_average",
            "units": "weighted_communication_intensity", 
            "description": "Combined email and document flows with weighted importance",
            "nodes": departments,
            "flows": combined_matrix.tolist(),
            "calculation_method": {
                "formula": "Combined_Flow[i,j] = (Email_Flow[i,j] * 0.6) + (Document_Flow[i,j] * 1.4)"
            },
            "metadata": combined_metadata
        }
        
        combined_file = self.output_dir / "combined_flows" / f"{org_name.lower().replace(' ', '_')}_combined_matrix.json"
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"✅ Saved combined flows: {combined_file}")
        
        print(f"\n🎯 Complete synthetic dataset generated for {org_name}")
        print(f"   Departments: {len(departments)}")
        print(f"   Total employees: {sum(employees_per_dept.values())}")
        print(f"   Total communication flows: {np.sum(combined_matrix):.1f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate synthetic organizational communication data")
    
    parser.add_argument("--org-name", default="TechFlow Innovations", 
                       help="Organization name")
    parser.add_argument("--org-type", default="Technology Company",
                       help="Organization type")
    parser.add_argument("--intensity", choices=["low", "medium", "high"], default="medium",
                       help="Communication intensity level")
    parser.add_argument("--formality", choices=["low", "medium", "high"], default="medium", 
                       help="Document formality level")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generator = OrganizationalDataGenerator(seed=args.seed)
    generator.save_organization_data(
        org_name=args.org_name,
        org_type=args.org_type,
        communication_intensity=args.intensity,
        formality_level=args.formality
    )


if __name__ == "__main__":
    main()