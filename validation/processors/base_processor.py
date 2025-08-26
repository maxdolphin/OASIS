"""
Base Dataset Processor

Abstract base class for processing real-world datasets into standardized flow matrices.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDatasetProcessor(ABC):
    """
    Abstract base class for processing real-world datasets into flow matrices.
    
    All dataset processors should inherit from this class and implement the
    abstract methods for their specific data format and domain.
    """
    
    def __init__(self, dataset_name: str, source_info: Dict[str, Any]):
        """
        Initialize the processor.
        
        Args:
            dataset_name: Name of the dataset
            source_info: Dictionary containing source metadata
        """
        self.dataset_name = dataset_name
        self.source_info = source_info
        self.logger = logging.getLogger(f"{__name__}.{dataset_name}")
        
    @abstractmethod
    def download_dataset(self) -> Optional[Path]:
        """
        Download dataset from source.
        
        Returns:
            Path to downloaded dataset or None if download fails
        """
        pass
        
    @abstractmethod
    def explore_structure(self, data_path: Path) -> Dict[str, Any]:
        """
        Analyze data structure and extract basic information.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Dictionary with structure information
        """
        pass
        
    @abstractmethod
    def extract_nodes(self, data: Any) -> List[str]:
        """
        Identify and extract network nodes from the dataset.
        
        Args:
            data: Loaded dataset
            
        Returns:
            List of node names
        """
        pass
        
    @abstractmethod
    def extract_flows(self, data: Any, nodes: List[str]) -> np.ndarray:
        """
        Create flow matrix between nodes.
        
        Args:
            data: Loaded dataset
            nodes: List of node names
            
        Returns:
            Square flow matrix (n x n)
        """
        pass
        
    def validate_matrix(self, flows: np.ndarray, nodes: List[str]) -> bool:
        """
        Validate the flow matrix for consistency.
        
        Args:
            flows: Flow matrix
            nodes: Node names
            
        Returns:
            True if matrix is valid, False otherwise
        """
        try:
            # Check dimensions
            if flows.shape[0] != flows.shape[1]:
                self.logger.error("Flow matrix is not square")
                return False
                
            if flows.shape[0] != len(nodes):
                self.logger.error("Matrix dimensions don't match number of nodes")
                return False
                
            # Check for negative values (most flow types should be non-negative)
            if np.any(flows < 0):
                self.logger.warning("Flow matrix contains negative values")
                
            # Check diagonal (should typically be zero for most flow types)
            if np.any(np.diag(flows) != 0):
                self.logger.info("Flow matrix has non-zero diagonal elements")
                
            # Check for NaN or infinite values
            if np.any(~np.isfinite(flows)):
                self.logger.error("Flow matrix contains NaN or infinite values")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating matrix: {e}")
            return False
            
    def generate_metadata(self, flows: np.ndarray, nodes: List[str], 
                         processing_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for the processed dataset.
        
        Args:
            flows: Flow matrix
            nodes: Node names  
            processing_info: Information about processing steps
            
        Returns:
            Metadata dictionary
        """
        total_flow = np.sum(flows)
        max_flow = np.max(flows)
        min_flow = np.min(flows[flows > 0])  # Minimum non-zero flow
        
        # Calculate basic network properties
        num_edges = np.sum(flows > 0)
        density = num_edges / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
        
        metadata = {
            "source": self.source_info.get("source", "Unknown"),
            "original_url": self.source_info.get("url", ""),
            "description": self.source_info.get("description", ""),
            "flow_type": self.source_info.get("type", "").lower().replace(" ", "_"),
            "units": processing_info.get("units", "Unknown units"),
            "scale": self._determine_scale(len(nodes), total_flow),
            "nodes_count": len(nodes),
            "edges_count": int(num_edges),
            "density": float(density),
            "total_flow": float(total_flow),
            "max_flow": float(max_flow),
            "min_flow": float(min_flow) if min_flow != np.inf else 0.0,
            "processed_date": datetime.now().isoformat(),
            "processor_version": "1.0",
            "processing_notes": processing_info.get("notes", ""),
            "license": processing_info.get("license", "See original source"),
            "citation": processing_info.get("citation", ""),
            "validation_status": "processed"
        }
        
        return metadata
        
    def _determine_scale(self, num_nodes: int, total_flow: float) -> str:
        """Determine the scale category based on nodes and flow."""
        if num_nodes >= 1000 or total_flow >= 1e6:
            return "massive"
        elif num_nodes >= 100 or total_flow >= 1e5:
            return "very_large"  
        elif num_nodes >= 50 or total_flow >= 1e4:
            return "large"
        elif num_nodes >= 20 or total_flow >= 1e3:
            return "medium"
        else:
            return "small"
            
    def save_processed_dataset(self, flows: np.ndarray, nodes: List[str],
                             metadata: Dict[str, Any], output_path: Path) -> bool:
        """
        Save processed dataset in standard JSON format.
        
        Args:
            flows: Flow matrix
            nodes: Node names
            metadata: Dataset metadata
            output_path: Where to save the processed dataset
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy array to list for JSON serialization
            flows_list = flows.tolist()
            
            # Create the dataset structure
            dataset = {
                "organization": self.dataset_name,
                "flows": flows_list,
                "nodes": nodes,
                "metadata": metadata
            }
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Successfully saved processed dataset to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving processed dataset: {e}")
            return False
            
    def process_dataset(self, output_dir: Path, force_download: bool = False) -> Optional[Path]:
        """
        Complete processing pipeline for the dataset.
        
        Args:
            output_dir: Directory to save processed dataset
            force_download: Whether to force re-download if data exists
            
        Returns:
            Path to processed dataset file or None if processing failed
        """
        try:
            self.logger.info(f"Starting processing of {self.dataset_name}")
            
            # Step 1: Download dataset (if implemented)
            data_path = None
            try:
                if hasattr(self, 'download_dataset'):
                    data_path = self.download_dataset()
            except NotImplementedError:
                self.logger.info("Download not implemented for this processor")
                
            # Step 2: Load and explore data structure
            if data_path and data_path.exists():
                structure_info = self.explore_structure(data_path)
                self.logger.info(f"Dataset structure: {structure_info}")
                
                # Load the data
                data = self._load_data(data_path)
            else:
                self.logger.warning("Using sample/synthetic data for processing")
                data = self._create_sample_data()
                
            # Step 3: Extract nodes
            nodes = self.extract_nodes(data)
            self.logger.info(f"Extracted {len(nodes)} nodes")
            
            # Step 4: Extract flows
            flows = self.extract_flows(data, nodes)
            self.logger.info(f"Created {flows.shape} flow matrix")
            
            # Step 5: Validate matrix
            if not self.validate_matrix(flows, nodes):
                self.logger.error("Matrix validation failed")
                return None
                
            # Step 6: Generate metadata
            processing_info = {
                "units": self._get_flow_units(),
                "notes": self._get_processing_notes(),
                "license": self.source_info.get("license", "See original source")
            }
            metadata = self.generate_metadata(flows, nodes, processing_info)
            
            # Step 7: Save processed dataset
            output_file = output_dir / f"{self.dataset_name.lower().replace(' ', '_')}.json"
            if self.save_processed_dataset(flows, nodes, metadata, output_file):
                self.logger.info(f"Successfully processed {self.dataset_name}")
                return output_file
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing {self.dataset_name}: {e}")
            return None
            
    def _load_data(self, data_path: Path) -> Any:
        """Load data from file. Override for specific formats."""
        if data_path.suffix.lower() == '.csv':
            return pd.read_csv(data_path)
        elif data_path.suffix.lower() == '.json':
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
            
    def _create_sample_data(self) -> Any:
        """Create sample data for testing. Override in subclasses."""
        raise NotImplementedError("Sample data creation not implemented")
        
    def _get_flow_units(self) -> str:
        """Get the units for flows. Override in subclasses."""
        return "Unknown units"
        
    def _get_processing_notes(self) -> str:
        """Get processing notes. Override in subclasses."""
        return f"Processed using {self.__class__.__name__}"