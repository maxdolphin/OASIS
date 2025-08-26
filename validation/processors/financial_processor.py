"""
Financial Flow Dataset Processor

Specialized processor for financial flow datasets including transaction networks and payment systems.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .base_processor import BaseDatasetProcessor


class FinancialFlowProcessor(BaseDatasetProcessor):
    """Processor for financial flow datasets."""
    
    def __init__(self, dataset_name: str, source_info: Dict[str, Any]):
        super().__init__(dataset_name, source_info)
        
    def download_dataset(self) -> Optional[Path]:
        """
        Download financial dataset. 
        Note: For now, this creates sample data until Kaggle API is integrated.
        """
        # TODO: Implement Kaggle API download
        self.logger.info("Creating sample financial transaction data for demonstration")
        return None
        
    def explore_structure(self, data_path: Path) -> Dict[str, Any]:
        """Explore financial dataset structure."""
        if data_path and data_path.exists():
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
                return {
                    "file_type": "CSV",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample_data": df.head(3).to_dict()
                }
        return {"status": "using_sample_data"}
        
    def extract_nodes(self, data: Any) -> List[str]:
        """Extract financial network nodes (accounts, institutions)."""
        if isinstance(data, pd.DataFrame):
            # Look for common financial node columns
            node_columns = ['account', 'customer', 'sender', 'receiver', 'origin', 'destination']
            
            nodes = set()
            for col in node_columns:
                if col in data.columns:
                    # Sample a subset for performance with large datasets
                    sample_data = data[col].dropna().unique()
                    if len(sample_data) > 1000:  # Limit nodes for performance
                        sample_data = np.random.choice(sample_data, 1000, replace=False)
                    nodes.update(sample_data)
                    
                # Also check ID columns
                for suffix in ['_id', '_account', '_number']:
                    id_col = f'{col}{suffix}'
                    if id_col in data.columns:
                        sample_data = data[id_col].dropna().unique()
                        if len(sample_data) > 1000:
                            sample_data = np.random.choice(sample_data, 1000, replace=False)
                        nodes.update(sample_data)
                        
            if nodes:
                # Convert to strings and sort
                return sorted([str(node) for node in list(nodes)])
                
        # Create sample financial nodes
        return self._create_sample_financial_nodes()
        
    def extract_flows(self, data: Any, nodes: List[str]) -> np.ndarray:
        """Extract financial flows between nodes."""
        n = len(nodes)
        flows = np.zeros((n, n))
        
        if isinstance(data, pd.DataFrame):
            # Process real financial transaction data
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Look for amount columns
            amount_cols = ['amount', 'value', 'transaction_amount', 'sum', 'total']
            amount_col = None
            for col in amount_cols:
                if col in data.columns:
                    amount_col = col
                    break
                    
            # Look for sender/receiver columns
            sender_col = None
            receiver_col = None
            
            for col in ['sender', 'from', 'origin', 'payer']:
                if col in data.columns or f'{col}_id' in data.columns:
                    sender_col = col if col in data.columns else f'{col}_id'
                    break
                    
            for col in ['receiver', 'to', 'destination', 'payee']:
                if col in data.columns or f'{col}_id' in data.columns:
                    receiver_col = col if col in data.columns else f'{col}_id'
                    break
                    
            if amount_col and sender_col and receiver_col:
                # Process transactions in batches for performance
                batch_size = 10000
                for start_idx in range(0, len(data), batch_size):
                    batch = data.iloc[start_idx:start_idx + batch_size]
                    
                    for _, row in batch.iterrows():
                        sender = str(row.get(sender_col, ''))
                        receiver = str(row.get(receiver_col, ''))
                        amount = row.get(amount_col, 0)
                        
                        if sender in node_to_idx and receiver in node_to_idx and amount > 0:
                            i, j = node_to_idx[sender], node_to_idx[receiver]
                            flows[i, j] += float(amount)
        else:
            # Create sample financial flows
            flows = self._create_sample_financial_flows(nodes)
            
        return flows
        
    def _create_sample_financial_nodes(self) -> List[str]:
        """Create sample financial network nodes."""
        return [
            "Central_Bank",
            "Commercial_Bank_Alpha", 
            "Commercial_Bank_Beta",
            "Investment_Bank_Gamma",
            "Credit_Union_Delta",
            "Payment_Processor_PayFlow",
            "Payment_Processor_QuickPay",
            "Mobile_Wallet_Service",
            "Corporate_Account_TechCorp",
            "Corporate_Account_RetailCorp",
            "SME_Account_LocalBusiness",
            "SME_Account_StartupInc",
            "Individual_Account_HighNet",
            "Individual_Account_Standard",
            "ATM_Network_Urban",
            "ATM_Network_Rural"
        ]
        
    def _create_sample_financial_flows(self, nodes: List[str]) -> np.ndarray:
        """Create realistic sample financial flows."""
        n = len(nodes)
        flows = np.zeros((n, n))
        
        # Set random seed for reproducibility
        np.random.seed(456)
        
        # Create flows based on realistic financial patterns
        central_banks = ["Central_Bank"]
        commercial_banks = ["Commercial_Bank_Alpha", "Commercial_Bank_Beta", "Investment_Bank_Gamma", "Credit_Union_Delta"]
        payment_processors = ["Payment_Processor_PayFlow", "Payment_Processor_QuickPay", "Mobile_Wallet_Service"]
        corporate_accounts = ["Corporate_Account_TechCorp", "Corporate_Account_RetailCorp"]
        sme_accounts = ["SME_Account_LocalBusiness", "SME_Account_StartupInc"]
        individual_accounts = ["Individual_Account_HighNet", "Individual_Account_Standard"]
        atm_networks = ["ATM_Network_Urban", "ATM_Network_Rural"]
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Central Bank → Commercial Banks (monetary policy)
        for cb in central_banks:
            if cb in node_to_idx:
                cb_idx = node_to_idx[cb]
                for bank in commercial_banks:
                    if bank in node_to_idx:
                        bank_idx = node_to_idx[bank]
                        # Large interbank flows: $10M - $100M
                        flows[cb_idx, bank_idx] = np.random.uniform(10_000_000, 100_000_000)
                        
        # Commercial Banks ↔ Payment Processors (clearing)
        for bank in commercial_banks:
            if bank in node_to_idx:
                bank_idx = node_to_idx[bank]
                for processor in payment_processors:
                    if processor in node_to_idx:
                        proc_idx = node_to_idx[processor]
                        # Payment clearing: $1M - $50M
                        flows[bank_idx, proc_idx] = np.random.uniform(1_000_000, 50_000_000)
                        flows[proc_idx, bank_idx] = np.random.uniform(1_000_000, 50_000_000)
                        
        # Corporate Accounts → Banks (deposits, payments)
        for corp in corporate_accounts:
            if corp in node_to_idx:
                corp_idx = node_to_idx[corp]
                for bank in commercial_banks[:2]:  # Primary banking relationships
                    if bank in node_to_idx:
                        bank_idx = node_to_idx[bank]
                        # Corporate flows: $500K - $10M
                        flows[corp_idx, bank_idx] = np.random.uniform(500_000, 10_000_000)
                        flows[bank_idx, corp_idx] = np.random.uniform(200_000, 5_000_000)
                        
        # SME Accounts ↔ Banks
        for sme in sme_accounts:
            if sme in node_to_idx:
                sme_idx = node_to_idx[sme]
                for bank in commercial_banks:
                    if bank in node_to_idx:
                        bank_idx = node_to_idx[bank]
                        # SME flows: $10K - $500K
                        flows[sme_idx, bank_idx] = np.random.uniform(10_000, 500_000)
                        flows[bank_idx, sme_idx] = np.random.uniform(5_000, 200_000)
                        
        # Individual Accounts ↔ Banks
        for individual in individual_accounts:
            if individual in node_to_idx:
                ind_idx = node_to_idx[individual]
                for bank in commercial_banks:
                    if bank in node_to_idx:
                        bank_idx = node_to_idx[bank]
                        # Individual flows: $1K - $100K
                        base_amount = 50_000 if "HighNet" in individual else 10_000
                        flows[ind_idx, bank_idx] = np.random.uniform(1_000, base_amount)
                        flows[bank_idx, ind_idx] = np.random.uniform(500, base_amount // 2)
                        
        # Payment Processors → Accounts (transaction processing)
        all_accounts = corporate_accounts + sme_accounts + individual_accounts
        for processor in payment_processors:
            if processor in node_to_idx:
                proc_idx = node_to_idx[processor]
                for account in all_accounts:
                    if account in node_to_idx:
                        acc_idx = node_to_idx[account]
                        # Payment processing: $100 - $50K
                        flows[proc_idx, acc_idx] = np.random.uniform(100, 50_000)
                        flows[acc_idx, proc_idx] = np.random.uniform(100, 50_000)
                        
        # ATM Networks ↔ Banks
        for atm in atm_networks:
            if atm in node_to_idx:
                atm_idx = node_to_idx[atm]
                for bank in commercial_banks:
                    if bank in node_to_idx:
                        bank_idx = node_to_idx[bank]
                        # ATM cash flows: $100K - $5M
                        flows[bank_idx, atm_idx] = np.random.uniform(100_000, 5_000_000)
                        flows[atm_idx, bank_idx] = np.random.uniform(50_000, 1_000_000)
                
        return flows
        
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample financial transaction data structure."""
        nodes = self._create_sample_financial_nodes()
        
        # Create sample data structure that mimics real financial datasets
        data_rows = []
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_IN', 'CASH_OUT', 'DEBIT']
        
        for _ in range(10000):  # 10K sample transactions
            sender = np.random.choice(nodes)
            receiver = np.random.choice([n for n in nodes if n != sender])
            
            data_rows.append({
                'sender': sender,
                'receiver': receiver,
                'amount': np.random.exponential(1000),  # Realistic amount distribution
                'type': np.random.choice(transaction_types),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
                'currency': 'USD'
            })
                    
        return pd.DataFrame(data_rows)
        
    def _get_flow_units(self) -> str:
        """Get financial flow units."""
        return "USD (US Dollars)"
        
    def _get_processing_notes(self) -> str:
        """Get financial-specific processing notes."""
        return ("Processed financial transaction network with banks, payment processors, "
                "and account holders. Flows represent monetary transfers between entities. "
                "Large datasets are sampled for performance optimization.")


class PaySimFinancialProcessor(FinancialFlowProcessor):
    """Specific processor for PaySim Financial Dataset."""
    
    def __init__(self):
        source_info = {
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
            "description": "Large-scale synthetic mobile money transaction flows with sender-receiver relationships",
            "type": "Financial Flow",
            "license": "Dataset-specific license"
        }
        super().__init__("PaySim Mobile Money Network", source_info)


class BankingTransactionsProcessor(FinancialFlowProcessor):
    """Specific processor for Banking Transactions dataset."""
    
    def __init__(self):
        source_info = {
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets",
            "description": "Recent financial transactions dataset designed for analytics and AI-powered banking solutions", 
            "type": "Financial Flow",
            "license": "Dataset-specific license"
        }
        super().__init__("Banking Transaction Network", source_info)