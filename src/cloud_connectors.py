"""
Cloud Service Connectors for Adaptive Organization Analysis
============================================================

Proof of concept for integrating with enterprise cloud services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class BaseConnector(ABC):
    """Abstract base class for all cloud service connectors."""
    
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with the cloud service."""
        pass
    
    @abstractmethod
    def get_organization_structure(self) -> Dict[str, Any]:
        """Extract organizational structure (nodes/departments)."""
        pass
    
    @abstractmethod
    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Extract flow matrix for the specified period."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get service-specific metadata."""
        pass


class GoogleWorkspaceConnector(BaseConnector):
    """
    Google Workspace connector for extracting organizational flows.
    
    Data sources:
    - Gmail: Email communication patterns
    - Drive: Document collaboration
    - Calendar: Meeting patterns
    - Admin SDK: Organizational structure
    """
    
    def __init__(self):
        self.service = None
        self.domain = None
        
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate using Google OAuth 2.0.
        
        Required credentials:
        - client_id
        - client_secret
        - refresh_token (or auth flow)
        """
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            
            # In production, use proper OAuth flow
            # This is simplified for POC
            creds = Credentials(
                token=credentials.get('access_token'),
                refresh_token=credentials.get('refresh_token'),
                token_uri='https://oauth2.googleapis.com/token',
                client_id=credentials['client_id'],
                client_secret=credentials['client_secret']
            )
            
            # Build services
            self.admin_service = build('admin', 'directory_v1', credentials=creds)
            self.reports_service = build('admin', 'reports_v1', credentials=creds)
            self.drive_service = build('drive', 'v3', credentials=creds)
            
            self.domain = credentials.get('domain')
            return True
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def get_organization_structure(self) -> Dict[str, Any]:
        """Extract org structure from Google Admin SDK."""
        if not self.admin_service:
            return {}
        
        try:
            # Get organizational units
            org_units = self.admin_service.orgunits().list(
                customerId='my_customer'
            ).execute()
            
            # Get users
            users = self.admin_service.users().list(
                customer='my_customer',
                maxResults=500
            ).execute()
            
            # Build node list from departments
            departments = set()
            user_dept_map = {}
            
            for user in users.get('users', []):
                dept = user.get('orgUnitPath', '/').split('/')[-1] or 'Root'
                departments.add(dept)
                user_dept_map[user['primaryEmail']] = dept
            
            return {
                'nodes': list(departments),
                'user_mapping': user_dept_map,
                'total_users': len(users.get('users', []))
            }
            
        except Exception as e:
            print(f"Error getting org structure: {e}")
            return {}
    
    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Extract communication flows from Google Workspace."""
        org_data = self.get_organization_structure()
        nodes = org_data.get('nodes', [])
        user_mapping = org_data.get('user_mapping', {})
        
        if not nodes:
            return np.array([[]])
        
        # Initialize flow matrix
        n = len(nodes)
        flow_matrix = np.zeros((n, n))
        node_index = {node: i for i, node in enumerate(nodes)}
        
        try:
            # Get email activities from Reports API
            activities = self.reports_service.activities().list(
                userKey='all',
                applicationName='gmail',
                startTime=start_date.isoformat() + 'Z',
                endTime=end_date.isoformat() + 'Z',
                maxResults=1000
            ).execute()
            
            # Process email flows
            for activity in activities.get('items', []):
                actor = activity['actor']['email']
                
                for event in activity.get('events', []):
                    if event['type'] == 'message_sent':
                        # Extract recipient from parameters
                        for param in event.get('parameters', []):
                            if param['name'] == 'destination':
                                recipient = param['value']
                                
                                # Map to departments
                                from_dept = user_mapping.get(actor)
                                to_dept = user_mapping.get(recipient)
                                
                                if from_dept and to_dept and from_dept in node_index and to_dept in node_index:
                                    flow_matrix[node_index[from_dept]][node_index[to_dept]] += 1
            
            # Get Drive collaboration data
            drive_activities = self.drive_service.activities().query(
                body={
                    'startTime': start_date.isoformat() + 'Z',
                    'endTime': end_date.isoformat() + 'Z'
                }
            ).execute()
            
            # Process collaboration flows
            # (simplified - would need more sophisticated processing)
            
            return flow_matrix
            
        except Exception as e:
            print(f"Error extracting flows: {e}")
            return flow_matrix
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Google Workspace metadata."""
        return {
            'connector': 'Google Workspace',
            'domain': self.domain,
            'data_sources': ['Gmail', 'Drive', 'Calendar', 'Admin SDK'],
            'capabilities': [
                'email_flows',
                'document_collaboration',
                'meeting_patterns',
                'org_structure'
            ]
        }


class MicrosoftGraphConnector(BaseConnector):
    """
    Microsoft Graph connector for Office 365 and Azure AD.
    
    Data sources:
    - Outlook: Email patterns
    - Teams: Collaboration flows
    - SharePoint: Document flows
    - Azure AD: Organization structure
    """
    
    def __init__(self):
        self.graph_client = None
        self.tenant_id = None
        
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate using Microsoft Azure AD OAuth."""
        try:
            from msal import ConfidentialClientApplication
            
            app = ConfidentialClientApplication(
                credentials['client_id'],
                authority=f"https://login.microsoftonline.com/{credentials['tenant_id']}",
                client_credential=credentials['client_secret']
            )
            
            result = app.acquire_token_silent(
                ["https://graph.microsoft.com/.default"],
                account=None
            )
            
            if not result:
                result = app.acquire_token_for_client(
                    scopes=["https://graph.microsoft.com/.default"]
                )
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                self.tenant_id = credentials['tenant_id']
                return True
                
        except Exception as e:
            print(f"Authentication failed: {e}")
            
        return False
    
    def get_organization_structure(self) -> Dict[str, Any]:
        """Extract organization from Azure AD."""
        import requests
        
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        # Get groups (departments)
        response = requests.get(
            'https://graph.microsoft.com/v1.0/groups',
            headers=headers
        )
        
        if response.status_code == 200:
            groups = response.json().get('value', [])
            nodes = [group['displayName'] for group in groups]
            
            # Get users and their group memberships
            users_response = requests.get(
                'https://graph.microsoft.com/v1.0/users?$expand=memberOf',
                headers=headers
            )
            
            return {
                'nodes': nodes,
                'groups': groups,
                'users': users_response.json().get('value', [])
            }
        
        return {}
    
    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Extract flows from Microsoft Graph."""
        # Implementation would query:
        # - Email flows from Exchange
        # - Teams channel messages
        # - SharePoint collaboration
        # - Meeting patterns from Calendar
        
        # Simplified POC
        org_data = self.get_organization_structure()
        n = len(org_data.get('nodes', []))
        
        # Would process actual data here
        return np.random.rand(n, n) * 100  # Placeholder
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Microsoft Graph metadata."""
        return {
            'connector': 'Microsoft Graph',
            'tenant_id': self.tenant_id,
            'data_sources': ['Exchange', 'Teams', 'SharePoint', 'Azure AD'],
            'capabilities': [
                'email_flows',
                'teams_collaboration',
                'document_sharing',
                'meeting_analytics'
            ]
        }


class SalesforceConnector(BaseConnector):
    """
    Salesforce connector for CRM and business process flows.
    
    Data sources:
    - Opportunities: Sales pipeline flows
    - Cases: Support ticket flows
    - Leads: Marketing flows
    - Custom Objects: Business-specific flows
    """
    
    def __init__(self):
        self.sf = None
        self.instance_url = None
        
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate using Salesforce OAuth."""
        try:
            from simple_salesforce import Salesforce
            
            self.sf = Salesforce(
                username=credentials['username'],
                password=credentials['password'],
                security_token=credentials['security_token'],
                domain=credentials.get('domain', 'login')
            )
            
            self.instance_url = self.sf.sf_instance
            return True
            
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def get_organization_structure(self) -> Dict[str, Any]:
        """Extract organization from Salesforce roles and users."""
        if not self.sf:
            return {}
        
        # Query roles (departments)
        roles = self.sf.query("""
            SELECT Id, Name, ParentRoleId 
            FROM UserRole
        """)
        
        # Query users
        users = self.sf.query("""
            SELECT Id, Name, Email, UserRoleId, Department 
            FROM User 
            WHERE IsActive = true
        """)
        
        # Build department list
        departments = set()
        for user in users['records']:
            dept = user.get('Department') or 'Unknown'
            departments.add(dept)
        
        return {
            'nodes': list(departments),
            'roles': roles['records'],
            'users': users['records']
        }
    
    def get_flow_data(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Extract business process flows from Salesforce."""
        org_data = self.get_organization_structure()
        nodes = org_data.get('nodes', [])
        
        if not nodes:
            return np.array([[]])
        
        n = len(nodes)
        flow_matrix = np.zeros((n, n))
        node_index = {node: i for i, node in enumerate(nodes)}
        
        # Query opportunity flows
        opps = self.sf.query(f"""
            SELECT Id, Name, Amount, StageName, 
                   CreatedById, OwnerId, LastModifiedById
            FROM Opportunity
            WHERE CreatedDate >= {start_date.isoformat()}
            AND CreatedDate <= {end_date.isoformat()}
        """)
        
        # Process opportunity handoffs as flows
        # (simplified - would need user->department mapping)
        
        # Query case flows
        cases = self.sf.query(f"""
            SELECT Id, CaseNumber, Status,
                   CreatedById, OwnerId, LastModifiedById
            FROM Case
            WHERE CreatedDate >= {start_date.isoformat()}
            AND CreatedDate <= {end_date.isoformat()}
        """)
        
        # Process case routing as flows
        
        return flow_matrix
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get Salesforce metadata."""
        return {
            'connector': 'Salesforce',
            'instance': self.instance_url,
            'data_sources': ['Opportunities', 'Cases', 'Leads', 'Activities'],
            'capabilities': [
                'sales_pipeline',
                'support_flows',
                'marketing_flows',
                'custom_processes'
            ]
        }


class ConnectorFactory:
    """Factory for creating appropriate connectors."""
    
    @staticmethod
    def create_connector(service_type: str) -> Optional[BaseConnector]:
        """Create a connector based on service type."""
        connectors = {
            'google': GoogleWorkspaceConnector,
            'microsoft': MicrosoftGraphConnector,
            'salesforce': SalesforceConnector
        }
        
        connector_class = connectors.get(service_type.lower())
        if connector_class:
            return connector_class()
        
        return None


class MultiSourceAggregator:
    """Aggregates data from multiple cloud sources."""
    
    def __init__(self):
        self.connectors = []
        
    def add_connector(self, connector: BaseConnector):
        """Add a connector to the aggregator."""
        self.connectors.append(connector)
        
    def get_combined_flow_matrix(self, start_date: datetime, end_date: datetime) -> np.ndarray:
        """Combine flow matrices from all connectors."""
        if not self.connectors:
            return np.array([[]])
        
        # Get the union of all nodes
        all_nodes = set()
        for connector in self.connectors:
            org_data = connector.get_organization_structure()
            all_nodes.update(org_data.get('nodes', []))
        
        nodes = list(all_nodes)
        n = len(nodes)
        combined_matrix = np.zeros((n, n))
        
        # Aggregate flows from each connector
        for connector in self.connectors:
            flow_matrix = connector.get_flow_data(start_date, end_date)
            # Map and add to combined matrix
            # (would need proper node mapping logic)
            
        return combined_matrix
    
    def get_combined_metadata(self) -> Dict[str, Any]:
        """Get combined metadata from all connectors."""
        return {
            'num_sources': len(self.connectors),
            'sources': [c.get_metadata() for c in self.connectors]
        }