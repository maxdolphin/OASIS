"""
HuggingFace Dataset Discovery Agent

Automatically discovers, evaluates, and queues HuggingFace datasets that can be
converted to flow network matrices for organizational/ecosystem analysis.

Workflow:
    Search HuggingFace -> Evaluate Datasets -> Score -> Present Candidates
    -> User Approval -> Processing Pipeline
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from datasets import load_dataset, get_dataset_config_names
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Keyword Taxonomy for Dataset Discovery
# =============================================================================

KEYWORD_TAXONOMY = {
    'network_structure': {
        'keywords': ['graph', 'network', 'edge', 'node', 'adjacency', 'connectivity'],
        'weight': 1.0,
        'description': 'Explicit graph/network structure datasets'
    },
    'supply_chain': {
        'keywords': ['supply chain', 'logistics', 'inventory', 'warehouse', 'distribution', 'shipping'],
        'weight': 0.95,
        'description': 'Supply chain and logistics flow datasets'
    },
    'transactions': {
        'keywords': ['transaction', 'payment', 'transfer', 'trade', 'exchange', 'financial flow'],
        'weight': 0.90,
        'description': 'Financial transaction and payment flow datasets'
    },
    'transportation': {
        'keywords': ['traffic', 'mobility', 'route', 'flight', 'airport', 'OD matrix', 'origin destination'],
        'weight': 0.90,
        'description': 'Transportation and mobility flow datasets'
    },
    'energy': {
        'keywords': ['power grid', 'energy flow', 'electricity', 'transmission', 'smart grid'],
        'weight': 0.85,
        'description': 'Energy and power grid flow datasets'
    },
    'communication': {
        'keywords': ['network traffic', 'data flow', 'email', 'messaging', 'social network'],
        'weight': 0.85,
        'description': 'Communication and data flow datasets'
    },
    'biological': {
        'keywords': ['protein', 'gene', 'metabolic', 'pathway', 'regulatory network', 'PPI'],
        'weight': 0.80,
        'description': 'Biological network datasets (protein, gene, metabolic)'
    },
    'ecological': {
        'keywords': ['food web', 'ecosystem', 'trophic', 'carbon flow', 'nutrient cycle'],
        'weight': 0.80,
        'description': 'Ecological and environmental flow datasets'
    },
    'trade': {
        'keywords': ['import', 'export', 'bilateral trade', 'commodity flow', 'input-output'],
        'weight': 0.85,
        'description': 'International trade and economic flow datasets'
    }
}

NEGATIVE_KEYWORDS = [
    'image classification', 'text generation', 'sentiment analysis',
    'object detection', 'speech recognition', 'language model',
    'question answering', 'summarization', 'NER', 'chatbot',
    'image captioning', 'text-to-image', 'translation'
]

# License scoring
LICENSE_SCORES = {
    'mit': 15,
    'apache-2.0': 15,
    'cc0-1.0': 15,
    'cc-by-4.0': 12,
    'cc-by-sa-4.0': 10,
    'cc-by-nc-4.0': 8,
    'cc-by-nc-sa-4.0': 8,
    'openrail': 10,
    'unknown': 5,
    'other': 5
}


class DatasetScorer:
    """Scores datasets for flow network conversion potential."""

    # Structure indicators (columns/fields that suggest network data)
    STRUCTURE_INDICATORS = {
        'high': ['edge_index', 'adjacency', 'edge_list', 'source_target'],
        'medium': ['source', 'target', 'from', 'to', 'origin', 'destination'],
        'low': ['node', 'vertex', 'link', 'connection']
    }

    def __init__(self):
        self.taxonomy = KEYWORD_TAXONOMY
        self.negative_keywords = NEGATIVE_KEYWORDS

    def calculate_flow_potential_score(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the flow network potential score for a dataset.

        Args:
            metadata: Dataset metadata from HuggingFace

        Returns:
            Dictionary with score breakdown and recommendations
        """
        scores = {
            'structure_score': self._score_structure(metadata),
            'size_score': self._score_size(metadata),
            'quality_score': self._score_quality(metadata),
            'license_score': self._score_license(metadata),
            'feasibility_score': self._score_feasibility(metadata)
        }

        total_score = sum(scores.values())

        # Determine recommendation
        if total_score >= 70:
            recommendation = 'high'
        elif total_score >= 50:
            recommendation = 'medium'
        elif total_score >= 30:
            recommendation = 'low'
        else:
            recommendation = 'skip'

        # Determine conversion complexity
        complexity = self._assess_complexity(metadata)

        return {
            'total_score': total_score,
            **scores,
            'recommendation': recommendation,
            'conversion_complexity': complexity
        }

    def _score_structure(self, metadata: Dict) -> float:
        """Score based on data structure (max 35 points)."""
        score = 0.0

        description = (metadata.get('description') or '').lower()
        tags = [t.lower() for t in (metadata.get('tags') or [])]
        card_data = metadata.get('cardData') or {}

        # Check for explicit structure indicators in description
        for keyword in self.STRUCTURE_INDICATORS['high']:
            if keyword in description:
                score += 15
                break

        for keyword in self.STRUCTURE_INDICATORS['medium']:
            if keyword in description:
                score += 10
                break

        # Check tags for graph-related content
        graph_tags = ['graph', 'network', 'graph-ml', 'graph-neural-network']
        for tag in tags:
            if any(gt in tag for gt in graph_tags):
                score += 10
                break

        # Check for tabular/structured format
        if 'tabular' in tags or 'csv' in str(card_data.get('format', '')).lower():
            score += 5

        return min(35.0, score)

    def _score_size(self, metadata: Dict) -> float:
        """Score based on dataset size (max 20 points)."""
        score = 20.0  # Start with max, deduct for issues

        # Get size information
        download_size = metadata.get('downloadSize') or metadata.get('size_bytes') or 0
        num_rows = metadata.get('num_rows') or 0

        # Ideal size: 1K-100K rows, <1GB download
        if download_size and download_size > 0:
            size_gb = download_size / (1024 ** 3)
            if size_gb > 10:
                score -= 15  # Too large
            elif size_gb > 1:
                score -= 5   # Large but manageable

        # Row count scoring
        if num_rows and num_rows > 0:
            if num_rows < 100:
                score -= 10  # Too small
            elif num_rows > 10_000_000:
                score -= 10  # Too large
            elif num_rows > 1_000_000:
                score -= 3   # Large

        return max(0.0, score)

    def _score_quality(self, metadata: Dict) -> float:
        """Score based on dataset quality indicators (max 20 points)."""
        score = 0.0

        # Has description
        if metadata.get('description'):
            desc_len = len(metadata['description'])
            if desc_len > 500:
                score += 8
            elif desc_len > 100:
                score += 5
            else:
                score += 2

        # Popularity metrics
        downloads = metadata.get('downloads') or 0
        likes = metadata.get('likes') or 0

        if downloads > 10000:
            score += 5
        elif downloads > 1000:
            score += 3
        elif downloads > 100:
            score += 1

        if likes > 100:
            score += 4
        elif likes > 10:
            score += 2
        elif likes > 0:
            score += 1

        # Has documentation/card
        if metadata.get('cardData'):
            score += 3

        return min(20.0, score)

    def _score_license(self, metadata: Dict) -> float:
        """Score based on license (max 15 points)."""
        license_id = (metadata.get('license') or 'unknown').lower()

        # Direct match
        if license_id in LICENSE_SCORES:
            return LICENSE_SCORES[license_id]

        # Partial match
        for key, score in LICENSE_SCORES.items():
            if key in license_id:
                return score

        return LICENSE_SCORES['unknown']

    def _score_feasibility(self, metadata: Dict) -> float:
        """Score based on feasibility of extraction (max 10 points)."""
        score = 10.0  # Start with max, deduct for issues

        # Check if gated
        if metadata.get('gated'):
            score -= 5

        # Check if private
        if metadata.get('private'):
            score -= 10

        # Check negative keywords (not useful for flow networks)
        description = (metadata.get('description') or '').lower()
        tags = ' '.join(metadata.get('tags') or []).lower()
        combined_text = description + ' ' + tags

        for neg_keyword in NEGATIVE_KEYWORDS:
            if neg_keyword in combined_text:
                score -= 3
                break

        return max(0.0, score)

    def _assess_complexity(self, metadata: Dict) -> str:
        """Assess the complexity of converting this dataset to a flow network."""
        description = (metadata.get('description') or '').lower()
        tags = [t.lower() for t in (metadata.get('tags') or [])]

        # Direct conversion indicators
        if any(kw in description for kw in ['edge_index', 'adjacency matrix', 'edge list']):
            return 'direct'

        # Moderate complexity indicators
        if any(kw in description for kw in ['source', 'target', 'from', 'to', 'origin', 'destination']):
            return 'moderate'

        # Check tags
        if 'graph' in tags or 'network' in tags:
            return 'moderate'

        return 'complex'


class HuggingFaceDiscoveryAgent:
    """
    Agent for discovering and evaluating HuggingFace datasets
    that can be converted to flow networks.
    """

    def __init__(self, db_manager=None):
        """
        Initialize the discovery agent.

        Args:
            db_manager: DatabaseManager instance for storing discovered datasets
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets is required. Install with: pip install datasets")

        self.api = HfApi()
        self.scorer = DatasetScorer()
        self.db_manager = db_manager
        self.taxonomy = KEYWORD_TAXONOMY

    def run_discovery(self,
                      categories: List[str] = None,
                      max_per_category: int = 50,
                      min_score: float = 30) -> Dict[str, Any]:
        """
        Run discovery across specified categories.

        Args:
            categories: List of category names from taxonomy (None = all categories)
            max_per_category: Maximum datasets to fetch per category
            min_score: Minimum score threshold for storing

        Returns:
            Dictionary with discovery results
        """
        if categories is None:
            categories = list(self.taxonomy.keys())

        results = {
            'started_at': datetime.now().isoformat(),
            'categories_searched': categories,
            'datasets_found': {},
            'total_found': 0,
            'high_potential': 0,
            'medium_potential': 0,
            'errors': []
        }

        run_id = None
        if self.db_manager:
            run_id = self._start_discovery_run(categories)

        for category in categories:
            if category not in self.taxonomy:
                results['errors'].append(f"Unknown category: {category}")
                continue

            logger.info(f"Searching category: {category}")

            try:
                category_results = self._search_category(
                    category,
                    max_per_category,
                    min_score
                )
                results['datasets_found'][category] = category_results
                results['total_found'] += category_results['count']
                results['high_potential'] += category_results['high_count']
                results['medium_potential'] += category_results['medium_count']

            except Exception as e:
                logger.error(f"Error searching {category}: {e}")
                results['errors'].append(f"{category}: {str(e)}")

        results['completed_at'] = datetime.now().isoformat()

        if self.db_manager and run_id:
            self._complete_discovery_run(run_id, results)

        return results

    def _search_category(self,
                         category: str,
                         max_results: int,
                         min_score: float) -> Dict[str, Any]:
        """
        Search for datasets in a specific category.

        Args:
            category: Category name from taxonomy
            max_results: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            Dictionary with category search results
        """
        category_info = self.taxonomy[category]
        keywords = category_info['keywords']
        category_weight = category_info['weight']

        datasets_found = []
        seen_ids = set()

        for keyword in keywords:
            try:
                # Search HuggingFace Hub
                search_results = self.api.list_datasets(
                    search=keyword,
                    limit=max_results,
                    sort='downloads',
                    direction=-1
                )

                for ds in search_results:
                    if ds.id in seen_ids:
                        continue
                    seen_ids.add(ds.id)

                    # Skip if already in database
                    if self.db_manager and self._dataset_exists(ds.id):
                        continue

                    # Get metadata
                    metadata = self._get_dataset_metadata(ds)

                    # Calculate score
                    score_result = self.scorer.calculate_flow_potential_score(metadata)

                    # Apply category weight
                    weighted_score = score_result['total_score'] * category_weight

                    if weighted_score >= min_score:
                        dataset_info = {
                            'hf_id': ds.id,
                            'hf_author': ds.author if hasattr(ds, 'author') else ds.id.split('/')[0] if '/' in ds.id else None,
                            'name': ds.id.split('/')[-1],
                            'metadata': metadata,
                            'score': score_result,
                            'weighted_score': weighted_score,
                            'discovery_keyword': keyword,
                            'discovery_category': category
                        }
                        datasets_found.append(dataset_info)

                        # Store in database
                        if self.db_manager:
                            self._store_dataset(dataset_info, category, keyword)

            except Exception as e:
                logger.warning(f"Error searching keyword '{keyword}': {e}")
                continue

        # Sort by score
        datasets_found.sort(key=lambda x: x['weighted_score'], reverse=True)

        # Count by recommendation
        high_count = sum(1 for d in datasets_found if d['score']['recommendation'] == 'high')
        medium_count = sum(1 for d in datasets_found if d['score']['recommendation'] == 'medium')

        return {
            'count': len(datasets_found),
            'high_count': high_count,
            'medium_count': medium_count,
            'datasets': datasets_found[:max_results]
        }

    def _get_dataset_metadata(self, ds) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a dataset.

        Args:
            ds: Dataset info object from HuggingFace

        Returns:
            Dictionary with metadata
        """
        metadata = {
            'description': getattr(ds, 'description', None),
            'tags': getattr(ds, 'tags', []),
            'downloads': getattr(ds, 'downloads', 0),
            'likes': getattr(ds, 'likes', 0),
            'license': getattr(ds, 'license', None),
            'gated': getattr(ds, 'gated', False),
            'private': getattr(ds, 'private', False),
            'cardData': getattr(ds, 'cardData', None),
            'lastModified': getattr(ds, 'lastModified', None),
            'downloadSize': getattr(ds, 'downloadSize', None),
            'num_rows': None  # Would need to load dataset to get this
        }

        # Try to get additional info from card data
        if hasattr(ds, 'card_data') and ds.card_data:
            card = ds.card_data
            if hasattr(card, 'dataset_info'):
                info = card.dataset_info
                if isinstance(info, dict):
                    metadata['num_rows'] = info.get('splits', {}).get('train', {}).get('num_examples')

        return metadata

    def _dataset_exists(self, hf_id: str) -> bool:
        """Check if dataset already exists in database."""
        if not self.db_manager:
            return False

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM discovered_datasets WHERE hf_id = ?', (hf_id,))
        return cursor.fetchone() is not None

    def _store_dataset(self, dataset_info: Dict, category: str, keyword: str) -> None:
        """Store discovered dataset in database."""
        if not self.db_manager:
            return

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        metadata = dataset_info['metadata']
        score = dataset_info['score']

        cursor.execute('''
            INSERT OR REPLACE INTO discovered_datasets (
                hf_id, hf_author, name, description, tags, license,
                num_rows, download_size_bytes,
                discovered_at, discovery_keywords, discovery_category,
                total_score, structure_score, size_score, quality_score,
                license_score, feasibility_score, recommendation, conversion_complexity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_info['hf_id'],
            dataset_info['hf_author'],
            dataset_info['name'],
            metadata.get('description'),
            json.dumps(metadata.get('tags', [])),
            metadata.get('license'),
            metadata.get('num_rows'),
            metadata.get('downloadSize'),
            datetime.now().isoformat(),
            json.dumps([keyword]),
            category,
            score['total_score'],
            score['structure_score'],
            score['size_score'],
            score['quality_score'],
            score['license_score'],
            score['feasibility_score'],
            score['recommendation'],
            score['conversion_complexity']
        ))

        conn.commit()

    def _start_discovery_run(self, categories: List[str]) -> int:
        """Start a new discovery run record."""
        if not self.db_manager:
            return None

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO discovery_runs (started_at, status, keywords_searched)
            VALUES (?, ?, ?)
        ''', (datetime.now().isoformat(), 'running', json.dumps(categories)))

        conn.commit()
        return cursor.lastrowid

    def _complete_discovery_run(self, run_id: int, results: Dict) -> None:
        """Complete a discovery run record."""
        if not self.db_manager:
            return

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE discovery_runs SET
                completed_at = ?,
                status = ?,
                total_found = ?,
                high_potential = ?,
                medium_potential = ?,
                errors = ?
            WHERE id = ?
        ''', (
            datetime.now().isoformat(),
            'completed' if not results['errors'] else 'completed_with_errors',
            results['total_found'],
            results['high_potential'],
            results['medium_potential'],
            json.dumps(results['errors']),
            run_id
        ))

        conn.commit()

    # =========================================================================
    # Approval Workflow Methods
    # =========================================================================

    def get_pending_approvals(self,
                              min_score: float = 0,
                              recommendation: str = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get datasets pending approval.

        Args:
            min_score: Minimum total score filter
            recommendation: Filter by recommendation ('high', 'medium', 'low')
            limit: Maximum number of results

        Returns:
            List of pending dataset records
        """
        if not self.db_manager:
            return []

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        query = '''
            SELECT * FROM discovered_datasets
            WHERE approval_status = 'pending'
            AND total_score >= ?
        '''
        params = [min_score]

        if recommendation:
            query += ' AND recommendation = ?'
            params.append(recommendation)

        query += ' ORDER BY total_score DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def approve_dataset(self, hf_id: str, approved_by: str = 'user') -> bool:
        """
        Approve a dataset for processing.

        Args:
            hf_id: HuggingFace dataset ID
            approved_by: Who approved (username or 'auto')

        Returns:
            True if successfully approved
        """
        if not self.db_manager:
            return False

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE discovered_datasets SET
                approval_status = 'approved',
                approved_by = ?,
                approved_at = ?
            WHERE hf_id = ? AND approval_status = 'pending'
        ''', (approved_by, datetime.now().isoformat(), hf_id))

        conn.commit()
        return cursor.rowcount > 0

    def reject_dataset(self, hf_id: str, reason: str) -> bool:
        """
        Reject a dataset.

        Args:
            hf_id: HuggingFace dataset ID
            reason: Rejection reason

        Returns:
            True if successfully rejected
        """
        if not self.db_manager:
            return False

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE discovered_datasets SET
                approval_status = 'rejected',
                rejection_reason = ?
            WHERE hf_id = ? AND approval_status = 'pending'
        ''', (reason, hf_id))

        conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # Processing Pipeline Methods
    # =========================================================================

    def process_approved_datasets(self, max_process: int = 10) -> Dict[str, Any]:
        """
        Process approved datasets and convert to flow networks.

        Args:
            max_process: Maximum number of datasets to process

        Returns:
            Dictionary with processing results
        """
        if not self.db_manager:
            return {'error': 'No database manager configured'}

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        # Get approved datasets
        cursor.execute('''
            SELECT * FROM discovered_datasets
            WHERE approval_status = 'approved'
            AND converted_network_id IS NULL
            ORDER BY total_score DESC
            LIMIT ?
        ''', (max_process,))

        datasets = [dict(row) for row in cursor.fetchall()]

        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'networks': [],
            'errors': []
        }

        for ds in datasets:
            results['processed'] += 1

            try:
                # Extract network
                network = self._extract_network(ds)

                if network:
                    # Save network
                    save_path = self._save_network(network, ds)

                    # Update database
                    cursor.execute('''
                        UPDATE discovered_datasets SET
                            approval_status = 'completed',
                            processing_attempts = processing_attempts + 1
                        WHERE hf_id = ?
                    ''', (ds['hf_id'],))

                    results['successful'] += 1
                    results['networks'].append({
                        'hf_id': ds['hf_id'],
                        'save_path': save_path,
                        'nodes': len(network.get('nodes', [])),
                        'edges': network.get('metadata', {}).get('total_edges', 'N/A')
                    })
                else:
                    raise ValueError("Network extraction returned None")

            except Exception as e:
                logger.error(f"Error processing {ds['hf_id']}: {e}")

                # Update error info
                cursor.execute('''
                    UPDATE discovered_datasets SET
                        processing_attempts = processing_attempts + 1,
                        last_processing_error = ?
                    WHERE hf_id = ?
                ''', (str(e), ds['hf_id']))

                results['failed'] += 1
                results['errors'].append({
                    'hf_id': ds['hf_id'],
                    'error': str(e)
                })

        conn.commit()
        return results

    def _extract_network(self, ds: Dict) -> Optional[Dict]:
        """
        Extract flow network from a dataset.

        Args:
            ds: Dataset record from database

        Returns:
            Network dictionary or None if extraction fails
        """
        # Import the flow extractor
        try:
            from src.huggingface_flow_extractor import HuggingFaceFlowExtractor
        except ImportError:
            from huggingface_flow_extractor import HuggingFaceFlowExtractor

        extractor = HuggingFaceFlowExtractor()

        hf_id = ds['hf_id']
        category = ds.get('discovery_category', '')
        complexity = ds.get('conversion_complexity', 'complex')

        logger.info(f"Extracting network from {hf_id} (complexity: {complexity})")

        # Try appropriate extraction method based on category and complexity
        network = None

        if complexity == 'direct':
            # Has edge_index or adjacency matrix
            network = extractor.extract_generic_graph_network(hf_id)

        elif category == 'biological':
            network = extractor.extract_protein_network(hf_id)

        elif category == 'transportation':
            network = extractor.extract_mobility_network(hf_id)

        elif category == 'supply_chain':
            network = extractor.extract_logistics_network(hf_id)

        elif category == 'transactions':
            # Try tabular flow extraction
            network = extractor.extract_tabular_flow_network(hf_id)

        else:
            # Generic extraction attempt
            try:
                network = extractor.extract_generic_graph_network(hf_id)
            except:
                network = extractor.extract_tabular_flow_network(hf_id)

        # Add discovery metadata
        if network:
            if 'metadata' not in network:
                network['metadata'] = {}
            network['metadata']['discovery'] = {
                'hf_id': hf_id,
                'category': category,
                'complexity': complexity,
                'extracted_at': datetime.now().isoformat()
            }

        return network

    def _save_network(self, network: Dict, ds: Dict) -> str:
        """
        Save extracted network to ecosystem samples.

        Args:
            network: Extracted network dictionary
            ds: Dataset record

        Returns:
            Path where network was saved
        """
        # Create filename from dataset name
        name = ds.get('name', ds['hf_id'].replace('/', '_'))
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)

        # Save to ecosystem_samples directory
        save_dir = Path('data/ecosystem_samples')
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{safe_name}.json"

        with open(save_path, 'w') as f:
            json.dump(network, f, indent=2)

        logger.info(f"Saved network to {save_path}")
        return str(save_path)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered datasets."""
        if not self.db_manager:
            return {}

        conn = self.db_manager._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Total counts by status
        cursor.execute('''
            SELECT approval_status, COUNT(*) as count
            FROM discovered_datasets
            GROUP BY approval_status
        ''')
        stats['by_status'] = {row['approval_status']: row['count'] for row in cursor.fetchall()}

        # Counts by recommendation
        cursor.execute('''
            SELECT recommendation, COUNT(*) as count
            FROM discovered_datasets
            WHERE approval_status = 'pending'
            GROUP BY recommendation
        ''')
        stats['pending_by_recommendation'] = {row['recommendation']: row['count'] for row in cursor.fetchall()}

        # Counts by category
        cursor.execute('''
            SELECT discovery_category, COUNT(*) as count
            FROM discovered_datasets
            GROUP BY discovery_category
        ''')
        stats['by_category'] = {row['discovery_category']: row['count'] for row in cursor.fetchall()}

        # Recent discovery runs
        cursor.execute('''
            SELECT * FROM discovery_runs
            ORDER BY started_at DESC
            LIMIT 5
        ''')
        stats['recent_runs'] = [dict(row) for row in cursor.fetchall()]

        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_discovery(categories: List[str] = None,
                    max_results: int = 20,
                    min_score: float = 50) -> Dict[str, Any]:
    """
    Run a quick discovery without database storage.

    Args:
        categories: Categories to search (None = all)
        max_results: Max results per category
        min_score: Minimum score threshold

    Returns:
        Discovery results
    """
    agent = HuggingFaceDiscoveryAgent(db_manager=None)
    return agent.run_discovery(
        categories=categories,
        max_per_category=max_results,
        min_score=min_score
    )


if __name__ == "__main__":
    # Example usage
    print("HuggingFace Dataset Discovery Agent")
    print("=" * 50)

    # Quick test without database
    print("\nRunning quick discovery for 'network_structure' category...")
    results = quick_discovery(
        categories=['network_structure'],
        max_results=5,
        min_score=30
    )

    print(f"\nFound {results['total_found']} datasets")
    print(f"  High potential: {results['high_potential']}")
    print(f"  Medium potential: {results['medium_potential']}")

    if results['datasets_found']:
        print("\nTop datasets:")
        for category, cat_results in results['datasets_found'].items():
            print(f"\n  {category}:")
            for ds in cat_results.get('datasets', [])[:3]:
                print(f"    - {ds['hf_id']}: {ds['weighted_score']:.1f} ({ds['score']['recommendation']})")
