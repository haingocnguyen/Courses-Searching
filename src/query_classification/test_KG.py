import json
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, List, Any
from collections import defaultdict
import uuid

class KnowledgeGraphConfig:
    """Configuration for adaptable knowledge graph construction"""
    def __init__(self):
        # Define data schema mappings
        self.schema = {
            'features': {
                'topics': ('knowledge_requirements', 'teaches'),
                'prerequisites': ('knowledge_requirements', 'prerequisites'),
                'career_paths': ('learning_path', 'career_paths'),
                'difficulty': ('learning_path', 'suitable_for'),
                'duration': ('duration_months',),
                'learning_style': ('course_info', 'type')
            },
            'relationships': {
                'teaches': {
                    'source': 'course',
                    'target': 'concept',
                    'path': ('knowledge_requirements', 'teaches')
                },
                'requires': {
                    'source': 'course',
                    'target': 'course',
                    'path': ('knowledge_requirements', 'prerequisites'),
                    'resolve_via': 'teaches'
                },
                'leads_to': {
                    'source': 'course',
                    'target': 'career',
                    'path': ('learning_path', 'career_paths')
                }
            },
            'concept_hierarchies': {
                'skills': ['category', 'subcategory', 'specific']
            }
        }

class GeneralizedKnowledgeBase:
    def __init__(self, data_path: str, config: KnowledgeGraphConfig = None):
        self.config = config or KnowledgeGraphConfig()
        self.data = self._load_and_preprocess(data_path)
        self.graph = nx.MultiDiGraph()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_components()
        
    def _build_components(self):
        """Build all knowledge base components with error handling"""
        self._construct_concept_hierarchy()
        self._build_knowledge_graph()
        self._build_indices()

    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        """Load data with schema-agnostic preprocessing"""
        raw_data = pd.read_json(path)
        
        # Generic type handling
        numeric_fields = ['duration_months', 'rating', 'positive_percentage']
        text_fields = ['title', 'description', 'category', 'sub_category']
        
        for field in numeric_fields:
            if field in raw_data.columns:
                raw_data[field] = pd.to_numeric(raw_data[field], errors='coerce').fillna(0)
        
        for field in text_fields:
            if field in raw_data.columns:
                raw_data[field] = raw_data[field].fillna('').astype(str)
        
        raw_data['features'] = raw_data.apply(self._extract_features, axis=1)
        raw_data['search_text'] = raw_data.apply(self._generate_search_text, axis=1)
        
        return raw_data

    def _extract_features(self, item: Dict) -> Dict:
        """Schema-aware feature extraction"""
        features = {}
        for feature, path in self.config.schema['features'].items():
            try:
                value = item
                for key in path:
                    value = value.get(key, {}) if isinstance(value, dict) else getattr(value, key, None)
                features[feature] = value
            except (KeyError, AttributeError):
                features[feature] = None
        
        # Normalize feature formats
        list_features = ['topics', 'prerequisites', 'career_paths']
        for feat in list_features:
            if not isinstance(features.get(feat), list):
                features[feat] = [features[feat]] if features[feat] else []
        
        return features

    def _generate_search_text(self, row) -> str:
        """Dynamic search text generation"""
        components = [
            row.get('title', ''),
            row.get('description', ''),
            ' '.join(row['features'].get('topics', [])),
            ' '.join(row['features'].get('career_paths', [])),
            row.get('category', '')
        ]
        return ' '.join(filter(None, components))

    def _construct_concept_hierarchy(self):
        """Build hierarchical concept relationships"""
        for hierarchy, levels in self.config.schema['concept_hierarchies'].items():
            for course in self.data.itertuples():
                for concept in course.features.get('topics', []):
                    self._add_concept_hierarchy(concept, levels)

    def _add_concept_hierarchy(self, concept: str, levels: List[str]):
        """Recursively add concept hierarchy to graph"""
        parts = concept.split('::')
        for i in range(len(parts)):
            node_id = f"concept:{':'.join(parts[:i+1])}"
            self.graph.add_node(node_id, 
                              type='concept',
                              level=levels[i] if i < len(levels) else 'specific')
            
            if i > 0:
                parent_id = f"concept:{':'.join(parts[:i])}"
                self.graph.add_edge(parent_id, node_id, 
                                  relationship='subconcept')

    def _build_knowledge_graph(self):
        """Multi-relationship graph construction"""
        # Add course nodes
        for course in self.data.itertuples():
            self.graph.add_node(course.course_id, 
                             type='course',
                             **course.features)
            
            # Add relationships
            for rel_name, rel_config in self.config.schema['relationships'].items():
                self._process_relationship(course, rel_name, rel_config)

    def _process_relationship(self, course, rel_name: str, rel_config: Dict):
        """Generic relationship processor"""
        try:
            # Get relationship data from nested structure
            relationship_data = course
            for key in rel_config['path']:
                relationship_data = relationship_data.__getattribute__(key)
            
            # Resolve relationships
            for value in relationship_data if isinstance(relationship_data, list) else [relationship_data]:
                if pd.isna(value):
                    continue
                
                if rel_config['target'] == 'course':
                    self._resolve_course_relationship(course, rel_name, value, rel_config)
                else:
                    target_id = f"{rel_config['target']}:{value.lower().strip()}"
                    self.graph.add_edge(course.course_id, target_id,
                                      relationship=rel_name)

        except (AttributeError, KeyError) as e:
            pass

    def _resolve_course_relationship(self, source_course, rel_name: str, target_value: str, rel_config: Dict):
        """Resolve course-to-course relationships using concept resolution"""
        resolve_field = rel_config.get('resolve_via')
        if not resolve_field:
            return
        
        # Find courses that teach the required concept
        for target_course in self.data.itertuples():
            if target_value.lower() in [t.lower() for t in target_course.features.get(resolve_field, [])]:
                self.graph.add_edge(source_course.course_id, 
                                 target_course.course_id,
                                 relationship=rel_name,
                                 resolved_concept=target_value)

    def _build_indices(self):
        """Dynamic index builder"""
        self.semantic_index = faiss.IndexFlatL2(384)  # Match encoder dimension
        text_embeddings = self.encoder.encode(self.data['search_text'].tolist())
        self.semantic_index.add(text_embeddings.astype('float32'))
        
        # Build attribute indices dynamically
        self.indices = {
            attr: self._build_attribute_index(attr)
            for attr in self.config.schema['features'].keys()
        }

    def _build_attribute_index(self, attribute: str) -> Dict:
        index = defaultdict(list)
        for idx, row in self.data.iterrows():
            values = row['features'].get(attribute, [])
            values = values if isinstance(values, list) else [values]
            for value in values:
                if pd.notna(value):
                    index[str(value).lower().strip()].append(idx)
        return dict(index)

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        try:
            query_embed = self.encoder.encode(query)
            if query_embed.ndim == 1:
                query_embed = np.expand_dims(query_embed, 0)
                
            _, indices = self.semantic_index.search(query_embed.astype('float32'), top_k)
            return self.data.iloc[indices[0]].to_dict('records')
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def graph_search(self, start_node: str, max_hops: int = 3) -> List[Dict]:
        """Generalized graph traversal with multi-hop support"""
        results = []
        try:
            for node in nx.dfs_preorder_nodes(self.graph, start_node, depth_limit=max_hops):
                if self.graph.nodes[node].get('type') == 'course':
                    results.append(self.data[self.data['course_id'] == node].iloc[0].to_dict())
        except nx.NetworkXError:
            pass
        return results

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Combine semantic and graph-based results"""
        semantic_results = self.semantic_search(query, top_k)
        concepts = self._extract_search_concepts(query)
        
        graph_results = []
        for concept in concepts:
            graph_results += self.graph_search(f"concept:{concept}", max_hops=2)
        
        return self._merge_results(semantic_results, graph_results)[:top_k]

    def _extract_search_concepts(self, query: str) -> List[str]:
        """Identify relevant concepts from search text"""
        query_embed = self.encoder.encode(query)
        _, indices = self.semantic_index.search(query_embed.astype('float32'), 5)
        return [
            concept
            for idx in indices[0]
            for concept in self.data.iloc[idx]['features']['topics']
        ]

    def _merge_results(self, semantic: List, graph: List) -> List:
        """Deduplicate and rank merged results"""
        seen = set()
        merged = []
        for item in semantic + graph:
            if item['course_id'] not in seen:
                seen.add(item['course_id'])
                merged.append(item)
        return sorted(merged, 
                    key=lambda x: x.get('rating', 0), 
                    reverse=True)

# Usage Example
if __name__ == "__main__":
    config = KnowledgeGraphConfig()
    knowledge_base = GeneralizedKnowledgeBase("D:\\Thesis\\Courses-Searching\\src\\db\\processed_courses_detail.json", config)
    
    # Example searches
    print("Semantic Search Results:")
    print(knowledge_base.semantic_search("machine learning for beginners", top_k=3))
    
    print("\nGraph Search Results:")
    print(knowledge_base.graph_search("concept:python", max_hops=2))
    
    print("\nHybrid Search Results:")
    print(knowledge_base.hybrid_search("data science career path", top_k=5))