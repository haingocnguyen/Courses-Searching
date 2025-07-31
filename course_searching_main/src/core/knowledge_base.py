import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from .caching import CacheManager
import json
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import ollama
from typing import *
from functools import lru_cache
from datetime import datetime
from collections import defaultdict
import time
import traceback
import uuid
import logging
import joblib
import os
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)  # You can adjust to INFO or ERROR in production
logger = logging.getLogger(__name__)
class KnowledgeBase:
    def __init__(self, data_path: str, use_caching=True):
        self.data_path = data_path
        self.use_caching = use_caching
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.embeddings_cache = self.cache_dir / "embeddings.joblib"
        self.graph_cache = self.cache_dir / "graph.graphml"
        self.data_cache = self.cache_dir / "data.joblib"  # Add data cache
        
        if self._should_rebuild():
            self._full_initialization()
        else:
            self._load_from_cache()
    def _convert_attributes_for_storage(self, G):
        """Convert list attributes to pipe-separated strings"""
        for node, data in G.nodes(data=True):
            for key in list(data.keys()):
                if isinstance(data[key], list):
                    data[key] = '|||'.join(map(str, data[key]))  # Use triple pipe as delimiter
        return G

    def _restore_attributes_from_storage(self, G):
        """Convert pipe-separated strings back to lists"""
        for node, data in G.nodes(data=True):
            for key in list(data.keys()):
                if '|||' in str(data[key]):
                    data[key] = str(data[key]).split('|||')
        return G
    def _should_rebuild(self):
        if not self.use_caching:
            return True
        return (not self.embeddings_cache.exists() or
                not self.graph_cache.exists() or
                not self.data_cache.exists() or  # Check for data cache
                os.path.getmtime(self.data_path) > os.path.getmtime(self.embeddings_cache))

    def _full_initialization(self):
        # Load and process data first
        self.data = self._load_and_preprocess(self.data_path)  # Pass data_path
        self.graph = self._build_knowledge_graph()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_indices()
        self._save_cache()

    def _save_cache(self):
        try:
            # Convert attributes before saving
            save_graph = self._convert_attributes_for_storage(self.graph)
            nx.write_graphml(save_graph, self.graph_cache)
            
            # Save other components
            joblib.dump({
                'data': self.data,
                'indices': self.indices,
                'semantic_index': self.semantic_index
            }, self.embeddings_cache)
            
        except Exception as e:
            print(f"Cache save failed: {str(e)}")
            self._clean_bad_cache()

    def _load_from_cache(self):
        try:
            # Load graph first
            raw_graph = nx.read_graphml(self.graph_cache)
            self.graph = self._restore_attributes_from_storage(raw_graph)
            
            # Load other components
            cache_data = joblib.load(self.embeddings_cache)
            self.data = cache_data['data']
            self.indices = cache_data['indices'] 
            self.semantic_index = cache_data['semantic_index']
            
        except Exception as e:
            print(f"Cache load failed: {str(e)}")
            self._clean_bad_cache()
            self._full_initialization()

    def _clean_bad_cache(self):
        """Remove corrupted cache files"""
        try:
            if self.graph_cache.exists():
                self.graph_cache.unlink()
            if self.embeddings_cache.exists():
                self.embeddings_cache.unlink()
        except Exception as e:
            print(f"Failed to clean cache: {str(e)}")
    def _validate_data(self):
        print("\nData Quality Report:")
        print(f"Total courses: {len(self.data)}")
        print("Missing values per column:")
        print(self.data.isnull().sum())
        print("Sample course features:")
        print(self.data.iloc[0]['features'])
        
    def _build_indices(self):
        """Create search indices with error handling"""
        try:
            # Semantic index
            text_embeddings = self.encoder.encode(
                self.data['search_text'].tolist()
            )
            self.semantic_index = faiss.IndexFlatIP(text_embeddings.shape[1])
            self.semantic_index.add(text_embeddings.astype('float32'))
            
            # Attribute indices
            self.indices = {
                'difficulty': self._build_attribute_index('difficulty'),
                'learning_style': self._build_attribute_index('learning_style'),
                'certifications': self._build_attribute_index('certifications')
            }
        except Exception as e:
            print(f"Index build error: {str(e)}")
            self.semantic_index = None
            self.indices = {}

   # In KnowledgeBase._build_attribute_index
    def _build_attribute_index(self, field: str) -> dict:
        """Build index with list value handling"""
        index = defaultdict(list)
        for idx, row in self.data.iterrows():
            values = row['features'].get(field, 'unknown')
            
            # Handle different value types
            if isinstance(values, list):
                for value in values:
                    if pd.notna(value):
                        index[str(value)].append(idx)
            elif pd.notna(values):
                index[str(values)].append(idx)
                
        return dict(index)

    def semantic_search(self, query: str, top_k: int = 10) -> List[dict]:
        """Safe semantic search implementation"""
        try:
            if not self.semantic_index:
                return []
                
            query_embed = self.encoder.encode(query)
            if query_embed.ndim == 1:
                query_embed = np.expand_dims(query_embed, 0)
                
            _, indices = self.semantic_index.search(
                query_embed.astype('float32'), 
                top_k
            )
            return self.data.iloc[indices[0]].to_dict('records')
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
        
    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        raw_data = pd.read_json(path)
        print("\nDATA VERIFICATION:")
        print("First course teaches:", raw_data.iloc[0]['knowledge_requirements']['teaches'])
        print("First course prerequisites:", raw_data.iloc[0]['knowledge_requirements']['prerequisites'])
        # Clean numerical fields
        numeric_fields = ['duration_months', 'rating']
        for field in numeric_fields:
            if field in raw_data.columns:
                raw_data[field] = pd.to_numeric(raw_data[field], errors='coerce').fillna(0)
        if 'course_id' not in raw_data.columns or raw_data['course_id'].isnull().any():
            raw_data['course_id'] = [f"course_{i}" for i in range(len(raw_data))]


        # Clean text fields
        text_fields = ['title', 'description', 'level']
        for field in text_fields:
            if field in raw_data.columns:
                raw_data[field] = raw_data[field].fillna('').astype(str)
        
        raw_data['features'] = raw_data.apply(self._extract_features, axis=1)
        
        # Universal search text composition
        raw_data['search_text'] = raw_data.apply(
            lambda x: ' '.join(filter(None, [
                x['title'],
                x['description'],
                ' '.join(x['features']['topics']),
                ' '.join(x['features']['career_paths']),
                x['features']['learning_style']
            ])), axis=1
        )
        
        return raw_data

    def _extract_features(self, item: pd.Series) -> dict:  # item là Pandas Series
        def safe_get(d, keys, default=None):
            current = d.to_dict() if isinstance(d, pd.Series) else d  # Chuyển Series thành dict
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current if current is not None else default

        return {
            'topics': safe_get(item, ["knowledge_requirements", "teaches"], []),
            'prerequisites': safe_get(item, ["knowledge_requirements", "prerequisites"], []),
            'career_paths': safe_get(item, ["learning_path", "career_paths"], []),
            'difficulty': (safe_get(item, ["learning_path", "suitable_for"], ["intermediate"]))[0].lower(),
            'duration': item.get("duration_months", 0),
            'learning_style': safe_get(item, ["course_info", "type"], "self-paced"),
            'certifications': item.get("certifications", [])
        }

    def _build_knowledge_graph(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()

        # 1. Tạo các node khái niệm phân cấp
        self._build_concept_hierarchy(graph)

        # 2. Với mỗi khoá học, thêm node và gọi tạo quan hệ
        for idx, row in self.data.iterrows():
            course_id = row.get('course_id', f"course_{uuid.uuid4()}")
            features = row['features']

            # Thêm node khoá học
            graph.add_node(course_id,
                        type='course',
                        created=datetime.now().isoformat(),
                        **features)

            # Debug: log mỗi khoá khi vào đây
            #print(f"[DEBUG] Building edges for course: {course_id}")

            # 3. Tạo các quan hệ teaches/requires/leads_to
            self._create_relationships(graph, course_id, features)

            # 4. Tạo quan hệ prerequisite giữa các khoá
            self._resolve_prerequisites(graph, course_id, features.get('prerequisites', []))
            print(f"[DBG] Edges just after relationship builds: {graph.number_of_edges()}")

        # 5. Tính toán metrics
        self._calculate_graph_metrics(graph)

        # In kết quả debug
        print("\nSample Graph Connections:")
        sample_course = next((n for n in graph.nodes if graph.nodes[n].get('type') == 'course'), None)
        if sample_course:
            print(f"Course {sample_course} connects to:")
            for u, v, key in graph.out_edges(sample_course, keys=True):
                edge_data = graph.get_edge_data(u, v, key)
                print(f"- {v} ({edge_data['relationship']}) [Key: {key}]")
        print("\nKnowledge Graph Verification:")
        print(f"Total nodes: {len(graph.nodes)}")
        print("Total edges:", graph.number_of_edges())
        print("Sample course features:")
        print(next((n for n in graph.nodes if graph.nodes[n].get('type') == 'course'), None))
        return graph


    
    def _create_relationships(self, graph, course_id, features):
        relationship_config = {
            'teaches': ('topics', 0.9),
            'requires': ('prerequisites', 0.8),
            'leads_to': ('career_paths', 0.85)
        }

        logger.debug(f"Processing course_id: {course_id}")

        for rel_type, (feature_key, weight) in relationship_config.items():
            values = features.get(feature_key, [])
            logger.debug(f"Feature '{feature_key}' for relationship '{rel_type}' has values: {values}")

            if not isinstance(values, list):
                values = [values]
                logger.debug(f"Converted non-list value to list: {values}")

            for value in values:
                if pd.notna(value) and value:
                    node_id = f"{rel_type.split('_')[0]}:{value.lower().strip()}"
                    logger.debug(f"Adding node: {node_id} with type {rel_type.split('_')[0]}")
                    graph.add_node(node_id, type=rel_type.split('_')[0])

                    logger.debug(f"Adding edge from {course_id} to {node_id} with rel: {rel_type}, weight: {weight}")
                    graph.add_edge(course_id, node_id, relationship=rel_type, weight=weight)

                    concept_nodes = self._get_concept_hierarchy(value)
                    logger.debug(f"Hierarchical concepts for '{value}': {concept_nodes}")
                    for concept_node in concept_nodes:
                        logger.debug(f"Adding hierarchical edge from {course_id} to {concept_node} with weight: {weight * 0.8}")
                        graph.add_edge(course_id, concept_node, relationship=rel_type, weight=weight * 0.8)

                        
    def _get_concept_hierarchy(self, concept: str) -> List[str]:
        """Break concept into hierarchical components"""
        parts = concept.lower().split('::')
        return [f"concept:{':'.join(parts[:i+1])}" for i in range(len(parts))]

    def _connect_related_courses(self, graph, source_course, concept):
        """Link courses teaching required concepts"""
        for _, row in self.data.iterrows():
            if concept in row['features'].get('teaches', []):
                graph.add_edge(source_course, row['course_id'],
                             relationship='fulfills',
                             weight=0.75)

    def _calculate_graph_metrics(self, graph):
        """Add centrality and community detection"""
        betweenness = nx.betweenness_centrality(graph)
        nx.set_node_attributes(graph, betweenness, 'centrality')
        
        communities = nx.community.louvain_communities(graph)
        for idx, comm in enumerate(communities):
            for node in comm:
                graph.nodes[node]['community'] = idx

    def _add_external_knowledge(self):
        """Integrate external knowledge sources"""
        # Example: Add skill taxonomy
        self.graph.add_node("skill:cloud::aws", 
                          type='skill',
                          level='specific',
                          description="Amazon Web Services cloud platform")
        
    def _build_concept_hierarchy(self, graph):
        """Construct hierarchical concept relationships"""
        all_concepts = set()
        for _, row in self.data.iterrows():
            all_concepts.update(row['features'].get('topics', []))
        
        for concept in all_concepts:
            parts = concept.lower().split('::')
            for i in range(len(parts)):
                node_id = f"concept:{':'.join(parts[:i+1])}"
                graph.add_node(node_id, type='concept', level=i)
                if i > 0:
                    parent_id = f"concept:{':'.join(parts[:i])}"
                    graph.add_edge(parent_id, node_id, 
                                 relationship='subconcept')
                    
    def _resolve_prerequisites(self, graph, course_id, prerequisites):
        """Create course-to-course prerequisite relationships"""
        for prereq in prerequisites:
            # Find courses that teach this prerequisite
            for _, row in self.data.iterrows():
                if row['course_id'] == course_id:
                    continue  # Skip self
                
                if prereq in row['features'].get('topics', []):
                    graph.add_edge(
                        course_id, 
                        row['course_id'],
                        relationship='prerequisite',
                        weight=0.85,
                        label=f"Requires {prereq}"
                    )