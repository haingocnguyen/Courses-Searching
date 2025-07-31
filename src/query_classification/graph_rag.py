import os
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import difflib
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from diskcache import Cache
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import torch
import threading
from cachetools import LRUCache
import streamlit as st


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("course_advisor_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


    
    # ======================
# 1. Optimized Knowledge Graph Construction 
# ======================
class CourseKnowledgeGraph:
    def __init__(self, data_path: str, use_cache: bool = True):
        self.virtual_edges = defaultdict(dict)
        self.cache_dir  = "graph_cache"
        self.data_path  = data_path
        self.use_cache  = use_cache
        self.data_hash  = self._calculate_data_hash()
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.use_cache and self._try_load_cache():
            logger.info("Loaded graph from cache")
            # **Re-init những thuộc tính còn thiếu**
            self.data     = pd.read_json(self.data_path)  # hoặc gọi self._load_data()
            device        = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder  = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            self.llm      = GraphEnhancementEngine()
            # concept_nodes cũng cần, vì _build_indices (lần trước) đã lưu:
            # khi save_cache, thêm 'concept_nodes' vào pickle rồi load vào self.concept_nodes
            # nếu không, bạn có thể rebuild từ self.graph:
            self.concept_nodes = [n for n,d in self.graph.nodes(data=True) if n.startswith('concept:')]
        else:
            self._initialize_graph()
    

    def _calculate_data_hash(self) -> str:
        with open(self.data_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _try_load_cache(self) -> bool:
        cache_file = os.path.join(self.cache_dir, f"{self.data_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.graph         = data['graph']
                    self.course_index  = data['course_index']
                    self.concept_index = data['concept_index']
                    self.concept_nodes = data.get('concept_nodes', [])

                return True
            except Exception as e:
                logger.error(f"Cache load failed: {str(e)}")
        return False

    def _save_cache(self):
        cache_file = os.path.join(self.cache_dir, f"{self.data_hash}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'course_index': self.course_index,
                'concept_index': self.concept_index,
                'concept_nodes': self.concept_nodes
            }, f)

    def _initialize_graph(self):
        self.data = self._load_data()
        self.graph = nx.MultiDiGraph()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.llm = GraphEnhancementEngine()
        
        self._batch_build_graph()
        self._batch_enhance_relationships()
        self._build_indices()
        
        if self.use_cache:
            self._save_cache()

    def _batch_build_graph(self):
        logger.info(f"Total courses to process: {len(self.data)}")
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data.iloc[i:i+self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch)} items")
            for _, row in batch.iterrows():
                self._add_course_node(row)
                self._connect_entities_safe(row, 'teaches', 'concept')
                self._connect_entities_safe(row, 'career_paths', 'career')
        logger.info(f"Graph construction completed. Total nodes: {len(self.graph.nodes)}, edges: {len(self.graph.edges)}")
        logger.debug(f"Sample nodes: {list(self.graph.nodes)[:5]}")

    def _connect_entities_safe(self, row, field, node_type):
        """Enhanced entity connection with validation"""
        try:
            entities = row['knowledge_requirements'].get(field, []) or []
            for entity in entities:
                if isinstance(entity, dict):  # Handle nested dicts
                    entity = entity.get('name', '')
                if isinstance(entity, str):
                    entity_clean = entity.lower().strip()
                    if entity_clean:
                        entity_id = f"{node_type}:{entity_clean.replace(' ', '_')}"
                        self.graph.add_node(entity_id, type=node_type)
                        self.graph.add_edge(row['course_id'], entity_id, relationship=field)
        except Exception as e:
            logger.error(f"Connection failed for {row['course_id']}: {str(e)}")

    def _batch_enhance_relationships(self):
        courses = self.data.to_dict('records')
        batch_size = 20
        relations = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(0, len(courses), batch_size):
                batch = courses[i:i+batch_size]
                futures.append(executor.submit(
                    self.llm.batch_extract_relationships,
                    batch
                ))
            
            for future in as_completed(futures):
                try:
                    relations.extend(future.result())
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
        
        for course, rel in zip(courses, relations):
            self._process_relationships_safe(course['course_id'], rel)

    def _handle_batch_results(self, futures):
        """Process completed futures from batch processing"""
        from concurrent.futures import as_completed
        
        try:
            for future in as_completed(futures):
                try:
                    future.result()  # Get results/raise exceptions
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Batch handler failed: {str(e)}")

    def _process_course_relationships(self, row):
        """Safe relationship processing with retries"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                teaches = [str(t) for t in row['knowledge_requirements'].get('teaches', []) if t]
                relations = self.llm.extract_relationships(
                    str(row['title']),
                    str(row.get('description', '')),
                    teaches
                )
                self._process_relationships_safe(row['course_id'], relations)
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {row['course_id']}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Final failure for {row['course_id']}")

    def _process_relationships_safe(self, course_id, relations):
        """Validated relationship processing"""
        valid_relationships = ['prerequisites', 'prepares_for', 'uses', 'complements']
        for rel_type in valid_relationships:
            for target in relations.get(rel_type, []):
                if isinstance(target, str):
                    target_id = self._resolve_node(target.strip())
                    if target_id and target_id != course_id:
                        self.graph.add_edge(course_id, target_id, 
                                          relationship=rel_type, 
                                          source='auto')

    
    def _load_data(self) -> pd.DataFrame:
        with open(self.data_path) as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        return df.apply(self._normalize_row, axis=1)
    
    def _normalize_row(self, row):
        # Lấy về dict hoặc khởi tạo mới
        kr = row.get('knowledge_requirements')
        if not isinstance(kr, dict):
            kr = {}
        teaches = kr.get('teaches') or []
        prerequisites = kr.get('prerequisites') or []
        
        row['knowledge_requirements'] = {
            'teaches': [t for t in teaches if isinstance(t, str) and t.strip()],
            'prerequisites': [p for p in prerequisites if isinstance(p, str) and p.strip()]
        }

        lp = row.get('learning_path')
        if not isinstance(lp, dict):
            lp = {}
        suitable_for = lp.get('suitable_for') or []
        career_paths = lp.get('career_paths') or []
        row['learning_path'] = {
            'suitable_for': [s for s in suitable_for if isinstance(s, str) and s.strip()],
            'career_paths': [c for c in career_paths if isinstance(c, str) and c.strip()]
        }

        ci = row.get('course_info') or {}
        langs = ci.get('subtitle_languages') or []
        row['course_info'] = {
            **ci,
            'subtitle_languages': [l for l in langs if isinstance(l, str) and l.strip()]
        }

        return row

    def _build_base_graph(self):
        """Initialize core graph structure from raw data"""
        for _, row in self.data.iterrows():
            self._add_course_node(row)
            self._connect_entities(row, 'teaches', 'concept')
            self._connect_entities(row, 'career_paths', 'career')
    
    def _add_course_node(self, row):
        """Create detailed course node with enhanced null safety"""
        self.graph.add_node(row['course_id'],
                          type='course',
                          title=row['title'],
                          description=row.get('description', ''),
                          duration=row.get('duration_months', 0),
                          difficulty=(row['learning_path']
                                      .get('suitable_for', [''])[0]
                                      .lower()),
                          rating=row.get('rating', 0),
                          languages=[lang.lower() for lang in 
                                    row['course_info']['subtitle_languages'] 
                                    if isinstance(lang, str)])
    
    def _connect_entities(self, row, field, node_type):
        """Connect courses to related entities with null safety"""
        entities = row['knowledge_requirements'].get(field, []) or []
        for entity in entities:
            if not isinstance(entity, str):  # Handle non-string values
                continue
            entity_id = f"{node_type}:{entity.lower().replace(' ', '_')}"
            self.graph.add_node(entity_id, type=node_type)
            self.graph.add_edge(row['course_id'], entity_id, relationship=field)
    
    def _enhance_relationships(self):
        """Discover implicit relationships using LLM with error handling"""
        for _, row in self.data.iterrows():
            try:
                # Handle missing teaches list
                teaches = row['knowledge_requirements'].get('teaches', []) or []
                relations = self.llm.extract_relationships(
                    row['title'],
                    row['description'],
                    teaches
                )
                self._process_relationships(row['course_id'], relations)
            except Exception as e:
                logger.error(f"Relationship extraction failed for {row['course_id']}: {str(e)}")
    
    def _process_relationships(self, course_id, relations):
        """Add validated relationships to graph"""
        for rel_type, targets in relations.items():
            for target in targets:
                target_id = self._resolve_node(target)
                if target_id and self._validate_relationship(course_id, target_id, rel_type):
                    self.graph.add_edge(course_id, target_id, 
                                      relationship=rel_type, source='auto')
    
    def _resolve_node(self, name: str) -> Optional[str]:
        """Fuzzy node resolution with similarity threshold"""
        candidates = [n for n in self.graph.nodes if not n.startswith('course:')]
        matches = difflib.get_close_matches(name.lower(), candidates, n=1, cutoff=0.7)
        return matches[0] if matches else None
    
    def _validate_relationship(self, source, target, rel_type):
        """Ensure relationship meets quality criteria"""
        if source == target:
            return False
        existing = self.graph.get_edge_data(source, target)
        return not existing or all(e['relationship'] != rel_type for e in existing.values())
    
    def _build_indices(self):
        # Sử dụng IVF index cho hiệu suất tốt hơn với dữ liệu lớn
        dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Course index
        quantizer = faiss.IndexFlatIP(dimension)
        if len(self.data) < 1000:
            self.course_index = faiss.IndexFlatIP(dimension)
            self.concept_index = faiss.IndexFlatIP(dimension)
        else:
            # Giữ nguyên logic cũ cho dữ liệu lớn
            self.course_index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.concept_index = faiss.IndexIVFFlat(quantizer, dimension, 50)
        texts = self.data.apply(lambda x: f"{x['title']} {x['description']}", axis=1)
        embeddings = self.encoder.encode(texts.tolist())
        self.course_index.train(embeddings.astype('float32'))
        self.course_index.add(embeddings.astype('float32'))
        
        # Concept index
        self.concept_nodes = [n for n in self.graph.nodes if n.startswith('concept:')]  # Lưu danh sách concept nodes
        concept_texts = [c.split(':', 1)[1].replace('_', ' ') for c in self.concept_nodes]
        concept_embeddings = self.encoder.encode(concept_texts)
        # self.concept_index = faiss.IndexIVFFlat(quantizer, dimension, 50)
        self.concept_index.train(concept_embeddings.astype('float32'))
        self.concept_index.add(concept_embeddings.astype('float32'))
    
    def get_virtual_relationship(self, source, target):
        """Infer implicit relationships on-demand"""
        if (source, target) in self.virtual_edges:
            return self.virtual_edges[(source, target)]
        
        source_data = self._get_node_data(source)
        target_data = self._get_node_data(target)
        relationship = self.llm.infer_relationship(source_data, target_data)
        
        if relationship['confidence'] > 0.7:
            self.virtual_edges[(source, target)] = relationship
            return relationship
        
        return None
    
    def _get_node_data(self, node_id):
        """Retrieve node information for LLM processing"""
        if node_id.startswith('course:'):
            return self.data[self.data['course_id'] == node_id].iloc[0].to_dict()
        return {'id': node_id, 'type': self.graph.nodes[node_id]['type']}

# ======================
# 2. Intelligent Query Processing
# ======================
class MultiHopQueryParser:
    """Advanced query understanding with multi-hop detection"""
    
    def __init__(self, graph: CourseKnowledgeGraph):
        self.graph = graph
        self.llm = GraphEnhancementEngine()
    
    def parse(self, query: str) -> Dict:
        logger.debug(f"[Parser] Raw query: {query}")
        raw = self.llm.analyze_query(query)
        logger.debug(f"[Parser] LLM analysis: {raw}")
        targets = self._expand_concepts(raw.get("concepts", []))
        logger.debug(f"[Parser] Expanded concepts: {targets}")
        pattern = self._detect_hop_pattern_with_llm(query)
        logger.debug(f"[Parser] Detected pattern: {pattern}")
        constraints = self._normalize_constraints(raw.get("constraints", {}))
        
        # Extract and validate num_courses
        num_courses = raw.get("num_courses")
        try:
            num_courses = int(num_courses) if num_courses is not None else None
            if num_courses is not None and num_courses <= 0:
                num_courses = None
        except (TypeError, ValueError):
            num_courses = None

        return {
            "target_concepts": targets,
            "relationship_pattern": pattern,
            "constraints": constraints,
            "num_courses": num_courses
        }
    
    def _expand_concepts(self, concepts: List[str]) -> List[str]:
        """Mở rộng concept với fallback tự động"""
        expanded = set()
        
        # Fallback nếu không có concepts đầu vào
        if not concepts:
            logger.warning("No input concepts, using fallback")
            return ["concept:data_science"]
        
        for concept in concepts:
            try:
                concept_embed = self.graph.encoder.encode([concept])
                distances, indices = self.graph.concept_index.search(concept_embed, 3)
                
                # Thêm các concept tìm được
                for i, score in zip(indices[0], distances[0]):
                    if score > 0.3:
                        expanded.add(self.graph.concept_nodes[i])
                        
                # Fallback nếu không tìm thấy concept phù hợp
                if not expanded:
                    expanded.add(f"concept:{concept.lower().replace(' ', '_')}")
                    
            except Exception as e:
                logger.error(f"Concept expansion failed for '{concept}': {str(e)}")
                expanded.add(f"concept:{concept.lower().replace(' ', '_')}")
        
        return list(expanded)
    
    def _detect_hop_pattern_with_llm(self, query: str) -> Dict:
        try:
            relationship_mapping = {
                'sequential': ['prerequisites', 'prepares_for'],
                'parallel': ['complements', 'related_to'],
                'conditional': ['requires', 'depends_on']
            }
            prompt = f"""
                You are an analyzer that categorizes the relationship pattern implied by a user’s educational query.
                Possible patterns:
                - "sequential": steps that build one after another
                - "parallel": topics combined or taken side-by-side
                - "conditional": topics with prerequisites or conditional relationships

                Please output a JSON object with:
                {{
                    "type": "sequential" | "parallel" | "conditional",
                    "confidence": a float between 0.0 and 1.0
                }}

                Query: "{query}"
                """
            response = self.llm._safe_llm_call(prompt)
            
            # Sửa phần xử lý confidence
            confidence = 0.7  # Giá trị mặc định
            try:
                raw_confidence = response.get("relationship_patterns", {}).get("confidence", 0.7)
                confidence = max(0.0, min(float(raw_confidence), 1.0))
            except (TypeError, ValueError):
                pass
            
            return {
                'type': response.get("relationship_patterns", {}).get("type", "sequential"),
                'confidence': confidence,
                'allowed_relationships': relationship_mapping.get(
                    response.get("relationship_patterns", {}).get("type", "sequential"), 
                    ['prerequisites', 'prepares_for']
                )
            }
        except Exception as e:
            logger.error(f"Pattern detection failed: {str(e)}")
            return {
                'type': 'sequential',
                'confidence': 0.7,
                'allowed_relationships': ['prerequisites', 'prepares_for']
            }
            
    def _normalize_constraints(self, raw: Dict) -> Dict:
        """Validate and normalize user constraints"""
        return {
            'difficulty': self._normalize_difficulty(raw.get('difficulty')),
            'max_duration': min(int(raw.get('max_duration', 12)), 24),
            'min_rating': max(float(raw.get('min_rating', 3.5)), 0),
            'languages': [lang.lower() for lang in raw.get('languages', [])]
        }
    
    def _normalize_difficulty(self, level: str) -> str:
        # Thêm xử lý cho giá trị số
        if isinstance(level, (int, float)):
            level_map = {
                1: 'beginner',
                2: 'intermediate',
                3: 'advanced'
            }
            return level_map.get(int(level), 'intermediate')
        # Phần xử lý string giữ nguyên
        level = level.lower()
        for std_level in ['beginner', 'intermediate', 'advanced']:
            if std_level in level:
                return std_level
        return 'intermediate'

# ======================
# 3. Hybrid Search Engine
# ======================
class GraphAwareSearch:
    """Combines semantic and graph-based retrieval"""
    
    def __init__(self, graph: CourseKnowledgeGraph):
        self.graph = graph
        self.encoder = graph.encoder
        
    
    def search(self, parsed_query: Dict) -> List[Dict]:
        logger.debug(f"[Search] Parsed query: {parsed_query}")

        # 1) Semantic
        sem = self._semantic_search(parsed_query["target_concepts"])
        logger.debug(f"[Search] Semantic results ({len(sem)}): {[r['course_id'] for r in sem]}")

        # 2) Graph
        try:
            graph = self._graph_traversal_search(parsed_query)
            logger.debug(f"[Search] Graph results ({len(graph)}): {[r['course_id'] for r in graph]}")
        except Exception as ex:
            logger.error(f"[Search] Graph traversal failed: {ex}")
            graph = []

        # 3) Merge
        merged = self._merge_results(sem, graph)
        logger.debug(f"[Search] Merged ({len(merged)}): {[r['course_id'] for r in merged]}")

        # 4) Filter
        filtered = self._apply_adaptive_filters(merged, parsed_query["constraints"])
        logger.debug(f"[Search] Filtered ({len(filtered)}): {[r['course_id'] for r in filtered]}")

        # 5) Rank
        ranked = self._rank_results(filtered, parsed_query)
        logger.debug(f"[Search] Final ranked ({len(ranked)}): {[r['course_id'] for r in ranked]}")

        return ranked
    
    def _semantic_search(self, concepts: List[str]) -> List[Dict]:
        """Vector-based similarity search"""
        try:
            query_text = ' '.join([c.split(':', 1)[1].replace('_', ' ') for c in concepts])
            query_embed = self.encoder.encode([query_text])
            distances, indices = self.graph.course_index.search(query_embed, 50)
            
            return [self._format_result(self.graph.data.iloc[i], score)
                for i, score in zip(indices[0], distances[0]) if score > 0.2]
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return self._fallback_keyword_search(concepts)
    
    def _fallback_keyword_search(self, concepts: List[str]) -> List[Dict]:
        """Keyword-based search when vector search fails"""
        query_keywords = set(' '.join(concepts).lower().split())
        results = []
        
        for _, row in self.graph.data.iterrows():
            course_text = f"{row['title']} {row['description']}".lower()
            matches = len(query_keywords & set(course_text.split()))
            if matches > 0:
                results.append(self._format_result(row, matches/len(query_keywords)))
        
        return sorted(results, key=lambda x: -x['score'])
    
    def _graph_traversal_search(self, parsed_query: Dict) -> List[Dict]:
        navigator = MultiHopNavigator(self.graph)
        return navigator.find_paths(
            start_nodes=parsed_query["target_concepts"],
            pattern=parsed_query["relationship_pattern"]
        )

    
    def _merge_results(self, semantic: List[Dict], graph: List[Dict]) -> List[Dict]:
        """Deduplicate and combine results"""
        merged = {}
        for result in semantic + graph:
            cid = result['course_id']
            if cid in merged:
                merged[cid]['score'] = max(merged[cid]['score'], result['score'])
            else:
                merged[cid] = result
        return list(merged.values())
    
    def _apply_adaptive_filters(self, results: List[Dict], constraints: Dict) -> List[Dict]:
        """Intelligent filtering with fallback logic"""
        filtered = []
        filter_attempts = [
            constraints,
            {**constraints, 'max_duration': constraints['max_duration'] + 6},
            {**constraints, 'min_rating': max(constraints['min_rating'] - 0.5, 3.0)},
            {**constraints, 'difficulty': None}
        ]
        
        for attempt in filter_attempts:
            filtered = [
                r for r in results
                if (attempt['difficulty'] is None or r['difficulty'] == attempt['difficulty'])
                and r['duration'] <= attempt['max_duration']
                and r['rating'] >= attempt['min_rating']
            ]
            if filtered:
                break
                
        return filtered
    
    def _rank_results(self, results: List[Dict], parsed_query: Dict) -> List[Dict]:
        """Multi-factor ranking algorithm"""
        for result in results:
            # Base relevance score
            result['score'] = result.get('score', 0) * 0.7
            
            # Contextual boosts
            if self._matches_career_path(result, parsed_query):
                result['score'] += 0.2
            if self._has_prerequisites(result, parsed_query):
                result['score'] += 0.1
            
            # Normalize score
            result['score'] = min(result['score'], 1.0)
        
        return sorted(results, key=lambda x: (-x['score'], -x['rating'], x['duration']))
    
    def _format_result(self, course_row, score: float) -> Dict:
        """Standardize result format"""
        return {
            'course_id': course_row['course_id'],
            'title': course_row['title'],
            'rating': course_row['rating'],
            'duration': course_row['duration_months'],
            'difficulty': course_row['learning_path']['suitable_for'][0].lower(),
            'languages': [lang.lower() for lang in course_row['course_info']['subtitle_languages']],
            'score': float(score)
        }
    def _matches_career_path(self, result: Dict, parsed_query: Dict) -> bool:
        """Kiểm tra phù hợp với lộ trình nghề nghiệp"""
        course_id = result['course_id']
        # Lấy tất cả các cạnh (course_id -> career_node) có relationship == 'career_paths'
        career_edges = [
            (u, v, data)
            for u, v, data in self.graph.graph.edges(course_id, data=True)
            if data.get('relationship') == 'career_paths'
        ]
        # Chuẩn hóa danh sách nghề người dùng quan tâm: ['data_engineer', 'ml_engineer', ...]
        query_careers = [
            c.split(':', 1)[1]
            for c in parsed_query.get('target_concepts', [])
            if c.startswith('career:')
        ]
        # Mỗi edge là (u, v, data): v là node_id kiểu 'career:data_engineer'
        return any(
            v.split(':', 1)[1] in query_careers
            for _, v, _ in career_edges
        )
    def _has_prerequisites(self, result: Dict, parsed_query: Dict) -> bool:
        """Kiểm tra course có prerequisites phù hợp với query"""
        course_id = result['course_id']
        
        # Lấy tất cả prerequisites từ graph
        prerequisites = [
            target 
            for _, target, data in self.graph.graph.out_edges(course_id, data=True)
            if data.get('relationship') == 'prerequisites'
        ]
        
        # Kiểm tra overlap với target concepts
        target_concepts = parsed_query['target_concepts']
        return any(prereq in target_concepts for prereq in prerequisites)

# ======================
# 4. LLM Integration Layer
# ======================
class GraphEnhancementEngine:
    """LLM-powered graph enhancement components"""
    def __init__(self):
        self.cache = Cache("llm_cache")
        self.cache.expire()
        self.semaphore = threading.Semaphore(10)
        self.metrics = {  # Thêm phần khởi tạo metrics
            'llm_calls': 0,
            'cache_hits': 0,
            'processing_time': 0
        }
    def batch_extract_relationships(self, courses: List[Dict]) -> List[Dict]:
        """Tối ưu batch processing với dynamic batch size"""
        results = []
        dynamic_batch_size = max(10, len(courses) // 10)  # Điều chỉnh batch size linh hoạt
        with ThreadPoolExecutor(max_workers=4) as executor:  # Giảm số worker
            for i in range(0, len(courses), dynamic_batch_size):
                batch = courses[i:i+dynamic_batch_size]
                try:
                    batch_result = self._process_batch(batch)
                    results.extend(batch_result)
                except Exception as e:
                    logger.error(f"Batch failed: {str(e)}")
        return results

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Xử lý batch với timeout và giới hạn kích thước prompt"""
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for course in batch:
                future = executor.submit(
                    self._extract_with_cache,
                    course['title'][:150],  # Giới hạn độ dài
                    course['description'][:300],
                    course['knowledge_requirements']['teaches'][:5]
                )
                futures.append(future)
        
        return [f.result() for f in as_completed(futures)]
    @lru_cache(maxsize=1000)
    def analyze_query(self, query: str) -> Dict:
        try:
            prompt = f"""**Educational Query Analysis Task**
            
            As an advanced query understanding system, analyze this learning request through Chain-of-Thought reasoning:

            1. Identify explicit and implicit learning goals
            2. Detect prerequisite relationships and knowledge dependencies
            3. Determine optimal learning path pattern
            4. Extract constraints and preferences
            5. Extract any specific number of courses requested (e.g., '5 courses', 'three', 'top 1/2/3...', 'two first...)
            6. If no specific number of coures requested from user, return default 3.

            **Example Query:**
            "I need 3 courses to learn machine learning basics"

            **Analysis Process:**
            - Core concepts: ["machine_learning"]
            - Implicit dependencies: Basic math -> Machine Learning
            - Path pattern: Sequential prerequisites
            - Constraints: Beginner-friendly
            - num_courses: 3

            **Current Query:** "{query}"

            Output strict JSON format:
            {{
                "concepts": ["list","of","core","concepts"],
                "num_courses": integer | null,
                "relationship_patterns": {{
                    "type": "sequential|parallel|conditional",
                    "confidence": 0.0-1.0
                }},
                "constraints": {{
                    "difficulty": "beginner|intermediate|advanced",
                    "max_duration": months,
                    "min_rating": 0-5,
                    "languages": ["lang1", "lang2"]
                }}
            }}"""

            response = self._safe_llm_call(prompt)
            return {
                "concepts": self._validate_list(response.get("concepts", [])),
                "num_courses": response.get("num_courses"),
                "relationship_patterns": response.get("relationship_patterns", {}),
                "constraints": self._validate_constraints(response.get("constraints", {}))
            }
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return self._fallback_response(query)
    def _fallback_response(self, query: str) -> Dict:
        return {
            "target_concepts": ["data_science"],
            "relationship_pattern": {
                "type": "sequential",
                "confidence": 0.7,
                "allowed_relationships": ['prerequisites', 'prepares_for']
            },
            "constraints": {
                "difficulty": "beginner",
                "max_duration": 12,
                "min_rating": 3.5,
                "languages": []
            }
        }

    def _map_relationships(self, pattern_type: str) -> List[str]:
        relationship_mapping = {
            'sequential': ['prerequisites', 'prepares_for'],
            'parallel': ['complements', 'related_to'],
            'conditional': ['requires', 'depends_on']
        }
        return relationship_mapping.get(pattern_type, ['prerequisites'])
    def _fallback_concepts(self, query: str) -> List[str]:
        """Trích xuất từ khóa fallback khi LLM không trả về kết quả"""
        return [query.strip().lower().replace(" ", "_")]
    def _validate_list(self, items):
        return [item for item in items if isinstance(item, str) and item.strip()]
    
    def _validate_constraints(self, constraints):
        # Xử lý max_duration
        max_duration = constraints.get("max_duration")
        try:
            max_duration = int(max_duration) if max_duration is not None else 12
        except (TypeError, ValueError):
            max_duration = 12
        max_duration = min(max_duration, 24)

        # Xử lý min_rating
        min_rating = constraints.get("min_rating")
        try:
            min_rating = float(min_rating) if min_rating is not None else 3.5
        except (TypeError, ValueError):
            min_rating = 3.5
        min_rating = max(min_rating, 0)

        return {
            "difficulty": constraints.get("difficulty", "intermediate").lower(),
            "max_duration": max_duration,
            "min_rating": min_rating,
            "languages": self._validate_list(constraints.get("languages", []))
        }

        
    def extract_relationships(self, title: str, description: str, teaches: List[str]) -> Dict:
        """Robust relationship extraction with validation"""
        prompt = f"""Strictly follow this JSON format:
        {{
            "prerequisites": ["list"],
            "prepares_for": ["list"],
            "uses": ["list"],
            "complements": ["list"]
        }}
        
        Course: {title[:200]}
        Description: {description[:500]}
        Topics: {', '.join(teaches[:20])}
        
        Extract relationships. Only use exact values from input:"""
        
        response = self._safe_llm_call_with_retry(prompt)
        return self._validate_response(response)

    def _safe_llm_call_with_retry(self, prompt: str, max_retries=3) -> Dict:
        """LLM communication with retries and backoff"""
        from time import sleep
        
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model="qwen3:8b",
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.1}
                )
                parsed = json.loads(response['response'])
                return parsed
            except json.JSONDecodeError:
                logger.warning(f"Retry {attempt+1}: Invalid JSON response")
                sleep(1 ** attempt)
            except Exception as e:
                logger.error(f"LLM Error: {str(e)}")
                sleep(2 ** attempt)
        return {}

    def _validate_response(self, response: Dict) -> Dict:
        """Ensure response structure integrity"""
        valid_keys = ['prerequisites', 'prepares_for', 'uses', 'complements']
        return {
            key: list(filter(lambda x: isinstance(x, str), response.get(key, [])))
            for key in valid_keys
        }
    
    def infer_relationship(self, source: Dict, target: Dict) -> Dict:
        """Infer implicit relationships between nodes"""
        prompt = f"""Analyze potential relationship between:
        Source: {source.get('title', source['id'])} ({source['type']})
        Target: {target.get('title', target['id'])} ({target['type']})
        
        Possible relationships: prepares_for, uses, requires, similar_to, part_of
        
        Output JSON:
        {{
            "relationship": "type",
            "confidence": 0.0-1.0,
            "explanation": "short rationale"
        }}"""
        
        return self._safe_llm_call(prompt)
    
    def _safe_llm_call(self, prompt: str) -> Dict:
        """Robust LLM communication handler"""
        try:
            response = ollama.generate(
                model="qwen3:8b",
                prompt=prompt,
                format="json",
                options={"temperature": 0.3}
            )
            return json.loads(response['response'])
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            return {}
    def _batch_enhance_relationships(self):
        courses = self.data.to_dict('records')
        batch_size = 20
        relations = [None] * len(courses)  # Khởi tạo list cố định
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(0, len(courses), batch_size):
                batch = courses[i:i+batch_size]
                future = executor.submit(
                    self.llm.batch_extract_relationships,
                    batch
                )
                futures.append((i, future))
            
            for start_idx, future in futures:
                try:
                    batch_result = future.result()
                    end_idx = start_idx + len(batch_result)
                    relations[start_idx:end_idx] = batch_result
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
        
        for course, rel in zip(courses, relations):
            self._process_relationships_safe(course['course_id'], rel)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), 
          stop=stop_after_attempt(3))
    def _extract_with_cache(self, title: str, desc: str, teaches: List[str]) -> Dict:
        with self.semaphore:
            cache_key = hashlib.md5(f"{title}{desc}{teaches}".encode()).hexdigest()
            
            if cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key]
            
            try:
                start_time = time.time()
                response = ollama.generate(
                    model="qwen3:8b",
                    prompt=self._create_prompt(title, desc, teaches),
                    format="json",
                    options={"temperature": 0.1}
                )
                self.metrics['llm_calls'] += 1
                self.metrics['processing_time'] += time.time() - start_time
                result = json.loads(response['response'])
                self.cache[cache_key] = result
                return result
            except Exception as e:
                logger.error(f"LLM Error: {str(e)}")
                return {}

    def _create_prompt(self, title: str, desc: str, teaches: List[str]) -> str:
        return f"""
        Strictly use JSON format:
        {{
            "prerequisites": ["..."],
            "prepares_for": ["..."],
            "uses": ["..."],
            "complements": ["..."]
        }}
        
        Course: {title[:200]}
        Description: {desc[:500]}
        Topics: {', '.join(teaches[:10])}
        """
# ======================
# 5. Main System Interface
# ======================
class CourseAdvisor:
    """Complete course recommendation system"""
    
    def __init__(self, data_path: str, use_cache: bool = True):
        start_time = time.time()
        self.knowledge_graph = CourseKnowledgeGraph(data_path, use_cache)
        self.query_parser = MultiHopQueryParser(self.knowledge_graph)
        self.search_engine = GraphAwareSearch(self.knowledge_graph)
        logger.info(f"System initialized in {time.time()-start_time:.2f}s")
    
    def query(self, natural_query: str) -> Dict:
        """End-to-end query processing"""
        try:
            parsed_query = self.query_parser.parse(natural_query)
            results = self.search_engine.search(parsed_query)
            return self._format_response(results, parsed_query)
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return self._error_response()
    
    def _format_response(self, results: List[Dict], parsed_query: Dict) -> Dict:
        """Structure final output"""
        # Determine number of courses to return
        num_courses = parsed_query.get("num_courses")
        
        # Validate and set defaults
        try:
            num_courses = int(num_courses)
            if num_courses <= 0:
                num_courses = 3
            else:
                # Cap the maximum number to prevent excessive results
                num_courses = min(num_courses, 20)
        except (TypeError, ValueError):
            num_courses = 3
        
        # Slice results to the requested number
        top_results = results[:num_courses]
        
        return {
            "results": [self._format_course(r) for r in top_results],
            "metadata": {
                "concepts": parsed_query['target_concepts'],
                "filters": parsed_query['constraints'],
                "result_count": len(results),
                "num_courses_requested": num_courses
            }
        }
    
    def _format_course(self, course: Dict) -> Dict:
        """Prepare course for output"""
        return {
            "title": course['title'],
            "rating": course['rating'],
            "duration": f"{course['duration']} months",
            "difficulty": course['difficulty'].capitalize(),
            "languages": course['languages'],
            "key_topics": self._get_course_topics(course['course_id'])
        }
    
    def _get_course_topics(self, course_id: str) -> List[str]:
        """Retrieve course topics from graph"""
        return [
            n.split(':', 1)[1].replace('_', ' ')
            for n in self.knowledge_graph.graph.neighbors(course_id)
            if n.startswith('concept:')
        ][:5]
    def _error_response(self) -> Dict:
        return {
            "results": [],
            "metadata": {
                "error": "Unable to process query",
                "timestamp": time.time()
            }
        }


# ======================
# 6. Helper Components
# ======================
class MultiHopNavigator:
    """Advanced graph path finder with virtual relationships"""
    
    def __init__(self, graph: CourseKnowledgeGraph):
        self.graph = graph
        self.visited = set()
    
    def find_paths(self, start_nodes: List[str], pattern: Dict) -> List[Dict]:
        """Multi-hop path discovery algorithm"""
        results = []
        max_depth = 3  # Configurable hop limit
        
        for depth in range(1, max_depth + 1):
            current_level = []
            for node in start_nodes:
                if node not in self.visited:
                    self.visited.add(node)
                    neighbors = self._get_qualified_neighbors(node, pattern)
                    current_level.extend(neighbors)
                    
                    if self.graph.graph.nodes[node].get('type') == 'course':
                        path_score = self._calculate_path_score(node, depth)
                        results.append({
                            "course_id": node,
                            "score": path_score,
                            "path_depth": depth
                        })
            start_nodes = current_level
        
        return results
    
    def _get_qualified_neighbors(self, node: str, pattern: Dict) -> List[str]:
        """Retrieve relevant neighbors based on search pattern"""
        neighbors = []
        for neighbor in self.graph.graph.neighbors(node):
            # Check physical relationships
            edge_data = self.graph.graph.get_edge_data(node, neighbor)
            if any(ed['relationship'] in pattern['allowed_relationships'] 
                   for ed in edge_data.values()):
                neighbors.append(neighbor)
            
            # Check virtual relationships
            virtual_rel = self.graph.get_virtual_relationship(node, neighbor)
            if virtual_rel and virtual_rel['relationship'] in pattern['allowed_relationships']:
                neighbors.append(neighbor)
        
        return neighbors
    
class OptimizedMultiHopNavigator(MultiHopNavigator):
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.neighbor_cache = LRUCache(maxsize=10_000)

    def find_paths(self, start_nodes: List[str], max_depth: int = 2) -> List[Dict]:
        results = []
        visited = set()
        queue = deque([(node, 0, []) for node in start_nodes])
        
        while queue:
            node, depth, path = queue.popleft()
            
            if depth > max_depth or node in visited:
                continue
                
            visited.add(node)
            
            if self._is_course_node(node):
                score = 1.0 / (depth + 1)
                results.append({
                    "course_id": node,
                    "score": score,
                    "path": path + [node]
                })
            
            for neighbor in self._get_cached_neighbors(node):
                if neighbor not in path:
                    queue.append((neighbor, depth + 1, path + [node]))
        
        return sorted(results, key=lambda x: -x['score'])

    def _get_cached_neighbors(self, node: str) -> List[str]:
        if node not in self.neighbor_cache:
            neighbors = list(self.graph.neighbors(node))
            self.neighbor_cache[node] = neighbors
        return self.neighbor_cache[node]

    def _is_course_node(self, node: str) -> bool:
        return self.graph.nodes[node].get('type') == 'course'

# if __name__ == "__main__":
#     # Example Usage
#     system = CourseAdvisor("D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\courses_sample_test.json")
    
#     sample_query = ("Give me top 1 machine learning course that suitable for AI engineer under 6 months")
    
#     start_time = time.time()
#     response = system.query(sample_query)
#     processing_time = time.time() - start_time
    
#     print(f"\nRecommended Courses (processed in {processing_time:.2f}s):")
#     for idx, course in enumerate(response['results'], 1):
#         print(f"{idx}. {course['title']}")
#         print(f"   - Topics: {', '.join(course['key_topics'])}")
#         print(f"   - Duration: {course['duration']} | Rating: {course['rating']}/5")
class IntentClassifier:
    def __init__(self):
        self.cache = Cache("intent_cache")
    
    def detect_intent(self, query: str) -> Dict:
        """Classify user intent using LLM"""
        prompt = f"""Classify the user's intent into one of these categories:
        1. course_search - Requests related to course recommendations
        2. chitchat - General conversation/greetings
        
        Classification rules:
        - Contains educational concepts about searching, finding course demand.
        - Asks about system capabilities: "what can you do" → chitchat 
        - Greetings/thanks/goodbye → chitchat
        - Asks about other concept but not chitchat or course searching, response politely that we do not support that in the current period, maybe later.
        - For queries without any clear request (just words or statements), ask user to clarify clearly before decide what they belongs to.
        Examples:
        - "Recommend Python courses" → course_search
        - "How does this work?" → chitchat
        - "Good morning!" → chitchat
        - "deep learning" → need to clarify more from user 
        
        Query: "{query}"
        Output JSON: {{"intent": "course_search|chitchat", "confidence": 0.0-1.0}}"""
        
        response = ollama.generate(
            model="qwen3:8b",
            prompt=prompt,
            format="json"
        )
        return json.loads(response['response'])
def render_courses(courses):
    """Display course recommendations in card layout"""
    for idx, course in enumerate(courses, 1):
        with st.container():
            st.markdown(f"""
            <div class="course-card">
                <h3>{idx}. {course['title']}</h3>
                <table>
                    <tr><td>⭐ Rating:</td><td>{course['rating']}/5</td></tr>
                    <tr><td>⏳ Duration:</td><td>{course['duration']}</td></tr>
                    <tr><td>📊 Difficulty:</td><td>{course['difficulty']}</td></tr>
                    <tr><td>🌐 Languages:</td><td>{', '.join(course['languages'])}</td></tr>
                </table>
                <p>🔑 Key Topics: {', '.join(course['key_topics'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
# QUICK_ACTIONS = [
#     {"label": "🔄 New Chat", "command": "/reset"},
#     {"label": "📚 History", "command": "/history"},
#     {"label": "⭐ Favorites", "command": "/favorites"}
# ]
# def handle_quick_action(command: str):
#     """Xử lý các action nhanh từ button"""
#     if command == "/reset":
#         # Reset conversation
#         st.session_state.messages = []
#         st.success("New Chat Begins Here...")
    
#     elif command == "/history":
#         # Hiển thị lịch sử
#         history = "\n".join([f"{m['role']}: {m['content']}" 
#                            for m in st.session_state.messages])
#         st.session_state.messages.append({
#             "role": "assistant",
#             "type": "text",
#             "content": f"📜 Chat History:\n{history}"
#         })
#         st.rerun()
    
#     elif command == "/favorites":
#         # Hiển thị khóa học yêu thích
#         if "favorites" not in st.session_state:
#             st.session_state.favorites = []
        
#         if st.session_state.favorites:
#             fav_list = "\n".join([f"⭐ {course['title']}" 
#                                 for course in st.session_state.favorites])
#             response = f"Favorite List:\n{fav_list}"
#         else:
#             response = "Favorite list is empty."
        
#         st.session_state.messages.append({
#             "role": "assistant",
#             "type": "text",
#             "content": response
#         })
#         st.rerun()
    
#     else:
#         st.warning(f"Action không xác định: {command}")
def show_quick_actions():
    cols = st.columns(len(QUICK_ACTIONS))
    for col, action in zip(cols, QUICK_ACTIONS):
        if col.button(action["label"]):
            handle_quick_action(action["command"])

# def main():
#     st.set_page_config(page_title="Course Advisor Chatbot", page_icon="🎓")
#     st.title("🎓 AI Course Searching")
#     show_quick_actions()
#     # Settings menu
#     if "user_prefs" not in st.session_state:
#         st.session_state.user_prefs = {
#             "theme": "light",
#             "notification": True
#         }
#     with st.sidebar.expander("⚙️ Setting"):
#         theme = st.selectbox("Theme", ["light", "dark"])
#         st.session_state.user_prefs.update({
#             "theme": theme
#         })
#     # Initialize session state components separately
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # Initialize searching separately to ensure it always exists
#     if "seaching" not in st.session_state:
#         st.session_state.advisor = CourseAdvisor(
#             "D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\courses_sample_test.json",
#             use_cache=True
#         )

#     st.markdown("""
#         <style>
#         .stChatInput textarea {
#             resize: vertical !important;
#             min-height: 40px !important;
#             max-height: 200px !important;
#             padding-right: 25px !important;  /* Tạo khoảng trống cho handle */
#         }
        
#         /* Custom resize handle */
#         .stChatInput textarea:after {
#             content: "";
#             position: absolute;
#             bottom: 2px;
#             right: 2px;
#             width: 12px;
#             height: 12px;
#             background: 
#                 linear-gradient(45deg, 
#                     transparent 0%, 
#                     transparent 50%, 
#                     #666 50%, 
#                     #666 100%
#                 );
#             cursor: se-resize;
#         }
        
#         /* Hiệu ứng hover */
#         .stChatInput textarea:hover:after {
#             background: 
#                 linear-gradient(45deg, 
#                     transparent 0%, 
#                     transparent 50%, 
#                     #444 50%, 
#                     #444 100%
#                 );
#         }
#         </style>
#         """, unsafe_allow_html=True)

#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             if message["type"] == "text":
#                 st.markdown(message["content"])
#             elif message["type"] == "courses":
#                 render_courses(message["content"])

#     # Handle user input
#     if prompt := st.chat_input("Describe your learning goals (e.g. 'I need 3 courses about machine learning for beginners'):"):
#         st.session_state.messages.append({
#             "role": "user", 
#             "type": "text",
#             "content": prompt
#         })

#         with st.spinner("Analyzing you request..."):
#             #custom_spinner("Analyzing you request...")
#             try:
#                 response = st.session_state.advisor.query(prompt)
#                 if response["results"]:
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "type": "courses",
#                         "content": response["results"]
#                     })
#                 else:
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "type": "text",
#                         "content": "🔍 No courses found matching your criteria. Try broadening your search."
#                     })
#             except Exception as e:
#                 st.session_state.messages.append({
#                     "role": "assistant",
#                     "type": "text",
#                     "content": f"⚠️ Error processing request: {str(e)}"
#                 })

#         st.rerun()


def process_user_input(query: str) -> Dict:
    """Main processing router"""
    classifier = IntentClassifier()
    intent = classifier.detect_intent(query)
    
    if intent.get('intent') == "course_search":
        return handle_course_search(query)
    return handle_chitchat(query)

def handle_course_search(query: str) -> Dict:
    """Process course-related queries"""
    try:
        response = st.session_state.advisor.query(query)
        return {
            "type": "courses",
            "content": response["results"]
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return {
            "type": "text",
            "content": "⚠️ Couldn't process your request. Please try different keywords."
        }

def handle_chitchat(query: str) -> Dict:
    """Handle general conversation"""
    prompt = f"""You are a friendly AI assistant for an educational platform, focused on course searching. Respond naturally in English.
    
    Guidelines:
    1. Keep responses concise 
    2. Redirect educational questions to course search
    3. Be polite but professional
    
    Examples:

    User: "What's the weather?"
    Assistant: "I specialize in course recommendations. Would you like help finding learning resources?"
    
    Current query: "{query}"
    """
    
    response = ollama.generate(
        model="qwen3:8b",
        prompt=prompt,
        options={"temperature": 0.8}
    )
    return {
        "type": "text",
        "content": response['response']
    }

# ======================
# 3. Updated UI Components
# ======================
QUICK_ACTIONS = [
    {"label": "🔄 New Session", "command": "/reset"},
    {"label": "📚 History", "command": "/history"},
    {"label": "⭐ Saved Items", "command": "/favorites"}
]

def handle_quick_action(command: str):
    """Handle quick action buttons"""
    if command == "/reset":
        st.session_state.messages = []
        st.success("New session started!")
    
    elif command == "/history":
        history = "\n".join([f"{m['role']}: {m['content']}" 
                           for m in st.session_state.messages])
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": f"📜 Conversation History:\n{history}"
        })
    
    elif command == "/favorites":
        if st.session_state.get('favorites'):
            fav_list = "\n".join([f"⭐ {item['title']}" 
                                for item in st.session_state.favorites])
            response = f"Your saved items:\n{fav_list}"
        else:
            response = "Your favorites list is empty."
        
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": response
        })

# ======================
# 4. Main Application Flow
# ======================
def main():
    st.set_page_config(page_title="EduAssistant", page_icon="🎓")
    st.title("🎓 Course Finder")
    
    # Initialize core components
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "advisor" not in st.session_state:
        st.session_state.advisor = CourseAdvisor(
            "D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\courses_sample_test.json", 
            use_cache=True
        )
    
    # Render UI
    render_sidebar_settings()
    show_quick_actions()
    display_chat_history()
    process_chat_input()

def render_sidebar_settings():
    """User preferences panel"""
    with st.sidebar.expander("⚙️ Preferences"):
        st.selectbox("Theme", ["light", "dark"], key="theme_preference")
        st.checkbox("Enable Notifications", True, key="notifications")

def display_chat_history():
    """Show conversation history"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "courses":
                render_courses(msg["content"])

def process_chat_input():
    """Handle user input"""
    if prompt := st.chat_input("Ask about courses or chat with me..."):
        st.session_state.messages.append({
            "role": "user", 
            "type": "text",
            "content": prompt
        })

        with st.spinner("Analyzing..."):
            try:
                response = process_user_input(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": response["type"],
                    "content": response["content"]
                })
            except Exception as e:
                handle_processing_error(e)
        
        st.rerun()

def handle_processing_error(error: Exception):
    """Error handling"""
    logger.error(f"System error: {str(error)}")
    st.session_state.messages.append({
        "role": "assistant",
        "type": "text",
        "content": "🔧 System maintenance in progress. Please try again later."
    })

if __name__ == "__main__":
    main()