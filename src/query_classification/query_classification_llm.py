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
from collections import deque 

logging.basicConfig(level=logging.DEBUG)  # You can adjust to INFO or ERROR in production
logger = logging.getLogger(__name__)


# ======================
# 1. Generalized Knowledge Base
# ======================
class KnowledgeBase:
    def __init__(self, data_path: str):
        self.data = self._load_and_preprocess(data_path)
        self.graph = self._build_knowledge_graph()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_indices()
        self._add_external_knowledge()
        self._validate_data()
        
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
        print("RAW RECORD 0:", raw_data.iloc[0].to_dict())
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
        """Enhanced prerequisite matching"""
        for prereq in prerequisites:
            # Find courses that teach this prerequisite or its sub-concepts
            for _, row in self.data.iterrows():
                if row['course_id'] == course_id:
                    continue
                    
                # Check hierarchical concepts
                course_topics = row['features'].get('topics', [])
                if any(prereq in topic or prereq in topic.split('::') 
                    for topic in course_topics):
                    graph.add_edge(
                        course_id, 
                        row['course_id'],
                        relationship='prerequisite',
                        weight=0.85,
                        label=f"Requires {prereq}"
                    )
# ======================
# 2. Core LLM Integration
# ======================
class LLMAssistant:
    def __init__(self):
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    def generate(self, prompt: str, model: str = "qwen2.5:1.5b") -> dict:
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.4}
            )
            return json.loads(response['response'])
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return {}

# ======================
# 3. Adaptive Query Processing
# ======================
class QueryProcessor:
    PROMPT_TEMPLATE = """
    Analyze the learning-related query and identify core components:
    
    Query: {query}
    
    Available Context Types:
    - Topics (e.g., programming, business)
    - Skills (e.g., Python, project management)
    - Difficulty Levels (beginner, intermediate, advanced)
    - Learning Styles (self-paced, instructor-led)
    - Career Paths (data science, web development)
    - Certifications (AWS, PMP)
    - Duration Constraints
    - Prerequisite Requirements
    
    Output JSON structure:
    {{
        "components": {{
            "learning_goals": [],
            "constraints": {{
                "difficulty": "",
                "duration_max": null,
                "required_prerequisites": [],
                "excluded_topics": []
            }},
            "preferences": {{
                "learning_style": "",
                "certifications": []
            }},
            "career_connections": []
        }},
        "search_strategy": {{
            “primary_approach”: (semantic | graph | hybrid)
            "fallback_approaches": [],
            "reasoning": "Step-by-step explanation of the analysis"
        }}
    }}
    Analyze the learning-related query and identify core components using these examples:

    Example 1:
    Query: "I want to switch to data science but only have basic math skills"
    Output:
    {{
    "components": {{
        "learning_goals": ["data science fundamentals"],
        "constraints": {{
        "difficulty": "beginner",
        "required_prerequisites": ["basic math"],
        "excluded_topics": ["advanced statistics"]
        }},
        "preferences": {{
        "learning_style": "hands-on"
        }},
        "career_connections": ["data science career transition"]
    }},
    "search_strategy": {{
        "primary_approach": "graph",
        "fallback_approaches": ["semantic", "constraint-based"],
        "reasoning": "Focus on career transition paths while filtering out advanced math requirements"
    }}
    }}

    Example 2:
    Query: "Advanced cloud security courses with AWS certification prep under 3 months"
    Output:
    {{
    "components": {{
        "learning_goals": ["cloud security", "AWS certification"],
        "constraints": {{
        "duration_max": 3,
        "required_prerequisites": ["basic cloud computing"]
        }},
        "preferences": {{
        "certifications": ["AWS Certified Security"]
        }},
        "career_connections": ["cloud security engineer"]
    }},
    "search_strategy": {{
        "primary_approach": "certification-based",
        "fallback_approaches": ["duration-filtered", "skill-graph"],
        "reasoning": "Prioritize certification-aligned content with strict timeline constraints"
    }}
    }}

    Now analyze this new query:
    Query: {query}

    Output JSON structure:
    {{
        "components": {{
            "learning_goals": [],
            "constraints": {{
                "difficulty": "",
                "duration_max": null,
                "required_prerequisites": [],
                "excluded_topics": []
            }},
            "preferences": {{
                "learning_style": "",
                "certifications": []
            }},
            "career_connections": []
        }},
        "search_strategy": {{
            "primary_approach": "",
            "fallback_approaches": [],
            "reasoning": "Step-by-step explanation of the analysis"
        }}
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        
    def analyze_query(self, query: str) -> dict:
        """Add this missing method implementation"""
        prompt = self.PROMPT_TEMPLATE.format(query=query)
        analysis = self.llm.generate(prompt)
        
        return self._normalize_analysis(analysis)
    def _normalize_analysis(self, raw: dict) -> dict:
        """Ensure consistent structure for downstream processing"""
        normalized = {
            'components': raw.get('components', {}),
            'strategy': raw.get('search_strategy', {})
        }
        
        # Normalize constraint values
        constraints = normalized['components'].get('constraints', {})
        if 'duration_max' in constraints:
            try:
                constraints['duration_max'] = int(constraints['duration_max'])
            except:
                constraints['duration_max'] = None
                
        return normalized

# ======================
# 4. Flexible Search System
# ======================
class AdaptiveSearcher:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        topics_series = self.kb.data['features'].apply(lambda f: f.get('topics', []))
        self._concept_list = topics_series.explode().dropna().unique().tolist()

        # Encode tất cả once và lưu FAISS index
        concept_embs = self.kb.encoder.encode(self._concept_list, show_progress_bar=False)
        self._concept_index = faiss.IndexFlatIP(concept_embs.shape[1])
        self._concept_index.add(np.array(concept_embs, dtype='float32'))

    def _ground_concepts(self, term: str, top_k: int = 5) -> List[str]:
        """
        Dùng LLM + embedding để map một chuỗi tự do vào
        một loạt khái niệm/chủ đề chuẩn có trong KB.
        """
        # 1) LLM đề xuất các synonym hoặc phrase chuẩn
        try:
            llm_resp = self.llm.generate(
                f"List 5 canonical course topics for '{term}' in learning context"
            )
            expansions = llm_resp.get('expansions', [])
        except Exception:
            expansions = []
        
        # 2) Embedding-based nearest neighbor: so sánh với all_concepts
        #    Encode term + expansions
        candidates = list(set([term] + expansions))
        cand_embs = self.kb.encoder.encode(candidates, show_progress_bar=False)
        cand_embs = np.array(cand_embs, dtype='float32')
        
        # 3) FAISS search trên index của all_concepts
        #    Mỗi dòng trả về top_k chỉ số
        D, I = self._concept_index.search(cand_embs, top_k)
        
        # 4) Thu thập các khái niệm gần nhất
        grounded = set()
        for neigh_idxs in I:
            for idx in neigh_idxs:
                grounded.add(self._concept_list[idx])
        
        # 5) Kết hợp cả expansions và các khái niệm từ NN
        grounded = list(grounded.union(expansions))
        
        return grounded
        
    def execute_search(self, query_analysis: dict) -> list:
        strategy = query_analysis['strategy']['primary_approach']
        
        if strategy == 'semantic':
            results = self._semantic_search(query_analysis)
        elif strategy in ('graph', 'topic-based'):
            results = self._graph_search(query_analysis)
        else:
            results = self._hybrid_search(query_analysis)
        filtered = self._apply_filters(results, query_analysis['components'])
        return self._rerank_results(filtered, query_analysis)

        
    def _hybrid_search(self, analysis: dict) -> list:
        semantic_results = self._semantic_search(analysis)
        graph_results = self._graph_search(analysis)
        
        # Merge and deduplicate
        seen = set()
        merged = []
        for result in semantic_results + graph_results:
            if result['course_id'] not in seen:
                seen.add(result['course_id'])
                merged.append(result)
        
        return merged
    
    def _semantic_search(self, analysis: dict) -> list:
        """
        1) Grounding learning_goals thành danh sách concept chuẩn
        2) Tìm semantic trên tất cả grounded terms
        3) Áp dụng filter + rerank
        """
        # 1. Ground each learning goal
        grounded = []
        for goal in analysis['components']['learning_goals']:
            grounded += self._ground_concepts(goal)
        # bao gồm cả career_connections nếu có
        grounded += analysis['components'].get('career_connections', [])
        grounded = list(set(grounded))

        # 2. Build the search text
        search_text = ' '.join(grounded)

        # 3. Semantic search
        raw_results = self.kb.semantic_search(search_text, top_k=50)

        # 4. Apply hard constraints & soft preferences
        filtered = self._apply_filters(raw_results, analysis['components'])

        # 5. Rerank and return
        return self._rerank_results(filtered, analysis)


    def _graph_search(self, analysis: dict) -> list:
        """
        1) Grounding learning_goals thành canonical concept list
        2) BFS multihop từ các concept node tương ứng
        3) Collect course nodes, dedupe, filter & rerank
        """
        components = analysis['components']
        # Ground the goals
        grounded = []
        for goal in components['learning_goals']:
            grounded += self._ground_concepts(goal)
        grounded = list(set(grounded))

        max_hops = self._determine_max_hops(components)
        min_weight = 0.6
        raw_results = []

        # BFS cho từng grounded concept
        for concept in grounded:
            # build list of starting concept-nodes
            start_nodes = [
                node for node, data in self.kb.graph.nodes(data=True)
                if data.get('type') == 'concept'
                   and concept.lower().replace(' ', '_') in node
            ]
            visited = set()
            queue = deque([(n, [], 0) for n in start_nodes])

            while queue:
                node_id, path, depth = queue.popleft()
                if depth > max_hops:
                    continue

                # Nếu gặp course node → lưu kết quả
                if self.kb.graph.nodes[node_id].get('type') == 'course':
                    course = self.kb.data.loc[
                        self.kb.data['course_id'] == node_id
                    ].iloc[0].to_dict()
                    course['path_score'] = self._calculate_path_score(path)
                    raw_results.append(course)
                    continue

                # Duyệt neighbors
                for nbr in self.kb.graph.neighbors(node_id):
                    if nbr in visited:
                        continue
                    edges = self.kb.graph.get_edge_data(node_id, nbr)
                    # chỉ qua các edge 'teaches' hoặc 'leads_to' đủ weight
                    if any(
                        ed.get('relationship') in ('teaches','leads_to')
                        and ed.get('weight', 0) >= min_weight
                        for ed in edges.values()
                    ):
                        visited.add(nbr)
                        queue.append((nbr, path + [(node_id, nbr)], depth + 1))

        # 4. Dedupe
        unique = self._deduplicate_results(raw_results)

        # 5. Apply filters & rerank
        filtered = self._apply_filters(unique, components)
        return self._rerank_results(filtered, analysis)
    # Các hàm hỗ trợ mới cần thêm
    def _determine_max_hops(self, components):
        """Xác định độ sâu tối đa dựa trên độ phức tạp truy vấn"""
        base_hops = 2
        complexity_factors = [
            len(components['learning_goals']),
            len(components.get('constraints', {})),
            len(components.get('prerequisites', []))
        ]
        return min(base_hops + sum(complexity_factors), 4)

    def _expand_concept_hierarchy(self, term):
        """Mở rộng hệ thống phân cấp khái niệm với semantic expansion"""
        expanded_terms = self.llm.generate(
            f"Expand concept '{term}' into related sub-concepts for educational search"
        ).get('expansions', [])
        
        return [
            f"concept:{sub_concept.lower().replace(' ', '_')}"
            for sub_concept in [term] + expanded_terms
        ]

    def _validate_node_relevance(self, node_id, query_term):
        """Validate node với LLM để lọc nhiễu"""
        node_info = self.kb.graph.nodes[node_id]
        validation = self.llm.generate(
            f"Validate node relevance:\n"
            f"Node Type: {node_info.get('type')}\n"
            f"Node Content: {node_info}\n"
            f"Search Term: {query_term}\n"
            "Return JSON: {'relevant': bool, 'reason': str}"
        )
        return validation.get('relevant', False)

    def _calculate_path_score(self, path):
        """Tính điểm đường đi dựa trên các yếu tố"""
        score = 0.0
        for source, target in path:
            edge_data = self.kb.graph.get_edge_data(source, target)
            max_weight = max(edata['weight'] for edata in edge_data.values())
            score += max_weight * 0.7
            
            # Ưu tiên đường đi ngắn hơn
            score -= 0.1 * len(path)
        
        return max(0.0, min(1.0, score))

    def _deduplicate_results(self, results):
        """Xử lý trùng lặp với semantic similarity"""
        seen = set()
        unique_results = []
        
        for result in results:
            course_id = result['course_id']
            if course_id not in seen:
                seen.add(course_id)
                unique_results.append(result)
            else:
                # Cập nhật score nếu có đường đi tốt hơn
                existing = next(r for r in unique_results if r['course_id'] == course_id)
                if result['path_score'] > existing['path_score']:
                    existing.update(result)
        
        return unique_results
    
    def _apply_filters(self, results: list, components: dict) -> list:
        # Dynamic constraint application
        filtered = []
        constraints = components['constraints']
        prefs = components['preferences']
        
        for course in results:
            # Check hard constraints
            if not self._meets_constraints(course, constraints):
                continue
                
            # Check preferences
            course['score'] = self._calculate_preference_score(course, prefs)
            
            filtered.append(course)
            
        return filtered
    
    def _calculate_preference_score(self, course: dict, prefs: dict) -> float:
        # Score based on preference matches
        score = 0.0
        if course['learning_style'] == prefs.get('learning_style', ''):
            score += 0.3
        if any(cert in course.get('certifications', []) for cert in prefs.get('certifications', [])):
            score += 0.2
        return score
    def _rerank_results(self, results: list, analysis: dict) -> list:
        """Multi-factor ranking with KG context"""
        ranked = []
        for course in results:
            features = course['features']
            course_id = course['course_id']
            
            # Calculate score components
            semantic_score = self._calculate_semantic_match(course, analysis)
            graph_score = self.kb.graph.nodes[course_id].get('centrality', 0)
            freshness_score = np.log(time.time() - pd.to_datetime(course.get('created', 0)).timestamp())
            
            # Combine scores
            total_score = (
                0.4 * semantic_score +
                0.3 * graph_score +
                0.2 * freshness_score +
                0.1 * course.get('rating', 0)/5.0
            )
            
            ranked.append((total_score, course))
            
        return [x[1] for x in sorted(ranked, reverse=True)]

    def _calculate_semantic_match(self, course, analysis):
        """Context-aware similarity scoring"""
        query_embed = self.kb.encoder.encode(
            ' '.join(analysis['components']['learning_goals'])
        )
        course_embed = self.kb.encoder.encode(course['search_text'])
        return np.dot(query_embed, course_embed)

# ======================
# 5. Validation & Explanation
# ======================
class ResultValidator:
    VALIDATION_PROMPT = """
    Assess the relevance of a learning resource against the original query:
    
    Original Query: {query}
    Course Details: {course_info}
    
    Evaluation Criteria:
    1. Alignment with stated learning goals
    2. Appropriate difficulty level
    3. Relevance to mentioned career paths
    4. Compatibility with preferred learning style
    5. Certification requirements (if any)
    
    Output JSON:
    {{
        "relevance_score": 0.0-1.0,
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }}
    
    Good Match Example:
    Query: "Project management course for software teams"
    Course: {{
    "title": "Agile Team Leadership",
    "topics": ["scrum", "team coordination"],
    "career_paths": ["tech project manager"],
    "learning_style": "case-study based"
    }}
    Validation:
    {{
    "relevance_score": 0.92,
    "strengths": [
        "Direct alignment with team management focus",
        "Practical case studies suitable for software teams"
    ],
    "weaknesses": [
        "Limited coverage of traditional PM methodologies"
    ],
    "recommendations": [
        "Supplement with waterfall methodology resources"
    ]
    }}

    Partial Match Example:  
    Query: "Python for financial analysis"
    Course: {{
    "title": "Data Science Basics",
    "topics": ["python", "statistics"],
    "career_paths": ["general data analysis"]
    }}
    Validation:
    {{
    "relevance_score": 0.65,
    "strengths": [
        "Strong Python fundamentals",
        "Statistical analysis coverage"
    ],
    "weaknesses": [
        "No financial domain-specific content",
        "Lacks time series analysis modules"
    ],
    "recommendations": [
        "Combine with financial modeling specialization course"
    ]
    }}

    Now evaluate:
    Query: {query}
    Course Details: {course_info}

    Output JSON:
    {{
        "relevance_score": 0.0-1.0,
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }}"""

    # In LearningAssistant summary generation
    SUMMARY_PROMPT = """
    Generate summaries using these examples:

    Example 1:
    Query: "UI/UX design courses with portfolio projects"
    Summary:
    {{
    "summary": "Top courses focus on practical design systems and portfolio development, though some lack advanced prototyping coverage.",
    "key_insights": [
        "Best match: 'Design Studio' course (4.8★) offers real client projects",
        "Consider adding complementary prototyping workshops"
    ],
    "alternatives": [
        "Digital Design Fundamentals + Advanced Prototyping bundle"
    ],
    "next_steps": [
        "Compare mentorship options",
        "Review portfolio requirements"
    ]
    }}

    Example 2:
    Query: "Ethical hacking certification prep under 6 months"
    Summary:
    {{
    "summary": "Certification-aligned programs found with hands-on labs, but require existing networking knowledge.",
    "key_insights": [
        "Top pick: 'Cybersecurity Bootcamp' includes exam voucher",
        "Check prerequisite networking modules"
    ],
    "alternatives": [
        "Self-paced CEH study path"
    ],
    "next_steps": [
        "Verify certification exam dates",
        "Assess lab environment requirements"
    ]
    }}

    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        
    def validate_result(self, course_id: str, query: str) -> dict:
        course = self.kb.data[self.kb.data['course_id'] == course_id].iloc[0]
        graph_context = self._get_graph_context(course_id)
        prompt = self.VALIDATION_PROMPT.format(
            query=query,
            course_info=json.dumps(course['features']),
            graph_context=graph_context
        )
        return self.llm.generate(prompt)
    
    def _get_graph_context(self, course_id: str) -> str:
        """Extract relevant graph neighborhood"""
        neighbors = []
        for neighbor in nx.neighbors(self.kb.graph, course_id):
            neighbors.append({
                'node': neighbor,
                'type': self.kb.graph.nodes[neighbor].get('type'),
                'relationship': self.kb.graph.edges[(course_id, neighbor)].get('relationship')
            })
        return json.dumps(neighbors[:5])

# ======================
# 6. Main Chatbot Class
# ======================
class LearningAssistant:
    def __init__(self, data_path: str):
        self.kb = KnowledgeBase(data_path)
        self.processor = QueryProcessor(self.kb)
        self.searcher = AdaptiveSearcher(self.kb)
        self.validator = ResultValidator(self.kb)
        
    def process_query(self, query: str) -> dict:
        response = {
            "query": query,
            "analysis": {},
            "results": {
                "courses": [],
                "summary": {
                    "overview": "",
                    "top_recommendations": [],
                    "considerations": []
                }
            },
            "metrics": {
                "processing_time": 0,
                "results_count": 0
            }
        }
        
        try:
            start_time = time.time()
            # Step 1: Query Understanding
            analysis = self.processor.analyze_query(query)
            response['analysis'] = analysis
            
            # Step 2: Search Execution
            results = self.searcher.execute_search(analysis)
            
            # Step 3: Result Validation
            # In LearningAssistant.process_query
            validated = []
            threshold = 0.20
            for course in results[:20]:  # Increased from 10
                validation = self.validator.validate_result(course['course_id'], query)
                
                if len(validated) < 3 and validation.get('relevance_score', 0) >= (threshold - 0.1):
                    validated.append({**course, "validation": validation})
                if not validated:
                    validated = results[:5]  # fallback top-5 thô
            # Step 4: Generate Summary
            response['results']['courses'] = validated
            response['results']['summary'] = self._generate_summary(validated, query)
            # Final formatting
            response['metrics'] = {
                "processing_time": time.time() - start_time,
                "results_count": len(validated)
            }
            
            # Ensure summary structure
            if 'summary' not in response['results']:
                response['results']['summary'] = {
                    "overview": "No summary generated",
                    "top_recommendations": [],
                    "considerations": []
                }
            print(f"\nSearch Stats: {len(results)} initial results, {len(validated)} after validation")
                
        except Exception as e:
            response['error'] = str(e)
            response['traceback'] = traceback.format_exc()
        
        return response

    def _generate_summary(self, results: list, query: str) -> dict:
        summary_prompt = f"""Generate a natural language summary for search results:
        
        Query: {query}
        Top Results: {json.dumps(results[:3])}
        
        Include:
        - Overall match quality
        - Key strengths of top results
        - Potential limitations to note
        - Alternative suggestions if relevant
        
        Output JSON:
        {{
            "summary": "...",
            "key_insights": [],
            "alternatives": [],
            "next_steps": []
        }}"""
        
        try:
            summary = self.validator.llm.generate(summary_prompt)
            return summary if isinstance(summary, dict) else {}
        except Exception as e:
            print(f"Summary generation failed: {str(e)}")
            return {}

# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    assistant = LearningAssistant("D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\processed_courses_detail.json")
    
    queries = [
        # "I need to learn cloud computing for a career switch to IT, prefer courses with hands-on projects",
        # "Looking for advanced machine learning courses that don't require heavy math background",
        # "Recommend short creative writing courses with peer feedback opportunities"
        "I need to learn math basic."
    ]
    
    for query in queries:
        print(f"\n{'='*60}\nProcessing query: {query}\n{'='*60}")
        start_time = datetime.now()
        result = assistant.process_query(query)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\nAnalysis ({duration:.2f}s):")
        print(json.dumps(result['analysis'], indent=2))
        
        print("\nTop Results:")
        if 'summary' in result['results']:
            summary = result['results']['summary']
            print(f"Summary: {summary.get('summary', 'No summary available')}")
            
            for idx, insight in enumerate(summary.get('key_insights', [])[:3]):
                print(f"{idx+1}. {insight}")
                
            print("\nCourses:")
            for idx, course in enumerate(result['results']['courses'][:3]):
                print(f"{idx+1}. {course.get('title')} | Score: {course.get('validation', {}).get('relevance_score', 0):.2f}")