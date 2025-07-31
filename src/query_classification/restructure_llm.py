import json
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import ollama
from typing import *
from functools import lru_cache
from collections import defaultdict, deque
import logging
import time
import difflib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# 1. Knowledge Base
# ======================
class KnowledgeBase:
    def __init__(self, data_path: str):
        self.data = self._load_data(data_path)
        self.graph = self._build_knowledge_graph()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_indices()
        logger.info("Knowledge base initialized with %d courses", len(self.data))

    def _load_data(self, path: str) -> pd.DataFrame:
        with open(path) as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)

        def safe_req(req_dict, key):
            """
            If req_dict is None or missing, returns [].
            Otherwise returns req_dict.get(key, []).
            """
            if not isinstance(req_dict, dict):
                return []
            return req_dict.get(key, []) or []

        # Apply normalization
        df['knowledge_requirements'] = df.apply(lambda x: {
            'teaches': [t.strip().lower() for t in safe_req(x.get('knowledge_requirements'), 'teaches')],
            'prerequisites': [p.strip().lower() for p in safe_req(x.get('knowledge_requirements'), 'prerequisites')]
        }, axis=1)

        df['learning_path'] = df.apply(lambda x: {
            'difficulty': (
                (x.get('learning_path') or {}).get('suitable_for', ['intermediate'])[0]
            ).lower(),
            'career_paths': [
                c.strip().lower() 
                for c in ((x.get('learning_path') or {}).get('career_paths') or [])
            ]
        }, axis=1)

        df['search_text'] = df.apply(lambda x: ' '.join([
            x['title'] or '',
            x['description'] or '',
            ' '.join(x['knowledge_requirements']['teaches']),
            ' '.join(x['learning_path']['career_paths'])
        ]), axis=1)

        return df

    def _build_knowledge_graph(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        
        try:
            # Add course nodes
            for _, row in self.data.iterrows():
                G.add_node(row['course_id'], 
                         type='course',
                         title=row['title'],
                         duration=row['duration_months'],
                         difficulty=row['learning_path']['difficulty'],
                         rating=row['rating'],
                         careers=row['learning_path']['career_paths'])
                
                # Connect teaches relationships
                for concept in row['knowledge_requirements']['teaches']:
                    self._add_concept_hierarchy(G, row['course_id'], concept)
                    
                # Connect prerequisites
                for prereq in row['knowledge_requirements']['prerequisites']:
                    prereq_id = f"prereq:{prereq.replace(' ', '_')}"
                    G.add_edge(row['course_id'], prereq_id, relationship='requires')
                
                # Connect career paths
                for career in row['learning_path']['career_paths']:
                    career_id = f"career:{career.replace(' ', '_')}"
                    G.add_node(career_id, type='career')
                    G.add_edge(row['course_id'], career_id, relationship='leads_to')
            
            logger.info("Built knowledge graph with %d nodes and %d edges", 
                      len(G.nodes), len(G.edges))
            return G
            
        except Exception as e:
            logger.error("Knowledge graph construction failed: %s", str(e))
            raise

    def _add_concept_hierarchy(self, graph: nx.MultiDiGraph, course_id: str, concept: str):
        # 1) Create a single full‐phrase node
        phrase_node = f"concept:{concept.replace(' ', '_')}"
        if not graph.has_node(phrase_node):
            graph.add_node(phrase_node, type='concept')
        # Link the course → full phrase
        graph.add_edge(course_id, phrase_node, relationship='teaches', weight=0.9)

        # 2) Optionally break into sub‐concepts for hierarchy
        # parent = phrase_node
        # for part in concept.split():
        #     part_node = f"concept:{part.lower()}"
        #     if not graph.has_node(part_node):
        #         graph.add_node(part_node, type='concept')
        #     graph.add_edge(parent, part_node, relationship='subconcept', weight=0.7)
        #     parent = part_node



    def _build_indices(self):
        try:
            # Semantic index
            course_embeddings = self.encoder.encode(self.data['search_text'].tolist())
            self.semantic_index = faiss.IndexFlatIP(course_embeddings.shape[1])
            self.semantic_index.add(course_embeddings.astype('float32'))
            
            # Graph indices
            self.concept_map = defaultdict(list)
            self.career_map = defaultdict(list)
            for _, row in self.data.iterrows():
                for concept in row['knowledge_requirements']['teaches']:
                    self.concept_map[concept].append(row['course_id'])
                for career in row['learning_path']['career_paths']:
                    self.career_map[career].append(row['course_id'])
            
            logger.info("Built semantic and graph indices")
        
        except Exception as e:
            logger.error("Index building failed: %s", str(e))
            raise
class GraphRAGProcessor:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._build_hierarchy()
    def _build_hierarchy(self):
        # cluster all courses under each career-node
        self.clusters = defaultdict(list)
        for node, data in self.kb.graph.nodes(data=True):
            if data.get('type')=='career':
                for course in nx.ancestors(self.kb.graph, node):
                    if self.kb.graph.nodes[course]['type']=='course':
                        self.clusters[node].append(course)
        # embed each cluster’s titles
        texts = [' '.join(self.kb.graph.nodes[c]['title'] for c in v)
                 for v in self.clusters.values()]
        embs = self.kb.encoder.encode(texts).astype('float32')
        self.cluster_index = faiss.IndexFlatIP(embs.shape[1])
        self.cluster_index.add(embs)
    def retrieve_context(self, query:str)->List[str]:
        q_emb = self.kb.encoder.encode([query]).astype('float32')
        _, idxs = self.cluster_index.search(q_emb, 3)
        keys = list(self.clusters.keys())
        out = []
        for i in idxs[0]:
            out += self.clusters[keys[i]]
        return list(set(out))

# ======================
# 2. Query Processing
# ======================
class QueryParser:
    def __init__(self):
        self.llm = LLMAssistant()
        self.cache = defaultdict(dict)

    def parse(self, query: str) -> Tuple[List[str], dict]:
        try:
            if query in self.cache:
                return self.cache[query]
            
            prompt = f"""Analyze this learning query and extract key components:
            {query}
            
            Output JSON format:
            {{
                "concepts": ["list of core educational concepts"],
                "filters": {{
                    "difficulty": ["beginner/intermediate/advanced"],
                    "max_duration": maximum months,
                    "min_rating": minimum rating (0-5),
                    "languages": ["English/Vietnamese/..."],
                    "career_paths": ["target careers"]
                }}
            }}"""
            
            response = self.llm.generate(prompt)
            concepts = [c.lower() for c in response.get('concepts', [])]
            filters = response.get('filters', {})
            
            # Normalize filters
            if 'difficulty' in filters:
                filters['difficulty'] = [d.lower() for d in filters['difficulty']]
            if 'languages' in filters:
                filters['languages'] = [lang.lower() for lang in filters['languages']]
            if 'career_paths' in filters:
                filters['career_paths'] = [c.lower() for c in filters['career_paths']]
            
            self.cache[query] = (concepts, filters)
            
            return concepts, filters
            
        except Exception as e:
            logger.error("Query parsing failed: %s", str(e))
            return [], {}

# ======================
# 3. Search Engine
# ======================
class HybridSearchEngine:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        self.beam_width = 50
        logger.info("Hybrid search engine initialized")

        # --- NEW: prepare concept‐node FAISS index ---
        # 1) Only full-phrase concept nodes (those you linked directly from courses)
        self.valid_concept_nodes = [
            f"concept:{c.replace(' ', '_')}"
            for c in self.kb.concept_map.keys()
        ]
        # 2) Corresponding human-readable texts
        self.valid_concept_texts = [c for c in self.kb.concept_map.keys()]

        # 3) Encode & index them
        texts = self.valid_concept_texts  # e.g. ["decision trees", "linear regression", ...]
        embs = self.kb.encoder.encode(texts, show_progress_bar=False).astype('float32')
        self.concept_index = faiss.IndexFlatIP(embs.shape[1])
        self.concept_index.add(embs)

    

    def _ground_concepts(self, llm_concepts: List[str], top_k: int = 3) -> List[str]:
        seeds = []
        # Precompute the underscore‐free labels for fuzzy matching
        labels = [text.replace(' ', '_') for text in self.valid_concept_texts]

        for c in llm_concepts:
            norm = c.lower().strip()
            phrase = norm.replace(' ', '_')
            exact = f"concept:{phrase}"
            # 1) exact
            if exact in self.valid_concept_nodes:
                seeds.append(exact)
                continue

            # 2) fuzzy string match
            matches = difflib.get_close_matches(phrase, labels, n=2, cutoff=0.7)
            if matches:
                for m in matches:
                    seeds.append(f"concept:{m}")
                continue

            # 3) embedding‐based fallback
            q_emb = self.kb.encoder.encode([norm]).astype('float32')
            D, I = self.concept_index.search(q_emb, top_k)
            for idx in I[0]:
                seeds.append(self.valid_concept_nodes[idx])

        return list(set(seeds))



    def search(self, concepts: list, filters: dict = None) -> list:
        try:
            start_time = time.time()
            logger.info("Starting search for concepts: %s", concepts)
            
            # Perform searches
            semantic_results = self._semantic_search(concepts)
            graph_results = self._graph_search(concepts)
            
            # Process results
            merged = self._merge_results(semantic_results, graph_results)
            filtered = self._apply_filters(merged, filters)
            final_results = self._rerank(filtered)
            
            logger.info("Search completed in %.2fs with %d results", 
                       time.time()-start_time, len(final_results))
            
            semantic_results = self._semantic_search(concepts)
            if semantic_results is None:
                logger.error("Semantic search returned None")
                semantic_results = []

            graph_results = self._graph_search(concepts)
            if graph_results is None:
                logger.error("Graph search returned None")
                graph_results = []

            merged = self._merge_results(semantic_results, graph_results)
            if merged is None:
                logger.error("Merge returned None")
                merged = []

            filtered = self._apply_filters(merged, filters)
            if filtered is None:
                logger.error("Filter returned None")
                filtered = []

            final_results = self._rerank(filtered)
            if final_results is None:
                logger.error("Rerank returned None")
                final_results = []

            return final_results
            
        except Exception as e:
            logger.error("Search failed: %s", str(e))
            return []

    def _semantic_search(self, concepts: list) -> list:
        try:
            query = " ".join(concepts)
            query_embed = self.kb.encoder.encode(query)
            
            scores, indices = self.kb.semantic_index.search(
                query_embed.reshape(1, -1), 
                100
            )
            
            results = []
            for i, score in zip(indices[0], scores[0]):
                course = self.kb.data.iloc[i].to_dict()
                course['semantic_score'] = float(score)
                results.append(course)
            
            logger.info("Semantic search returned %d results", len(results))
            return results
            
        except Exception as e:
            logger.error("Semantic search failed: %s", str(e))
            return []

    def _graph_search(self, concepts: list) -> list:
        # Map LLM‐parsed concepts → actual graph nodes
        logger.debug(f"Grounding concepts: {concepts}")
        seed_nodes = self._ground_concepts(concepts)
        logger.debug(f"Resolved seed nodes: {seed_nodes}")
        course_scores = defaultdict(float)
        beam = self.beam_width

        for node in seed_nodes:
            if node not in self.kb.graph:
                continue
            queue = deque([(node, 1.0)])
            visited = set()

            for _ in range(3):  # max hops
                next_queue = []
                for curr, score in queue:
                    if curr in visited:
                        continue
                    visited.add(curr)
                    logger.debug(f"Visiting node: {curr} (current score: {score:.2f})")
                    if self.kb.graph.nodes[curr]['type'] == 'course':
                        course_scores[curr] += score
                        continue

                    for nbr in self.kb.graph.neighbors(curr):
                        w = max(ed.get('weight', 0.7)
                                for ed in self.kb.graph.get_edge_data(curr, nbr).values())
                        next_queue.append((nbr, score * w * 0.8))

                # beam‐width prune
                queue = sorted(next_queue, key=lambda x: -x[1])[:beam]

        # convert to list of course dicts
        results = []
        for cid, sc in course_scores.items():
            row = self.kb.data.loc[self.kb.data['course_id']==cid].iloc[0].to_dict()
            row['graph_score'] = sc
            results.append(row)

        logger.info("Graph search returned %d results", len(results))
        return results

    def _merge_results(self, semantic: list, graph: list) -> list:
        merged = defaultdict(dict)
        
        for item in semantic:
            merged[item['course_id']] = {
                **item,
                'combined_score': item['semantic_score'] * 0.6
            }
        
        for item in graph:
            cid = item['course_id']
            if cid in merged:
                merged[cid]['combined_score'] += item['graph_score'] * 0.4
            else:
                merged[cid] = {
                    **item,
                    'combined_score': item['graph_score'] * 0.4
                }
        
        return list(merged.values())

    def _apply_filters(self, results: list, filters: dict) -> list:
        if not filters:
            return results or []
            
        filtered = []
        for course in results:
            valid = True
            
            # Difficulty filter
            if 'difficulty' in filters:
                valid &= course.get('learning_path', {}).get('difficulty') in filters['difficulty']
            
            # Duration filter
            if 'max_duration' in filters:
                valid &= course.get('duration_months', float('inf')) <= filters['max_duration']
            
            # Rating filter
            if 'min_rating' in filters:
                valid &= course.get('rating', 0) >= filters['min_rating']
            
            # Language filter
            if 'languages' in filters:
                langs = course.get('course_info', {}).get('subtitle_languages') or []
                # ensure it's a list
                if isinstance(langs, str):
                    langs = [langs]
                langs = [l.lower() for l in langs]
                valid &= any(lang in langs for lang in filters['languages'])
            
            # Career path filter
            if 'career_paths' in filters:
                careers = course.get('learning_path', {}).get('career_paths') or []
                if isinstance(careers, str):
                    careers = [careers]
                careers = [c.lower() for c in careers]
                valid &= any(career in careers for career in filters['career_paths'])
            
            if valid:
                filtered.append(course)
        
        logger.info("Applied filters: %d -> %d results", len(results), len(filtered))
        return filtered


    def _rerank(self, results: list) -> list:
        return sorted(
            results,
            key=lambda x: (
                -x['combined_score'],
                -x['rating'],
                x['duration_months']
            )
        )
class GraphRAGSearchEngine(HybridSearchEngine):
    def __init__(self, kb): 
        super().__init__(kb)
        self.rag = GraphRAGProcessor(kb)
    def search(self, concepts, filters=None):
        # 1) RAG pulls in relevant courses
        context = self.rag.retrieve_context(" ".join(concepts))
        # 2) semantic on full KB + graph‐traversal *within* that context
        sem = self._semantic_search(concepts)
        gr  = self._context_aware_graph_search(concepts, context)
        merged = self._merge_results(sem, gr)
        filtered = self._apply_filters(merged, filters)
        return self._rerank(filtered)
    def _context_aware_graph_search(self, concepts, context):
        seeds = self._ground_concepts(concepts)
        scores = defaultdict(float)
        # give every course in RAG context a base score
        for c in context:
            scores[c] += 1.0
        # personalized‐PageRank restart on context
        pr = nx.pagerank(self.kb.graph, personalization={c:1 for c in context})        # only keep those nodes that are actually courses
        for node_id, sc in pr.items():
            if self.kb.graph.nodes[node_id].get('type') == 'course':
                scores[node_id] += 0.5 * sc
        return [ self.kb.data[self.kb.data.course_id==cid].iloc[0].to_dict() 
                 | {'graph_score':scores[cid]} 
                 for cid in scores ]

# ======================
# 4. LLM Integration
# ======================
class LLMAssistant:
    def __init__(self):
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    def generate(self, prompt: str, model: str = "qwen2.5:1.5b") -> dict:
        try:
            start_time = time.time()
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.4}
            )
            logger.debug("LLM response time: %.2fs", time.time()-start_time)
            return json.loads(response['response'])
        except Exception as e:
            logger.error("LLM request failed: %s", str(e))
            return {}

# ======================
# 5. Main System
# ======================
class CareerAdvisorSystem:
    def __init__(self, data_path: str):
        self.kb = KnowledgeBase(data_path)
        self.parser = QueryParser()
        self.searcher = GraphRAGSearchEngine(self.kb)
        logger.info("Career advisor system initialized")

    def query(self, natural_query: str) -> dict:
        try:
            start_time = time.time()
            
            # Parse query
            concepts, filters = self.parser.parse(natural_query)
            logger.info("Parsed concepts: %s", concepts)
            logger.info("Parsed filters: %s", filters)
            
            # Execute search
            results = self.searcher.search(concepts, filters)
            
            return {
                "query": natural_query,
                "results": results[:10],
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "total_results": len(results),
                    "explanation": self._generate_explanation(concepts, filters)
                }
            }
        except Exception as e:
            logger.error("Query processing failed: %s", str(e))
            return {"error": "System error occurred"}
    def _generate_explanation(self, concepts, filters):
        return {
            "searched_concepts": concepts,
            "applied_filters": filters,
            "search_strategy": "Hybrid (semantic + 3-hop graph traversal)",
            "scoring": {
                "semantic_weight": 0.6,
                "graph_weight": 0.4
            }
        }
# ======================
# 6. Execution
# ======================
if __name__ == "__main__":
    # Initialize system
    system = CareerAdvisorSystem("D:\\Thesis\\Courses-Searching\\course_searching_main\\data\\processed_courses_detail.json")
    
    # Interactive interface
    print("Career Path Advisor System")
    print("Enter your learning query (e.g., 'I want to learn AI with Python in 6 months')")
    
    while True:
        try:
            query = input("\nYour query (type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
                
            # Process query
            start_time = time.time()
            response = system.query(query)
            
            # Handle errors
            if 'error' in response:
                print(f"\nError: {response['error']}")
                continue
                
            # Display results
            print(f"\nFound {len(response['results'])} relevant courses (processed in {response['metadata']['processing_time']:.2f}s):")
            for idx, course in enumerate(response['results'][:5], 1):
                # Safely grab subtitle_languages as a list
                langs = course.get('course_info', {}).get('subtitle_languages') or []
                if isinstance(langs, str):
                    langs = [langs]

                careers = course.get('learning_path', {}).get('career_paths') or []
                # Ensure careers is a list
                if not isinstance(careers, list):
                    careers = [careers]

                print(f"""
            {idx}. {course.get('title', 'Untitled')}
            - Rating: {course.get('rating', 'N/A')}/5
            - Duration: {course.get('duration_months', 'N/A')} months
            - Difficulty: {course.get('learning_path', {}).get('difficulty', '').capitalize()}
            - Career paths: {', '.join(careers)}
            - Languages: {', '.join(langs)}
                """)
            print(f"\nSearch explanation: {json.dumps(response['metadata']['explanation'], indent=2)}")
                            
        except KeyboardInterrupt:
            print("\nExiting system...")
            break
            
    print("Thank you for using Career Path Advisor!")