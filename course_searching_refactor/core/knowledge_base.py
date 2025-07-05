import time
import logging
import numpy as np
import streamlit as st
import faiss
from typing import List, Dict
from models.embedding import cosine_similarity
from core.query_processor import QueryProcessor
from database.neo4j_client import get_neo4j_connection

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def load_course_embeddings_cached():
    """Load and cache course embeddings from Neo4j"""
    neo4j_conn = get_neo4j_connection()
    query = """
    MATCH (c:Course)
    WHERE c.embedding_sbert IS NOT NULL
    OPTIONAL MATCH (c)-[:TEACHES]->(sk:Skill)
    OPTIONAL MATCH (c)-[:HAS_SUBJECT]->(sub:Subject)
    RETURN
        c.url AS url,
        c.name AS name,
        c.description AS description,
        c.rating AS rating,
        c.duration AS duration,
        collect(DISTINCT sk.name) AS skills,
        collect(DISTINCT sub.name) AS subjects,
        c.embedding_sbert AS emb
    """
    results = neo4j_conn.execute_query(query)
    course_embs = {}
    for record in results:
        url = record.get("url")
        emb_list = record.get("emb", [])
        if url and emb_list:
            course_embs[url] = {
                "embedding": np.array(emb_list, dtype=np.float32),
                "name": record.get("name", ""),
                "description": record.get("description", ""),
                "rating": record.get("rating", ""),
                "duration": record.get("duration", 0),
                "skills": record.get("skills", []),
                "subjects": record.get("subjects", [])
            }
    return course_embs

@st.cache_data(ttl=3600)
def load_skill_embeddings_cached():
    """Load and cache skill embeddings from Neo4j"""
    neo4j_conn = get_neo4j_connection()
    query = """
    MATCH (s:Skill)
    WHERE s.embedding_sbert IS NOT NULL
    RETURN s.name AS name, s.embedding_sbert AS emb
    """
    results = neo4j_conn.execute_query(query)
    skill_embs = {}
    for record in results:
        name = record.get("name")
        emb_list = record.get("emb", [])
        if name and emb_list:
            skill_embs[name] = np.array(emb_list, dtype=np.float32)
    return skill_embs

@st.cache_resource
def build_faiss_indexes():
    """Build and cache FAISS indexes for fast similarity search"""
    course_data = load_course_embeddings_cached()
    skill_data = load_skill_embeddings_cached()
    
    # Build course index
    course_index = None
    course_urls = []
    if course_data:
        embeddings = [data["embedding"] for url, data in course_data.items()]
        if embeddings:
            d = len(embeddings[0])
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Use FlatIP for speed - no complex IVF
            course_index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embeddings_array)
            course_index.add(embeddings_array)
            logger.info(f"Fast Flat index built for {len(embeddings)} courses")
                
            course_urls = list(course_data.keys())
    
    # Build skill index
    skill_index = None
    skill_names = []
    if skill_data:
        embeddings = list(skill_data.values())
        if embeddings:
            d = len(embeddings[0])
            skill_index = faiss.IndexFlatIP(d)
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            skill_index.add(embeddings_array)
            skill_names = list(skill_data.keys())
    
    logger.info(f"FAISS indexes built: {len(course_urls)} courses, {len(skill_names)} skills")
    
    return {
        'course_index': course_index,
        'course_urls': course_urls,
        'course_data': course_data,
        'skill_index': skill_index,
        'skill_names': skill_names,
        'skill_data': skill_data
    }

class KnowledgeBaseQA:
    """Main knowledge base QA system combining semantic search and structured queries"""
    
    def __init__(self, neo4j_conn, embedding_model, top_skill_k: int = 5):
        self.neo4j_conn = neo4j_conn
        self.embedding_model = embedding_model
        self.query_processor = QueryProcessor()
        self.top_skill_k = top_skill_k
        
        # Load cached indexes
        try:
            indexes = build_faiss_indexes()
            self.course_emb_index = indexes['course_index']
            self.course_urls = indexes['course_urls'] 
            self.course_embeddings_data = indexes['course_data']
            self.skill_emb_index = indexes['skill_index']
            self.skill_names = indexes['skill_names']
            self.skill_embeddings_data = indexes['skill_data']
            
            logger.info(f"FAISS indexes loaded: {len(self.course_urls)} courses, {len(self.skill_names)} skills")
        except Exception as e:
            logger.error(f"Error loading FAISS indexes: {e}")
            # Fallback to empty indexes
            self.course_emb_index = None
            self.skill_emb_index = None
            self.course_urls = []
            self.skill_names = []
            self.course_embeddings_data = {}
            self.skill_embeddings_data = {}

    def get_course_details(self, url: str) -> Dict:
        """Get detailed course information by URL"""
        query = """
        MATCH (c:Course {url: $url})
        OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
        OPTIONAL MATCH (c)-[:HAS_LEVEL]->(l:Level)
        OPTIONAL MATCH (c)-[:TAUGHT_BY]->(i:Instructor)
        RETURN 
            c.name AS name,
            c.description AS description,
            c.rating AS rating,
            c.duration AS duration,
            l.name AS level,
            collect(DISTINCT s.name) AS skills,
            i.name AS instructor
        """
        result = self.neo4j_conn.execute_query(query, {"url": url})
        return result[0] if result else None

    def _find_similar_skills(self, query: str, top_k: int = None) -> List[str]:
        """Find similar skills using FAISS similarity search"""
        start_time = time.perf_counter()
        desired_k = top_k or self.top_skill_k
        num_skills = len(self.skill_names)

        # Never ask Faiss for more neighbors than available vectors
        k = min(desired_k, num_skills)

        query_emb = self.embedding_model.get_embedding(query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_emb)

        if self.skill_emb_index is None or num_skills == 0:
            return []

        similarities, indices = self.skill_emb_index.search(query_emb, k)

        top_skills = []
        for sim_score, idx in zip(similarities[0], indices[0]):
            if sim_score > 0.5:
                top_skills.append(self.skill_names[idx])

        logger.debug(f"Skill similarity search took {time.perf_counter() - start_time:.2f} seconds")
        return top_skills

    def _get_courses_by_skills(self, skill_names: List[str]) -> List[Dict]:
        """Get courses that teach specific skills"""
        courses = []
        for skill in skill_names:
            query = """
            MATCH (c:Course)-[:TEACHES]->(s:Skill {name: $skill_name})
            OPTIONAL MATCH (c)-[:HAS_SUBJECT]->(sub:Subject)
            RETURN DISTINCT
                c.url AS url,
                c.name AS name,
                c.description AS description,
                c.rating AS rating,
                c.duration AS duration,
                collect(DISTINCT sub.name) AS subjects
            """
            params = {"skill_name": skill}
            results = self.neo4j_conn.execute_query(query, params)
            for rec in results:
                rec["skills"] = [skill]
                courses.append(rec)
        unique = {}
        for c in courses:
            if c["url"] not in unique:
                unique[c["url"]] = c
        return list(unique.values())

    def _get_candidate_urls_by_embedding(self, query: str, top_n: int = 20) -> List[str]:
        """Get candidate course URLs using embedding similarity"""
        query_emb = self.embedding_model.get_embedding(query).astype(np.float32)
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # Search using FAISS
        similarities, indices = self.course_emb_index.search(query_emb.reshape(1, -1), top_n)
        return [self.course_urls[i] for i in indices[0]]

    def _get_fallback_courses_embedding_only(self, query: str, exclude_urls: List[str], top_k: int = 5) -> List[Dict]:
        """Fallback course search using only embedding similarity"""
        query_emb = self.embedding_model.get_embedding(query).astype(np.float32)
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # Find top N + number to exclude
        n = top_k + len(exclude_urls)
        similarities, indices = self.course_emb_index.search(query_emb.reshape(1, -1), n)
        
        results = []
        for i in range(n):
            idx = indices[0][i]
            url = self.course_urls[idx]
            
            if url in exclude_urls:
                continue
                
            data = self.course_embeddings_data[url]
            results.append({
                "url": url,
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "rating": data.get("rating", ""),
                "duration": data.get("duration", 0),
                "skills": data.get("skills", []),
                "subjects": data.get("subjects", []),
                "similarity": float(similarities),
                "source": "embedding_only"
            })
            if len(results) >= top_k:
                break
        return results

    def process_query(self, user_query: str) -> List[Dict]:
        """Main query processing with simplified workflow for speed"""
        try:
            # Step 1: Find similar skills (fast)
            t0 = time.perf_counter()
            top_skills = self._find_similar_skills(user_query)
            t1 = time.perf_counter()
            logger.info(f"Skills found in {t1-t0:.2f}s: {top_skills}")

            # Step 2: Try main Cypher query
            plan = self.query_processor.generate_query_plan(user_query, candidate_skills=top_skills)
            self.last_main_plan = plan
            cypher = plan.get('final_query', "")
            query_type = plan.get('query_type', 'course')
            
            logger.info(f"🎯 Generated {query_type} query: {cypher}")
            
            if not cypher:
                logger.warning("LLM did not return main Cypher. Using semantic search only.")
                raw_results = []
            else:
                raw_results = self.neo4j_conn.execute_query(cypher)
                
            t2 = time.perf_counter()
            logger.info(f"📊 Neo4j returned {len(raw_results)} results in {t2-t1:.3f}s")

            # Step 3: Process results based on query type
            if raw_results:
                processed_results = []
                
                non_course_types = ["instructor", "organization", "provider", "review", "subject", "skill", "level", "statistical"]
                
                if query_type == 'course' or query_type == 'unknown':
                    query_lower = user_query.lower()
                    if any(keyword in query_lower for keyword in ['instructor', 'teacher', 'professor', 'taught by', 'who teaches']):
                        query_type = "instructor"
                    elif any(keyword in query_lower for keyword in ['organization', 'university', 'college', 'school', 'institution']):
                        query_type = "organization"
                    elif any(keyword in query_lower for keyword in ['provider', 'platform', 'coursera', 'edx', 'udemy']):
                        query_type = "provider"
                    elif any(keyword in query_lower for keyword in ['review', 'feedback', 'comment', 'rating']):
                        query_type = "review"
                    elif any(keyword in query_lower for keyword in ['subject', 'topic', 'area', 'field']):
                        query_type = "subject"
                    elif any(keyword in query_lower for keyword in ['skill', 'ability', 'competency']):
                        query_type = "skill"
                    elif any(keyword in query_lower for keyword in ['level', 'difficulty', 'beginner', 'intermediate', 'advanced']):
                        query_type = "level"
                    elif any(keyword in query_lower for keyword in ['average', 'statistics', 'how many', 'count', 'total']):
                        query_type = "statistical"
                
                if query_type in non_course_types:
                    logger.info(f"🎯 Processing {query_type} query results")
                    
                    for i, rec in enumerate(raw_results):
                        # Assign high similarity for exact matches
                        rec.setdefault("similarity", 0.9 - (i * 0.02))
                        rec.setdefault("source", f"{query_type}_match")
                        rec.setdefault("skills", [])
                        rec.setdefault("subjects", [])
                        rec["query_type"] = query_type
                        
                        # Standardize field names for each type
                        self._standardize_result_fields(rec, query_type)
                        processed_results.append(rec)
                    
                    logger.info(f"Successfully returning {len(processed_results)} {query_type} results")
                    return processed_results[:10]
                
                else:
                    # Course queries - use existing embedding logic
                    logger.info(f"🎯 Processing course query results")
                    query_emb = self.embedding_model.get_embedding(user_query)
                    
                    valid_results = []
                    for rec in raw_results:
                        url = rec.get("url")
                        if url and url in self.course_embeddings_data:
                            course_emb = self.course_embeddings_data[url]["embedding"]
                            sim = cosine_similarity(query_emb, course_emb)
                            rec["similarity"] = float(sim)
                            rec["source"] = "skill_match"
                            rec.setdefault("skills", [])
                            rec.setdefault("subjects", [])
                            rec["query_type"] = "course"
                            valid_results.append(rec)
                    
                    valid_results.sort(key=lambda x: (-x["similarity"], x.get("url", "")))
                    logger.info(f"Ranked {len(valid_results)} course results")
                    
                    if len(valid_results) >= 1:
                        return valid_results[:10]
            
            # Simplified fallback: only use semantic search
            if query_type == "course" or not raw_results:
                logger.info(f"Using simplified semantic fallback")
                return self._simple_semantic_search(user_query, top_k=10)
            else:
                logger.info(f"No results for {query_type} query")
                return []

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            return []
        
    def _simple_semantic_search(self, user_query: str, top_k: int = 10) -> List[Dict]:
        """Simplified semantic search without complex refinement"""
        try:
            logger.info(f"🔍 Running simple semantic search for: {user_query}")
            
            # Get query embedding
            query_emb = self.embedding_model.get_embedding(user_query).astype(np.float32)
            faiss.normalize_L2(query_emb.reshape(1, -1))
            
            # Search FAISS index
            similarities, indices = self.course_emb_index.search(query_emb.reshape(1, -1), min(top_k, len(self.course_urls)))
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                similarity = float(similarities[0][i])
                
                # Skip low similarity results
                if similarity < 0.3:
                    continue
                    
                url = self.course_urls[idx]
                data = self.course_embeddings_data[url]
                
                # Build result with course details
                result = {
                    "url": url,
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    "rating": data.get("rating", ""),
                    "duration": data.get("duration", 0),
                    "skills": data.get("skills", []),
                    "subjects": data.get("subjects", []),
                    "similarity": similarity,
                    "source": "semantic_search",
                    "query_type": "course"
                }
                results.append(result)
            
            logger.info(f"Semantic search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _standardize_result_fields(self, result: Dict, query_type: str):
        """Standardize field names for different entity types"""
        if query_type == "instructor":
            if "i.name" in result and "instructor_name" not in result:
                result["instructor_name"] = result["i.name"]
            if "i.rating" in result and "instructor_rating" not in result:
                result["instructor_rating"] = result["i.rating"]
            if "i.description" in result and "instructor_description" not in result:
                result["instructor_description"] = result["i.description"]
                
        elif query_type == "organization":
            if "o.name" in result and "organization_name" not in result:
                result["organization_name"] = result["o.name"]
            if "o.description" in result and "organization_description" not in result:
                result["organization_description"] = result["o.description"]
                
        elif query_type == "provider":
            if "p.name" in result and "provider_name" not in result:
                result["provider_name"] = result["p.name"]
            if "p.description" in result and "provider_description" not in result:
                result["provider_description"] = result["p.description"]
                
        elif query_type == "review":
            if "r.comment" in result and "review_comment" not in result:
                result["review_comment"] = result["r.comment"]
            if "r.rating" in result and "review_rating" not in result:
                result["review_rating"] = result["r.rating"]
            if "r.stars" in result and "review_stars" not in result:
                result["review_stars"] = result["r.stars"]
                
        elif query_type == "subject":
            if "sub.name" in result and "subject_name" not in result:
                result["subject_name"] = result["sub.name"]
            if "sub.description" in result and "subject_description" not in result:
                result["subject_description"] = result["sub.description"]
                
        elif query_type == "skill":
            if "s.name" in result and "skill_name" not in result:
                result["skill_name"] = result["s.name"]
            if "s.description" in result and "skill_description" not in result:
                result["skill_description"] = result["s.description"]
                
        elif query_type == "level":
            if "l.name" in result and "level_name" not in result:
                result["level_name"] = result["l.name"]
            if "l.description" in result and "level_description" not in result:
                result["level_description"] = result["l.description"]

    def process_query_with_context(self, user_query: str, chat_history: list = None) -> List[Dict]:
        """Process query with history-based context for LLM prompts"""
        # Build enhanced query only for LLM prompt
        enhanced_query = self._enhance_query_with_context(user_query, chat_history)

        try:
            # Step 1: Skill-based lookup using raw user_query
            top_skills = self._find_similar_skills(user_query)

            # Generate main Cypher using enhanced query
            plan = self.query_processor.generate_query_plan(enhanced_query, candidate_skills=top_skills)
            cypher = plan.get('final_query', "")
            raw_results = []
            if cypher:
                raw_results = self.neo4j_conn.execute_query(cypher)

            # Ranking results by embedding similarity
            query_emb = self.embedding_model.get_embedding(user_query)
            ranked = []
            for rec in raw_results:
                url = rec.get('url')
                if url in self.course_embeddings_data:
                    emb = self.course_embeddings_data[url]['embedding']
                    sim = cosine_similarity(query_emb, emb)
                    rec.update({'similarity': sim, 'source': 'skill_match'})
                    ranked.append(rec)
            ranked.sort(key=lambda x: (-x['similarity'], x.get('url','')))

            # If enough results, return top
            THRESHOLD = 1
            if len(ranked) >= THRESHOLD:
                return ranked[:10]
            logger.info(f"🔍 Number of raw results from Cypher: {len(ranked)}")
            logger.info(f"📏 Threshold for sufficiency: {THRESHOLD}")
            
            return ranked

        except Exception as e:
            logger.error(f"Error in process_query_with_context: {e}", exc_info=True)
            return []
    
    def _enhance_query_with_context(self, current_query: str, chat_history: list = None) -> str:
        """Enhance query with context from chat history"""
        if not chat_history:
            return current_query
            
        # Find previously mentioned topics/skills
        mentioned_topics = set()
        education_keywords = ["course", "learn", "training", "skill", "programming", "data", "web", "python", "java", "javascript"]
        
        for msg in chat_history[-10:]:  # Look at 10 recent messages
            if msg["role"] == "user":
                content = msg["content"].lower()
                for keyword in education_keywords:
                    if keyword in content:
                        # Extract potential skill/topic around the keyword
                        words = content.split()
                        for i, word in enumerate(words):
                            if keyword in word:
                                # Get 1-2 words around
                                start = max(0, i-1)
                                end = min(len(words), i+3)
                                topic = " ".join(words[start:end])
                                mentioned_topics.add(topic)
        
        if mentioned_topics:
            context_info = f"Previous topics discussed: {', '.join(list(mentioned_topics)[:3])}. "
            return context_info + current_query
        
        return current_query

@st.cache_resource
def get_knowledge_base_qa():
    """Cached KnowledgeBaseQA system instance"""
    from database.neo4j_client import get_neo4j_connection
    from models.embedding import get_embedding_model
    neo4j_conn = get_neo4j_connection()
    embedding_model = get_embedding_model()
    return KnowledgeBaseQA(neo4j_conn, embedding_model)