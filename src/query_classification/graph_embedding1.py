# all_in_one.py
import os, sys, shutil, atexit, logging, re, time, json, traceback
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import ollama
import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import torch
from langgraph.graph import StateGraph, END

# ---------------------------------------------
# 0. Clear caches at exit
# ---------------------------------------------
def clear_all_caches():
    try:
        import streamlit as _st
        _st.cache_data.clear()
        _st.cache_resource.clear()
    except:
        pass
    for d in (
        os.path.expanduser("~/.streamlit/cache"),
        os.path.expanduser("~/.cache/huggingface"),
        os.getenv("TEMP", "")
    ):
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

atexit.register(clear_all_caches)


# ---------------------------------------------
# 1. Utilities, Neo4j, Embeddings, LLM wrapper
# ---------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    force=True
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else 0.0

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self._driver = GraphDatabase.driver(uri, auth=(user,pwd))
    def execute_query(self, q, p=None):
        try:
            with self._driver.session() as s:
                return [dict(r) for r in s.run(q, p or {})]
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []
    def close(self): self._driver.close()

class SBERTEmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def get_embedding(self, text:str)->np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

class OllamaLLM:
    def __init__(self, model="qwen3:4b", small=False):
        self.model, self.opts = model, {'temperature':0.1} if small else {'temperature':0.3}
    def invoke(self, messages:List[Dict[str,str]])->str:
        prompt = "\n".join(m["content"] for m in messages)
        return ollama.generate(model=self.model, prompt=prompt, options=self.opts)['response']

def get_llm(small=False): return OllamaLLM(small=small)

# ---------------------------------------------
class QueryProcessor:
    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL | re.MULTILINE)

    def generate_query_plan(self, user_query: str, candidate_skills: list = None) -> Dict:
        start_time = time.perf_counter()
        skills_str = ""
        if candidate_skills:
            escaped = [s.replace("'", "\\'").replace('"', '\\"') for s in candidate_skills]
            quoted = [f"'{s}'" for s in escaped]
            joined = ",".join(quoted)
            skills_str = f"[{joined}]"

        prompt = f"""
You are a Neo4j Cypher expert. Generate a valid Cypher query according to these STRICT RULES:

1. Query MUST start with MATCH.
2. Use ONLY node labels: Course, Skill, Level, Organization, Instructor, Career.
3. Use ONLY relationships: TEACHES, HAS_LEVEL, OFFERED_BY, TAUGHT_BY, REQUIRES.
4. ALWAYS use single quotes for string values: 'value'.
5. If there are multiple conditions (skill, level, provider, etc.), you MUST either:
   - Combine them in a single MATCH (comma-separated) and then a single WHERE clause, 
     OR
   - Use multiple MATCH lines, followed by exactly one WHERE clause that combines all conditions with AND/OR.
   NEVER use two WHERE clauses in a row.
6. If user query implies a course level (e.g., “Beginner”, “Intermediate”, “Advanced”), you MUST:
   a. First try to match via the HAS_LEVEL relationship, e.g.:
        MATCH (c:Course)-[:HAS_LEVEL]->(l:Level {{name: '<LevelName>'}})
   b. ONLY if no such Level node exists for that course, fallback to checking the course title, e.g.:
        c.name CONTAINS '<level keyword>'
   c. Special rule for levels:
            - Detect any of these keywords in the query (case-insensitive):
                Beginner, Introductory, Basics, Fundamentals...
                Intermediate, Mid-level... 
                Advanced, Expert, Hard... 
            - Map the matched keyword to Level.name exactly (“Beginner”, “Intermediate” or “Advanced”).
            - Remove that keyword (and any prepositions like “of”) from the skill phrase.
            - **Normalize the skill name** by converting it to Title Case (capitalize the first letter of each word).
            - Then generate one step to MATCH the Skill{{name: $skill_name}} and one step to MATCH courses with HAS_LEVEL → Level{{name: $level}}.
        
7. MUST include RETURN clause with course properties.
8. final_query MUST begin with MATCH (case insensitive).
9. NEVER use double quotes "" inside the Cypher.
10. If rating is involved, use toFloat(c.rating) instead of c.rating.
11. For Course→X relationships (TEACHES, HAS_LEVEL, etc.), always match directly from Course:
     MATCH (c:Course)-[:REL_TYPE]->(x:NodeType {{…}}).
12. **CRITICAL**: When CANDIDATE_SKILLS is provided, you MUST use ALL relevant skills from the list in the WHERE clause using s.name IN [...]. 
    - Analyze the user query to determine which skills from CANDIDATE_SKILLS are relevant
    - Include ALL relevant skills, not just one
    - Use the EXACT skill names from CANDIDATE_SKILLS without modification
    - For example: s.name IN ['AWS cloud computing', 'AWS Certified Cloud Practitioner', 'AWS cloud']

SCHEMA:
- Course properties: url, name, duration, rating, description
- Relationships:
  (Course)-[:TEACHES]->(Skill)
  (Course)-[:HAS_LEVEL]->(Level)
  (Course)-[:OFFERED_BY]->(Organization)
  (Course)-[:TAUGHT_BY]->(Instructor)
  (Career)-[:REQUIRES]->(Skill)

--- EXAMPLE A: Skill + Level + Provider ---
USER QUERY: Find intermediate Data Science courses taught by University of Michigan
STEPS:
1. Match courses teaching 'Data Science' skill.
2. Match courses having Level 'Intermediate'.
3. Match courses offered by 'University of Michigan'.
4. Return URL, name, duration, rating, description.

FINAL_QUERY:
MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:HAS_LEVEL]->(l:Level {{name: 'Intermediate'}}), (c)-[:OFFERED_BY]->(o:Organization {{name: 'University of Michigan'}})
WHERE s.name IN ['Data Science']
RETURN c.url AS url, c.name AS name, c.duration AS duration, toFloat(c.rating) AS rating, c.description AS description

--- EXAMPLE B: Skill + Level only---
USER QUERY: Show beginner Rust courses
STEPS:
1. Match courses teaching 'Rust' skill.
2. Try match Level 'Beginner'
3. Return URL, name, duration, rating, description.

FINAL_QUERY:
MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:HAS_LEVEL]->(l:Level)
WHERE s.name = 'Rust' AND l.name = 'Beginner'
RETURN c.url AS url, c.name AS name, c.duration AS duration, toFloat(c.rating) AS rating, c.description AS description

--- IMPORTANT: MUST RETURN A JSON WITH BOTH "steps" AND "final_query" KEYS. 
--- If you cannot build a valid Cypher, set "final_query" to an empty string "" (but still include the key).

USER QUERY: {user_query}
CANDIDATE_SKILLS: {skills_str}
!!!IMPORTANT: use *all* of these in your Cypher WHERE clause.
OUTPUT FORMAT (JSON):
{{
    "steps": ["step1", "step2", ...],
    "final_query": "MATCH... RETURN..."   ← must exist, or empty string
}}
"""
        try:
            logger.debug("LLM prompt for query plan:\n%s", prompt)
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1},
                format='json'
            )
            logger.debug("Raw LLM response:\n%s", response['response'])
            plan = self.extract_json(response['response'])

            # Nếu LLM không include 'final_query', gán empty string
            if 'final_query' not in plan or not isinstance(plan['final_query'], str):
                plan['final_query'] = ""
            else:
                cleaned = self.clean_cypher(plan['final_query'])
                try:
                    self.validate_cypher(cleaned)
                    plan['final_query'] = cleaned
                except Exception:
                    plan['final_query'] = ""

            # Thêm debug_info như cũ
            plan.setdefault('debug_info', {})
            return plan

        except Exception as e:
            gen_time = time.perf_counter() - start_time
            logger.error(f"Query generation failed after {gen_time:.2f}s: {str(e)}")
            return {
                "steps": ["Error generating query plan"],
                "final_query": "",
                "debug_info": {
                    'error': str(e),
                    'gen_time': gen_time
                }
            }


    def generate_refinement_query(self, user_query: str, candidate_urls: List[str]) -> Dict:
        """
        Generate a Cypher query that filters among candidate_urls by applying conditions
        from user_query (e.g., rating > X, level, etc.).
        """
        start_time = time.perf_counter()
        quoted_urls = [f"'{u}'" for u in candidate_urls]
        joined_urls = ", ".join(quoted_urls)

        prompt = f"""
You are a Neo4j Cypher expert. Below is a user’s natural language query and a list of candidate course URLs.
Generate a Cypher query that filters ONLY among these candidate course URLs by applying any additional conditions
implied in the user’s query (e.g., rating thresholds, level filters, provider names, etc.).

STRICT RULES:
1. Query MUST start with MATCH.
2. Use ONLY node labels: Course, Skill, Level, Organization, Instructor, Career.
3. Use ONLY relationships: TEACHES, HAS_LEVEL, OFFERED_BY, TAUGHT_BY, REQUIRES.
4. ALWAYS use single quotes for string values: 'value'.
5. Combine conditions using WHERE when needed.
6. MUST include RETURN clause with course properties.
7. final_query MUST begin with MATCH (case insensitive).
8. NEVER use double quotes "" inside the Cypher.
9. If rating is involved, use toFloat(c.rating) instead of c.rating.
10. For Course→X relationships (TEACHES, HAS_LEVEL, etc.), always match directly from Course:
    MATCH (c:Course)-[:REL_TYPE]->(x:NodeType {{…}}).

SCHEMA:
- Course properties: url, name, duration, rating, description
- Relationships:
  (Course)-[:TEACHES]->(Skill)
  (Course)-[:HAS_LEVEL]->(Level)
  (Course)-[:OFFERED_BY]->(Organization)
  (Course)-[:TAUGHT_BY]->(Instructor)
  (Career)-[:REQUIRES]->(Skill)

CANDIDATE_URLS: [{joined_urls}]

USER QUERY: {user_query}

OUTPUT FORMAT (JSON):
{{
    "steps": ["step1", "step2", ...],
    "final_query": "MATCH... RETURN..."
}}
"""
        try:
            logger.debug(f"Refinement Prompt to LLM:\n{prompt[:400]}...")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1},
                format='json'
            )
            logger.debug("Raw LLM response for generate_refinement_query:\n%s", response['response'])
            gen_time = time.perf_counter() - start_time
            logger.info(f"Refinement query generated in {gen_time:.2f}s (model={self.model_name})")

            plan = self.extract_json(response['response'])
            plan['final_query'] = self.clean_cypher(plan['final_query'])
            self.validate_cypher(plan['final_query'])
            plan['debug_info'] = {
                'gen_time': gen_time,
                'model': self.model_name,
                'prompt_tokens': len(prompt.split()),
                'response_tokens': len(response['response'].split())
            }
            return plan

        except Exception as e:
            gen_time = time.perf_counter() - start_time
            logger.error(f"Refinement query generation failed after {gen_time:.2f}s: {str(e)}")
            return {
                "steps": ["Error generating refinement query"],
                "final_query": "",
                "debug_info": {
                    'error': str(e),
                    'gen_time': gen_time
                }
            }

    def extract_json(self, text: str) -> Dict:
        text = text.replace("True", "true").replace("False", "false")
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        match = self.json_pattern.search(text)
        if not match:
            raise ValueError("No JSON found in response")
        json_str = match.group()

        def replace_inner_quotes(m):
            inner = m.group(1)
            inner = re.sub(r'(?<!\\)"', "'", inner)
            inner = inner.replace('\\', '\\\\')
            return f'"final_query": "{inner}"'

        json_str = re.sub(
            r'"final_query"\s*:\s*"((?:\\"|[^"])*)"',
            replace_inner_quotes,
            json_str,
            flags=re.DOTALL
        )
        json_str = json_str.replace('“', '"').replace('”', '"')
        json_str = re.sub(r'\s*]\s*{', '], {', json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e.msg}")
            logger.error(f"Problematic JSON content: {json_str}")
            raise

    def clean_cypher(self, query: str) -> str:
        q = query.strip()
        q = re.sub(r"^```cypher|```$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r";+\s*$", "", q)
        q = re.sub(r'name:\s*"([^"]+)"', r"name: '\1'", q)
        q = re.sub(
            r'\(c:Course\)-\[:TEACHES\]->\(s:Skill\s*\{[^\}]+\}\)-\[:HAS_LEVEL\]->\(l:Level\s*\{[^\}]+\}\)',
            "(c:Course)-[:TEACHES]->(s:Skill {name: 'Python'}), (c)-[:HAS_LEVEL]->(l:Level {name: 'Beginner'})",
            q
        )
        q = re.sub(r'\s+', ' ', q).strip()
        q = q.rstrip('"')
        q = q.replace('\\n', ' ').strip()
        return q

    def validate_cypher(self, query: str):
        if not query:
            raise ValueError("Empty query")
        q = re.sub(r"^```cypher|```$", "", query.strip(), flags=re.IGNORECASE).strip()
        if not re.match(r'(?i)^MATCH\s', q):
            raise ValueError("Query must start with MATCH clause")
        if "RETURN" not in query.upper():
            raise ValueError("Query missing RETURN clause")
        forbidden = ["CREATE", "DELETE", "SET", "REMOVE", "MERGE"]
        for word in forbidden:
            if word in query.upper():
                raise ValueError(f"Forbidden keyword detected: {word}")


# ----------------------------------------
# KnowledgeBaseQA: Combined Workflow with SBERT (cập nhật ranking)
# ----------------------------------------
class KnowledgeBaseQA:
    def __init__(self, neo4j_conn: Neo4jConnection, embedding_model: SBERTEmbeddingModel, top_skill_k: int = 5):
        self.neo4j_conn = neo4j_conn
        self.embedding_model = embedding_model
        self.query_processor = QueryProcessor()
        self.top_skill_k = top_skill_k
        self.last_main_plan = {}
        self.last_refine_plan = {}

    def get_course_details(self, url: str) -> Dict:
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

    def _load_all_skill_embeddings(self) -> Dict[str, np.ndarray]:
        query = """
        MATCH (s:Skill)
        WHERE s.embedding_sbert IS NOT NULL
        RETURN s.name AS name, s.embedding_sbert AS emb
        """
        results = self.neo4j_conn.execute_query(query)
        skill_embs = {}
        for record in results:
            name = record.get("name")
            emb_list = record.get("emb", [])
            if name and emb_list:
                skill_embs[name] = np.array(emb_list, dtype=np.float32)
        return skill_embs

    def _load_all_course_embeddings(self) -> Dict[str, Dict[str, Any]]:
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
        results = self.neo4j_conn.execute_query(query)
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

    def _find_similar_skills(self, query: str, top_k: int = None) -> List[str]:
        top_k = top_k or self.top_skill_k
        query_emb = self.embedding_model.get_embedding(query)
        skill_embs = self._load_all_skill_embeddings()
        if not skill_embs:
            return []
        sims = []
        for skill_name, emb in skill_embs.items():
            sim = cosine_similarity(query_emb, emb)
            sims.append((skill_name, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        top_skills = [name for name, score in sims[:top_k] if score > 0.15]
        return top_skills

    def _get_courses_by_skills(self, skill_names: List[str]) -> List[Dict]:
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
        query_emb = self.embedding_model.get_embedding(query)
        all_courses = self._load_all_course_embeddings()
        sims = []
        for url, data in all_courses.items():
            emb = data["embedding"]
            sim = cosine_similarity(query_emb, emb)
            sims.append((url, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        candidate_urls = [url for url, _ in sims[:top_n]]
        return candidate_urls

    def _get_fallback_courses_embedding_only(self, query: str, exclude_urls: List[str], top_k: int = 5) -> List[Dict]:
        query_emb = self.embedding_model.get_embedding(query)
        all_courses = self._load_all_course_embeddings()
        candidates = {
            url: data for url, data in all_courses.items()
            if url not in set(exclude_urls)
        }
        sims = []
        for url, data in candidates.items():
            emb = data["embedding"]
            sim = cosine_similarity(query_emb, emb)
            sims.append((url, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_k]
        fallback = []
        for url, sim in top:
            data = candidates[url]
            fallback.append({
                "url": url,
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "rating": data.get("rating", ""),
                "duration": data.get("duration", 0),
                "skills": data.get("skills", []),
                "subjects": data.get("subjects", []),
                "similarity": float(sim),
                "source": "embedding_only"
            })
        return fallback

    def process_query(self, user_query: str) -> List[Dict]:
        """
        1. Skill-based Cypher via LLM → raw_results.
           - Nếu raw_results >= THRESHOLD, xếp hạng trong raw_results bằng cosine-similarity 
             giữa embedding course (c.embedding_sbert) và embedding query, rồi trả về top.
        2. Nếu raw_results < THRESHOLD, generate candidates bằng SBERT, refine via LLM.
        3. Nếu vẫn còn < THRESHOLD, fallback embedding-only.
        """
        try:
            # ---------- Bước 1: Skill-based lookup ----------
            top_skills = self._find_similar_skills(user_query)
            logger.info(f"Top skills from SBERT: {top_skills}")

            plan = self.query_processor.generate_query_plan(user_query, candidate_skills=top_skills)
            self.last_main_plan = plan
            cypher = plan.get('final_query', "")
            if not cypher:
                logger.warning("LLM did not return main Cypher. Skipping graph lookup.")
                logger.info("Raw LLM response:\n%s", plan)
                raw_results = []
            else:
                logger.info(f"Running main Cypher:\n{cypher}")
                raw_results = self.neo4j_conn.execute_query(cypher)

            # Bây giờ raw_results có thể nhiều record. Ta thực hiện ranking bằng embedding:
            ranked_skill_matches = []
            query_emb = self.embedding_model.get_embedding(user_query)

            # Lấy all course embeddings để so sánh
            all_course_embs = self._load_all_course_embeddings()

            for rec in raw_results:
                url = rec.get("url")
                course_data = all_course_embs.get(url)
                if course_data and course_data.get("embedding") is not None:
                    emb = course_data["embedding"]
                    sim = cosine_similarity(query_emb, emb)
                else:
                    # Nếu không có embedding_sbert cho course, gán sim = 0
                    sim = 0.0
                rec["similarity"] = float(sim)
                rec["source"] = "skill_match"
                rec.setdefault("skills", [])
                rec.setdefault("subjects", [])
                ranked_skill_matches.append(rec)

            # Sắp xếp giảm dần theo similarity
            ranked_skill_matches.sort(key=lambda x: (-x["similarity"], x.get("url", "")))

            THRESHOLD = 3
            if len(ranked_skill_matches) >= THRESHOLD:
                return ranked_skill_matches[:10]

            # Nếu không đủ, tiếp tục bước 2 và 3 tương tự như trước:
            # ---------- Bước 2: Candidate Generation by SBERT ----------
            candidate_urls = self._get_candidate_urls_by_embedding(user_query, top_n=20)
            logger.info(f"Candidate URLs (top 20 by embedding): {candidate_urls}")

            if not candidate_urls:
                return self._get_fallback_courses_embedding_only(user_query, [], top_k=5)

            # ---------- Bước 3: Refinement via LLM ----------
            refinement_plan = self.query_processor.generate_refinement_query(user_query, candidate_urls)
            self.last_refine_plan = refinement_plan
            refine_cypher = refinement_plan.get('final_query', "")
            if not refine_cypher:
                logger.warning("LLM did not return refinement Cypher. Falling back to embedding-only.")
                return self._get_fallback_courses_embedding_only(user_query, [r["url"] for r in ranked_skill_matches], top_k=5)

            logger.info(f"Running refinement Cypher:\n{refine_cypher}")
            refined_raw = self.neo4j_conn.execute_query(refine_cypher)
            refined_results = []
            for rec in refined_raw:
                rec["similarity"] = 1.0
                rec["source"] = "refined_match"
                rec.setdefault("skills", [])
                rec.setdefault("subjects", [])
                refined_results.append(rec)

            if len(refined_results) >= THRESHOLD:
                # Bạn có thể cũng ranking refined_results bằng embedding nếu muốn, nhưng mặc định gán sim=1.0
                return refined_results[:10]

            # ---------- Bước 4: Fallback embedding-only ----------
            existing_urls = [r["url"] for r in ranked_skill_matches] + [r["url"] for r in refined_results]
            fallback = self._get_fallback_courses_embedding_only(user_query, existing_urls, top_k=5)

            combined = ranked_skill_matches + refined_results + fallback
            def sort_key(x):
                order = {"skill_match": 0, "refined_match": 1, "embedding_only": 2}
                return (order.get(x.get("source"), 3), -x.get("similarity", 0.0), x.get("url", ""))
            combined.sort(key=sort_key)

            return combined[:10]

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return []
    def process_query_with_context(self, user_query: str, chat_history: list = None) -> List[Dict]:
        """Process query với context từ chat history"""
        
        # Tạo enhanced query từ context
        enhanced_query = self._enhance_query_with_context(user_query, chat_history)
        
        # Sử dụng enhanced query cho process_query hiện tại
        return self.process_query(enhanced_query)
    
    def _enhance_query_with_context(self, current_query: str, chat_history: list = None) -> str:
        """Tăng cường query với context từ lịch sử"""
        if not chat_history:
            return current_query
            
        # Tìm các chủ đề/skills đã được nhắc đến trước đó
        mentioned_topics = set()
        education_keywords = ["course", "learn", "training", "skill", "programming", "data", "web", "python", "java", "javascript"]
        
        for msg in chat_history[-10:]:  # Xem 10 tin nhắn gần nhất
            if msg["role"] == "user":
                content = msg["content"].lower()
                for keyword in education_keywords:
                    if keyword in content:
                        # Extract potential skill/topic around the keyword
                        words = content.split()
                        for i, word in enumerate(words):
                            if keyword in word:
                                # Lấy 1-2 từ xung quanh
                                start = max(0, i-1)
                                end = min(len(words), i+3)
                                topic = " ".join(words[start:end])
                                mentioned_topics.add(topic)
        
        if mentioned_topics:
            context_info = f"Previous topics discussed: {', '.join(list(mentioned_topics)[:3])}. "
            return context_info + current_query
        
        return current_query
# ---------------------------------------------
# 3. Agents
# ---------------------------------------------

class FlexibleIntentClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.educational_contexts = [
            "I want to understand what courses are available for learning programming",
            "Help me analyze the landscape of data science education options",
            "What are the characteristics of machine learning courses?",
            "Give me an overview of web development training programs",
            "I need insights about cybersecurity certification programs",
            "Help me understand the different levels of Python courses",
            "Analyze the quality and duration of available SQL training",  
            "What should I know about cloud computing course options?",
            "Give me a summary of available programming bootcamps",
            "Help me understand the landscape of online education in AI",
            "What are the best courses for learning JavaScript?",
            "Can you give me an overview of beginner machine learning courses?",
            "I'm looking for insights on advanced cybersecurity training"
        ]
        self.general_chat_contexts = [
            "Hello, how are you today?",
            "What's the weather like outside?", 
            "Tell me an interesting story",
            "I'm feeling tired today",
            "Thank you for your help",
            "What time is it now?",
            "Good morning, have a nice day",
            "Who are you?",
            "What can you help me with?",
            "Tell me about your capabilities",
            "What is your purpose?",
            "Can you tell me a joke?",
            "How are you doing?",
            "Nice to meet you",
            "What is your birthday?"
        ]
        self.scope_inquiry_contexts = [
            "What do you do?",
            "What is your scope?", 
            "Tell me about your scope",
            "What are your capabilities?",
            "What can you help with?",
            "What is this system for?",
            "What services do you provide?",
            "What is your function?"
        ]
        
        self.edu_embs = self.model.encode(self.educational_contexts, convert_to_tensor=True)
        self.chat_embs = self.model.encode(self.general_chat_contexts, convert_to_tensor=True) 
        self.scope_embs = self.model.encode(self.scope_inquiry_contexts, convert_to_tensor=True)


    def classify_intent(self, query: str, chat_history: list = None) -> dict:
        context_query = self._build_context_query(query, chat_history)
        
        q_emb = self.model.encode(context_query, convert_to_tensor=True)
        edu_sim = util.pytorch_cos_sim(q_emb, self.edu_embs)[0].max().item()
        chat_sim = util.pytorch_cos_sim(q_emb, self.chat_embs)[0].max().item()
        scope_sim = util.pytorch_cos_sim(q_emb, self.scope_embs)[0].max().item()

        # 0) If *none* of the categories fires strongly, treat as general chat
        if max(edu_sim, chat_sim, scope_sim) < 0.2:
            return {"intent":"general_chat", "confidence":"low",
                    "details":{"edu": round(edu_sim,3),
                            "chat":round(chat_sim,3),
                            "scope":round(scope_sim,3)}}

        # 1) scope inquiries
        if scope_sim > 0.4 and scope_sim >= edu_sim and scope_sim >= chat_sim:
            intent = "scope_inquiry"
            conf   = "high" if scope_sim > 0.6 else "medium"

        # 2) educational searches (raise threshold to >0.4)
        elif edu_sim > 0.4 and edu_sim >= chat_sim:
            intent = "course_search"
            conf   = "high" if edu_sim > 0.6 else "medium"

        # 3) general chat
        elif chat_sim > 0.3:
            intent = "general_chat"
            conf   = "high" if chat_sim > 0.5 else "medium"

        # 4) fallback—for anything borderline, don’t go back to LLM; call it general_chat
        else:
            intent = "general_chat"
            conf   = "medium"

        return {"intent": intent, "confidence": conf,
                "details": {"edu":round(edu_sim,3),
                            "chat":round(chat_sim,3),
                            "scope":round(scope_sim,3)}}
    
    def _build_context_query(self, current_query: str, chat_history: list = None) -> str:
        """Tạo query có context từ lịch sử chat"""
        if not chat_history:
            return current_query
            
        # Lấy 3 tin nhắn gần nhất để tạo context
        recent_messages = chat_history[-6:]  # 3 cặp user-assistant
        
        context_parts = []
        for msg in recent_messages:
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant" and msg.get("type") != "courses":
                context_parts.append(f"Assistant: {msg['content'][:100]}...")
                
        context = " ".join(context_parts[-4:])  # Giới hạn context
        return f"{context} Current query: {current_query}"

    def _llm_classify(self, query: str) -> str:
        prompt = f"""
You are an educational advisor assistant. Classify this query into one of these categories:
- "course_search": User wants searching/analysis/overview of educational courses
- "scope_inquiry": User asks about system capabilities/scope/what you can do  
- "general_chat": General conversation, greetings, casual talk

User query: "{query}"

Respond with exactly one of: course_search, scope_inquiry, general_chat
"""
        llm = OllamaLLM(small=True)
        r = llm.invoke([{"role": "user", "content": prompt}]).strip().lower()
        if "course_search" in r:
            return "course_search"
        elif "scope_inquiry" in r:
            return "scope_inquiry"
        else:
            return "general_chat"
    
class CourseAnalyzer:
    def __init__(self, llm_model="qwen3:4b"):
        self.llm_model = llm_model

    def analyze(self, courses:List[Dict], q:str)->str:
        if not courses:
            return "I couldn't find any matching courses to analyze."
        # compute stats...
        ratings = []
        for c in courses:
            r = c.get("rating", None)
            try:
                ratings.append(float(r))
            except (TypeError, ValueError):
                # skip non‐numeric ratings
                continue
        avg_rating = np.mean(ratings) if ratings else 0
        total = len(courses)
        # top skills
        skills = sum([c.get("skills",[]) for c in courses], [])
        from collections import Counter
        top_skills = [s for s,_ in Counter(skills).most_common(3)]
        # build prompt
        context = f"""
You are an educational advisor. The user asked: "{q}"
Found {total} courses, average rating {avg_rating:.1f}.
Top skills covered: {', '.join(top_skills)}.
Provide an analytical overview in 3-4 sentences, without naming any single course.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])

# ---------------------------------------------
# 4. Workflow nodes
# ---------------------------------------------
def handle_general_chat(state: dict) -> dict:
        user_query = state["question"]
        chat_history = state.get("chat_history", [])
        context = ""
        if chat_history:
            recent_context = []
            for msg in chat_history[-4:]:  # 2 cặp tin nhắn gần nhất
                if msg["role"] == "user":
                    recent_context.append(f"User previously asked: {msg['content']}")
                elif msg["role"] == "assistant" and msg.get("type") != "courses":
                    recent_context.append(f"You previously responded: {msg['content'][:150]}...")
            
            if recent_context:
                context = f"Previous conversation context:\n{chr(10).join(recent_context)}\n\n"
    
        prompt = f"""
    {context} You are Course Finder, a friendly course finding system designed to provide comprehensive overviews and summaries of educational courses to help users make informed decisions. You focus on offering insights and information about courses without recommending specific ones, empowering users to choose for themselves.
    Taking into account the conversation context if available. Maintain continuity with previous exchanges. If this appears to be a follow-up question or continuation of a previous topic, acknowledge that context appropriately. For follow-up question, if they do not ask about out-of-scope tasks or ask to introduce the scope, do not need to introduce the scope again.
    Current user message: "{user_query}"

    Respond naturally and helpfully in a conversational, warm tone, emphasizing that you are Course Finder. If they greet you, greet them back. If they ask how you are, respond appropriately. Subtly mention your role as Course Finder to guide them toward exploring course-related topics if relevant.
    If the user asks about something outside your scope, such as requesting a joke or asking for weather updates, politely decline to answer and redirect them to your educational focus. For example: 'I'm Course Finder, and I'm here to help you explore educational courses. For other topics like jokes, you might want to try another resource, but I'm happy to assist with any course-related questions!"""
        llm = OllamaLLM()
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["assistant_answer"] = response
        return state
# 3. Cập nhật node scope inquiry
def explain_system_scope(state: dict) -> dict:
    scope_explanation = """
Hi! I'm an educational course analysis assistant. Here's what I can help you with:

🎯 **My Purpose**: I provide comprehensive overviews and insights about educational courses to help you make informed decisions.

📊 **What I Do**:
- Analyze course landscapes across different subjects (programming, data science, etc.)
- Provide statistical insights about available courses 
- Give you overviews of course characteristics, levels, and quality
- Help you understand what's available in specific learning areas

🚫 **What I Don't Do**:
- I don't recommend specific individual courses
- I don't make personalized course selections for you
- I focus on giving you information so YOU can decide

💡 **How to Use Me**:
Try asking things like:
- "Help me understand the Python course landscape"
- "What should I know about data science course options?"
- "Analyze the characteristics of web development programs"

Feel free to ask me about any subject area you're interested in learning about!
"""
    state["assistant_answer"] = scope_explanation
    return state
def check_course_intent(state: dict) -> dict:
    cls = FlexibleIntentClassifier()
    chat_history = state.get("chat_history", [])
    res = cls.classify_intent(state["question"], chat_history)
    state["intent"] = res["intent"]  
    state["intent_details"] = res
    return state

def run_course_analysis(state: dict) -> dict:
    chat_history = state.get("chat_history", [])
    if chat_history:
        state["raw_courses"] = state["advisor"].process_query_with_context(
            state["question"], chat_history
        )
    else:
        state["raw_courses"] = state["advisor"].process_query(state["question"])
    return state

def analyze_courses(state:dict)->dict:
    state["assistant_answer"] = CourseAnalyzer().analyze(state["raw_courses"], state["question"])
    return state

def suggest_analysis_topics(state:dict)->dict:
    tips = [
        "Help me understand Python course landscape",
        "What should I know about data science options?",
        "Analyze web development training programs"
    ]
    prompt = "Your query isn't about course analysis. Try:\n" + "\n".join(f"- {t}" for t in tips)
    state["assistant_answer"] = OllamaLLM().invoke([{"role":"system","content":"Be friendly."},{"role":"user","content":prompt}])
    state["suggestions"] = tips
    return state

def create_course_workflow():
    wf = StateGraph(dict)
    wf.add_node("check_intent", check_course_intent)
    wf.add_node("course_query", run_course_analysis)
    wf.add_node("analyze_courses", analyze_courses)
    wf.add_node("handle_general_chat", handle_general_chat)
    wf.add_node("explain_system_scope", explain_system_scope)

    wf.set_entry_point("check_intent")
    
    # Conditional routing based on intent
    def route_intent(state):
        intent = state.get("intent")
        if intent == "course_search":
            return "course_query"
        elif intent == "scope_inquiry": 
            return "explain_system_scope"
        else:  # general_chat
            return "handle_general_chat"
    
    wf.add_conditional_edges(
        "check_intent",
        route_intent,
        {
            "course_query": "course_query",
            "explain_system_scope": "explain_system_scope", 
            "handle_general_chat": "handle_general_chat"
        }
    )
    
    wf.add_edge("course_query", "analyze_courses")
    wf.add_edge("analyze_courses", END)
    wf.add_edge("handle_general_chat", END)
    wf.add_edge("explain_system_scope", END)
    
    return wf.compile()

course_wf = create_course_workflow()

# ---------------------------------------------
# 5. Streamlit App
# ---------------------------------------------
st.set_page_config(page_title="EduAssistant", layout="wide")

# init
if 'neo4j' not in st.session_state:
    st.session_state.neo4j = Neo4jConnection("bolt://localhost:7687","neo4j","1234567890")
if 'model' not in st.session_state:
    st.session_state.model = SBERTEmbeddingModel()
if 'advisor' not in st.session_state:
    st.session_state.advisor = KnowledgeBaseQA(st.session_state.neo4j, st.session_state.model)
if 'messages' not in st.session_state:
    st.session_state.messages = []

def render_course_card(c):
    with st.container():
        st.markdown(f"### [{c['name']}]({c['url']})")
        st.caption(f"Rating: {c.get('rating','N/A')} | Skills: {', '.join(c.get('skills',[]))}")
        st.metric("Relevance", f"{c.get('similarity',0):.2f}")

def display_chat():
    for m in st.session_state.messages:
        if m["role"]=="user":
            st.chat_message("user").markdown(m["content"])
        else:
            msg = st.chat_message("assistant")
            if m.get("type")=="courses":
                st.markdown(f"🎯 Found {len(m['content'])} courses:")
                for c in m["content"]:
                    render_course_card(c)
            else:
                msg.markdown(m["content"])
import re

def clean_answer(text: str) -> str:
    """Lọc bỏ các thẻ như <think>...</think> và strip khoảng trắng"""
    # Xóa phần trong <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()
def manage_chat_history():
    """Giữ chat history trong giới hạn hợp lý"""
    MAX_MESSAGES = 20  # Giữ tối đa 20 tin nhắn (10 cặp)
    
    if len(st.session_state.messages) > MAX_MESSAGES:
        # Giữ lại tin nhắn đầu (system introduction) và các tin nhắn gần nhất
        st.session_state.messages = (
            st.session_state.messages[:2] +  # Giữ 2 tin nhắn đầu
            st.session_state.messages[-(MAX_MESSAGES-2):]  # Và các tin nhắn gần nhất
        )
def handle_query(q: str):
    st.session_state.messages.append({"role": "user", "content": q})

    # 1) Run the workflow, instrumented:
    try:
        state = {
            "question": q, 
            "advisor": st.session_state.advisor,
            "chat_history": st.session_state.messages[:-1]  # Exclude current message
        }
        
        for idx, sub_state in enumerate(course_wf.stream(state, {"recursion_limit": 10})):
            st.write(f"✔️ Completed node #{idx}: intent={sub_state.get('intent')}")
            state = sub_state
    except Exception as e:
        st.exception(e)
        logger.error("Error inside workflow.stream()", exc_info=True)
        st.stop()

    if len(state) == 1 and isinstance(next(iter(state.values())), dict):
        state = next(iter(state.values()))

    # 2) Inspect the resulting state:
    st.write("🔍 Final state:", state)

    # 3) Branch on intent
    intent = state.get("intent")
    st.write("🎯 Detected intent:", intent)

    if intent == "course_search":
        st.session_state.messages.append({
            "role": "assistant",
            "type": "courses",
            "content": state.get("raw_courses", [])
        })
        st.chat_message("assistant").markdown("🔎 I've found some courses that might interest you!")

    else:
        answer = clean_answer(state.get("assistant_answer", "❗Sorry, I cannot generate answer now"))
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": answer
        })
        st.chat_message("assistant").markdown(answer)
    manage_chat_history()


# layout
st.sidebar.title("Settings")
display_chat()
if prompt := st.chat_input("Ask about courses..."):
    handle_query(prompt)