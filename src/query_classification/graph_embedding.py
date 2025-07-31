import os
# Quiet oneDNN notices and deprecation warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import shutil
import signal
import atexit

# 1. Định nghĩa hàm clear cache
def clear_all_caches():
    # Streamlit cache
    try:
        import streamlit as st
        # với Streamlit >=1.18
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass

    # Xóa thư mục ~/.streamlit/cache
    st_cache_dir = os.path.expanduser("~/.streamlit/cache")
    if os.path.isdir(st_cache_dir):
        shutil.rmtree(st_cache_dir, ignore_errors=True)

    # Xóa cache Hugging Face
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.isdir(hf_cache_dir):
        shutil.rmtree(hf_cache_dir, ignore_errors=True)

    # (Tuỳ chọn) Xóa Windows TEMP
    temp_dir = os.getenv('TEMP', None)
    if temp_dir and os.path.isdir(temp_dir):
        # chú ý: cẩn trọng, chỉ xóa bớt file thôi
        for fname in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, fname)
            try:
                if os.path.isdir(fpath):
                    shutil.rmtree(fpath, ignore_errors=True)
                else:
                    os.remove(fpath)
            except Exception:
                pass

# 2. Đăng ký signal handler cho Ctrl+C
def _on_sigint(sig, frame):
    clear_all_caches()
    sys.exit(0)


# 3. Đăng ký atexit (cho các exit khác)
atexit.register(clear_all_caches)
import logging
logger = logging.getLogger(__name__)
import sys
import re
import time
import json
from typing import List, Dict, Any

import numpy as np
import ollama
import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from datetime import datetime
import traceback

# ----------------------------------------
# Utility: Cosine Similarity (Giữ nguyên)
# ----------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Trả về cosine similarity giữa hai vector a và b."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ----------------------------------------
# Neo4j Connection Wrapper (Giữ nguyên)
# ----------------------------------------
class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {str(e)}")
            return []

    def close(self):
        self._driver.close()


# ----------------------------------------
# Embedding Model: SBERT Thay Thế LSTM
# ----------------------------------------
class SBERTEmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Tải SBERT model
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> np.ndarray:
        """Trả về vector embedding của SBERT cho text."""
        return self.model.encode(text, convert_to_numpy=True)


# ----------------------------------------
# QueryProcessor: LLM-Based Cypher Generation (Giữ nguyên)
# ----------------------------------------
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

OUTPUT FORMAT (JSON):
{{
    "steps": ["step1", "step2", ...],
    "final_query": "MATCH... RETURN..."   ← must exist, or empty string
}}
"""
        try:
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


# ----------------------------------------
# Streamlit UI Components (giữ nguyên)
# ----------------------------------------
def render_course_card(course: Dict):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = course.get('name', 'Unnamed Course')
            url = course.get('url', '#')
            rating = course.get('rating', "No rating")
            level = course.get('level', 'N/A')
            skills = course.get('skills', [])
            similarity = course.get('similarity', 0.0)
            source = course.get('source', 'N/A')

            st.markdown(f"### [{name}]({url})")
            st.caption(f"**Level:** {level} | **Rating:** {rating} | **Source:** {source}")
            if skills:
                st.write("**Skills:** " + ", ".join(skills))

        with col2:
            st.metric("Relevance", f"{similarity:.2f}")


def render_sidebar_settings():
    with st.sidebar.expander("⚙️ SETTINGS", expanded=True):
        current_theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "System"],
            index=2,
            help="Select display theme"
        )
        st.caption(f"Version: 1.0.0 | Mode: {current_theme}")


def show_quick_actions():
    with st.container(border=True):
        st.markdown("**🚀 Quick Queries:**")
        cols = st.columns(3)
        sample_queries = [
            ("Python Basics", "Find beginner Python courses"),
            ("AWS basics", "Show basic cloud computing courses with AWS"),
            ("Top Data Science", "Data science courses with rating > 4.5")
        ]
        for col, (_, query) in zip(cols, sample_queries):
            if col.button(query, use_container_width=True):
                handle_query_submission(query)


def handle_query_submission(query: str):
    user_msg = {
        "role": "user",
        "type": "text",
        "content": query,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_msg)

    try:
        with st.spinner("🔄 Processing..."):
            start_time = time.time()
            try:
                results = st.session_state.advisor.process_query(query)
                plan    = st.session_state.advisor.last_main_plan or {}
                refine_plan  = st.session_state.advisor.last_refine_plan or {}
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("Error inside process_query or plan fetch:\n%s", tb)
                raise
            system_msg = {
                "role": "assistant",
                "type": "courses" if results else "text",
                "content": results if results else "No matching courses found",
                "metadata": {
                    "processing_time": f"{time.time()-start_time:.2f}s",
                    "query_type": "course_search",
                    "main_cypher": plan.get("final_query", ""),
                    "refine_cypher": refine_plan.get("final_query", "")
                }
            }
            st.session_state.messages.append(system_msg)
            st.rerun()

    except Exception as e:
        # show error + stack lên UI
        tb = traceback.format_exc()
        logger.error("handle_query_submission failed:\n%s", tb)
        error_msg = {
            "role": "assistant",
            "type": "text",
            "content": f"⚠️ System error: {type(e).__name__}\n```\n{tb}\n```",
        }
        st.session_state.messages.append(error_msg)

def handle_processing_error(error: Exception):
    error_msg = {
        "role": "assistant",
        "type": "text",
        "content": f"⚠️ System error: {type(error).__name__}",
        "metadata": {
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
    }
    st.session_state.messages.append(error_msg)


def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=("👤" if msg["role"] == "user" else "🤖")):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "courses":
                st.markdown(f"🎯 Found {len(msg['content'])} results:")
                for course in msg['content']:
                    render_course_card(course)
            if "metadata" in msg:
                with st.expander("🔍 Details"):
                    md = msg["metadata"]
                    if "processing_time" in md:
                        st.caption(f"🕒 Processing time: {md['processing_time']}")
                    if "query_type" in md:
                        st.caption(f"🔖 Query type: {md['query_type']}")
                    if md.get("main_cypher"):
                        st.subheader("Main Cypher")
                        st.code(md["main_cypher"], language="cypher")
                    if md.get("refine_cypher"):
                        st.subheader("Refinement Cypher")
                        st.code(md["refine_cypher"], language="cypher")



# ----------------------------------------
# Main Application (giữ nguyên)
# ----------------------------------------
def main():
    st.set_page_config(
        page_title="EduAssistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for course card hover
    st.markdown("""
    <style>
    .course-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        transition: transform 0.2s;
        background: var(--background-color);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .course-card:hover {
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    if 'neo4j' not in st.session_state:
        try:
            st.session_state.neo4j = Neo4jConnection(
                "bolt://localhost:7687",
                "neo4j",
                "1234567890"
            )
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")
            st.stop()

    if 'model' not in st.session_state:
        # Tạo SBERTEmbeddingModel thay cho LSTM
        st.session_state.model = SBERTEmbeddingModel()

    if 'advisor' not in st.session_state:
        st.session_state.advisor = KnowledgeBaseQA(
            st.session_state.neo4j,
            st.session_state.model
        )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    render_sidebar_settings()
    show_quick_actions()
    display_chat_history()

    if prompt := st.chat_input("Ask about courses..."):
        handle_query_submission(prompt)

    # Auto-scroll to bottom
    if st.session_state.messages:
        container = st.container()
        with container:
            js = """
            <script>
            window.addEventListener('DOMContentLoaded', function() {
                var messages = document.querySelector('.stChatMessage');
                messages.scrollTop = messages.scrollHeight;
            });
            </script>
            """
            st.components.v1.html(js, height=0)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    try:
        main()
    except KeyboardInterrupt:
        # This will catch CTRL+C
        clear_all_caches()
        sys.exit(0)

