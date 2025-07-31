import numpy as np
import re
import time
import json
from typing import List, Dict, Any
import logging
from shared.database import Neo4jConnector
from shared.embedding import SBERTEmbeddingModel
from backend.services.llm_service import LLMService
import ollama

logger = logging.getLogger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class QueryProcessor:
    def __init__(self, llm_service: LLMService, model_name: str = "qwen3:4b"):
        self.llm_service = llm_service
        self.model_name = model_name            # Giữ model_name
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL | re.MULTILINE)


    async def generate_query_plan(self, user_query: str, candidate_skills: list = None, preferences: Dict = None) -> Dict:
        start_time = time.perf_counter()
        skills_str = ""
        if candidate_skills:
            escaped = [s.replace("'", "\\'").replace('"', '\\"') for s in candidate_skills]
            quoted = [f"'{s}'" for s in escaped]
            joined = ",".join(quoted)
            skills_str = f"[{joined}]"
        pref_str = json.dumps(preferences or {})

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
13. If preferences include 'level' (e.g., 'beginner'), include MATCH (c:Course)-[:HAS_LEVEL]->(l:Level {{name: 'Beginner'}}).

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
PREFERENCES: {pref_str}

OUTPUT FORMAT (JSON):
{{
    "steps": ["step1", "step2", ...],
    "final_query": "MATCH... RETURN..."   ← must exist, or empty string
}}
"""
        try:
            # response = await self.llm_service.generate(
            #     model=self.model_name,
            #     prompt=prompt,
            #     temperature=0.1,
            #     top_k=40,
            #     top_p=0.9
            # )
            raw = await self.llm_service.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=0.1,
                top_k=40,
                top_p=0.9
            )
            logger.info(f"LLM Response: {raw}")
            plan = await self._extract_json_block(raw)
            plan['final_query'] = self.clean_cypher(plan.get('final_query', ''))
            self.validate_cypher(plan['final_query'])
            plan.setdefault('debug_info', {})['gen_time'] = time.perf_counter() - start_time
            return plan

        except Exception as e:
            logger.error("Query generation failed: %s", str(e))
            return {"steps": [], "final_query": ""}
    async def _extract_json_block(self, text: str) -> Dict:
        """
        Tìm khối JSON cuối cùng chứa "steps" và "final_query" bằng cách:
        - Xác định vị trí của "steps"
        - Tìm dấu '{' mở trước đó
        - Đếm ngoặc để tìm dấu '}' đóng tương ứng
        """
        key = '"steps"'
        idx = text.find(key)
        if idx == -1:
            raise ValueError("Không tìm thấy 'steps' trong response của LLM")
        # tìm dấu '{' gần nhất trước idx
        start = text.rfind('{', 0, idx)
        if start == -1:
            raise ValueError("Không tìm thấy dấu '{' mở của JSON")
        # parse theo ngoặc
        depth = 0
        end = None
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            raise ValueError("Không tìm thấy dấu '}' đóng của JSON")
        json_str = text[start:end+1]
        # loại bỏ backticks hoặc whitespace
        json_str = json_str.strip("` \n")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON:\n%s", json_str)
            raise
    async def generate_refinement_query(self, user_query: str, candidate_urls: List[str]) -> Dict:
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
            response = await self.llm_service.generate(prompt, temperature=0.1)
            plan = self.extract_json(response)
            if 'final_query' not in plan or not isinstance(plan['final_query'], str):
                plan['final_query'] = ""
            else:
                cleaned = self.clean_cypher(plan['final_query'])
                try:
                    self.validate_cypher(cleaned)
                    plan['final_query'] = cleaned
                except Exception:
                    plan['final_query'] = ""
            return plan
        except Exception as e:
            gen_time = time.perf_counter() - start_time
            logger.error(f"Refinement query generation failed after {gen_time:.2f}s: {str(e)}")
            return {"steps": ["Error generating refinement query"], "final_query": ""}

    def extract_json(self, text: str) -> Dict:
        text = text.replace("True", "true").replace("False", "false")
        match = self.json_pattern.search(text)
        if not match:
            raise ValueError("No JSON found in response")
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e.msg}")
            raise

    def clean_cypher(self, query: str) -> str:
        q = query.strip()
        q = re.sub(r"^```cypher|```$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r";+\s*$", "", q)
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def validate_cypher(self, query: str):
        if not query:
            raise ValueError("Empty query")
        q = re.sub(r"^```cypher|```$", "", query.strip(), flags=re.IGNORECASE).strip()
        if not re.match(r'(?i)^MATCH\s', q):
            raise ValueError("Query must start with MATCH clause")
        if "RETURN" not in query.upper():
            raise ValueError("Query missing RETURN clause")

class KnowledgeBaseQA:
    def __init__(self, neo4j_conn: Neo4jConnector, embedding_model: SBERTEmbeddingModel, llm_service: LLMService, top_skill_k: int = 5):
        self.neo4j_conn = neo4j_conn
        self.embedding_model = embedding_model
        self.query_processor = QueryProcessor(llm_service)
        self.top_skill_k = top_skill_k
        self.last_main_plan = {}
        self.last_refine_plan = {}

    async def process_query(self, user_query: str, preferences: Dict = None) -> List[Dict]:
        try:
            top_skills = self._find_similar_skills(user_query)
            plan = await self.query_processor.generate_query_plan(user_query, candidate_skills=top_skills, preferences=preferences)
            self.last_main_plan = plan
            cypher = plan.get('final_query', "")
            if not cypher:
                logger.warning("LLM did not return main Cypher. Skipping graph lookup.")
                return []
            logger.info(f"Running main Cypher:\n{cypher}")
            raw_results = self.neo4j_conn.run(cypher)
            query_emb = self.embedding_model.get_embedding(user_query)
            all_course_embs = self._load_all_course_embeddings()
            ranked_skill_matches = []
            for rec in raw_results:
                url = rec.get("url")
                course_data = all_course_embs.get(url)
                if course_data is not None and course_data.get("embedding") is not None:
                    sim = cosine_similarity(query_emb, course_data["embedding"])
                else:
                    sim = 0.0
                rec["similarity"] = float(sim)
                rec["source"] = "skill_match"
                rec.setdefault("skills", [])
                ranked_skill_matches.append(rec)
            ranked_skill_matches.sort(key=lambda x: (-x["similarity"], x.get("url", "")))
            THRESHOLD = 3
            if len(ranked_skill_matches) >= THRESHOLD:
                return ranked_skill_matches[:10]
            candidate_urls = self._get_candidate_urls_by_embedding(user_query, top_n=20)
            if not candidate_urls:
                return self._get_fallback_courses_embedding_only(user_query, [r["url"] for r in ranked_skill_matches], top_k=5)
            refinement_plan = await self.query_processor.generate_refinement_query(user_query, candidate_urls)
            self.last_refine_plan = refinement_plan
            refine_cypher = refinement_plan.get('final_query', "")
            if not refine_cypher:
                logger.warning("LLM did not return refinement Cypher. Falling back to embedding-only.")
                return self._get_fallback_courses_embedding_only(user_query, [r["url"] for r in ranked_skill_matches], top_k=5)
            logger.info(f"Running refinement Cypher:\n{refine_cypher}")
            refined_raw = self.neo4j_conn.run(refine_cypher)
            refined_results = []
            for rec in refined_raw:
                rec["similarity"] = 1.0
                rec["source"] = "refined_match"
                rec.setdefault("skills", [])
                refined_results.append(rec)
            if len(refined_results) >= THRESHOLD:
                return refined_results[:10]
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

    def _find_similar_skills(self, query: str, top_k: int = None) -> List[str]:
        top_k = top_k or self.top_skill_k
        query_emb = self.embedding_model.get_embedding(query)
        skill_embs = self._load_all_skill_embeddings()
        if not skill_embs:
            return []
        sims = [(skill_name, cosine_similarity(query_emb, emb)) for skill_name, emb in skill_embs.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in sims[:top_k] if score > 0.15]

    def _load_all_skill_embeddings(self) -> Dict[str, np.ndarray]:
        query = """
        MATCH (s:Skill)
        WHERE s.embedding_sbert IS NOT NULL
        RETURN s.name AS name, s.embedding_sbert AS emb
        """
        results = self.neo4j_conn.run(query)
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
        results = self.neo4j_conn.run(query)
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

    def _get_candidate_urls_by_embedding(self, query: str, top_n: int = 20) -> List[str]:
        query_emb = self.embedding_model.get_embedding(query)
        all_courses = self._load_all_course_embeddings()
        sims = []
        for url, data in all_courses.items():
            emb = data["embedding"]
            sim = cosine_similarity(query_emb, emb)
            sims.append((url, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [url for url, _ in sims[:top_n]]

    def _get_fallback_courses_embedding_only(self, query: str, exclude_urls: List[str], top_k: int = 5) -> List[Dict]:
        query_emb = self.embedding_model.get_embedding(query)
        all_courses = self._load_all_course_embeddings()
        candidates = {url: data for url, data in all_courses.items() if url not in set(exclude_urls)}
        sims = [(url, cosine_similarity(query_emb, data["embedding"])) for url, data in candidates.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:top_k]
        return [{"url": url, "name": candidates[url]["name"], "description": candidates[url]["description"], 
                 "rating": candidates[url]["rating"], "duration": candidates[url]["duration"], 
                 "skills": candidates[url]["skills"], "subjects": candidates[url]["subjects"], 
                 "similarity": float(sim), "source": "embedding_only"} 
                for url, sim in top]