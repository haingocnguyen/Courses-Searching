import re
import json
import time
import logging
from typing import Dict, List
import ollama
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes natural language queries into Neo4j Cypher queries"""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL | re.MULTILINE)

    def generate_query_plan(self, user_query: str, candidate_skills: list = None) -> Dict:
        """Generate Cypher query plan from natural language query"""
        start_time = time.perf_counter()
        
        skills_str = ""
        if candidate_skills:
            escaped = [s.replace("'", "\\'").replace('"', '\\"') for s in candidate_skills]
            quoted = [f"'{s}'" for s in escaped]
            joined = ",".join(quoted)
            skills_str = f"[{joined}]"

        prompt = self._build_cypher_prompt(user_query, skills_str)

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1}
            )
            
            plan = self.extract_json_safe(response['response'])
            
            # Validate and clean query
            if 'final_query' in plan and plan['final_query']:
                cleaned = self.clean_cypher(plan['final_query'])
                if cleaned:
                    try:
                        self.validate_cypher(cleaned)
                        plan['final_query'] = cleaned
                        logger.debug(f"Generated query successfully")
                    except Exception as e:
                        logger.warning(f"Generated Cypher failed validation: {e}")
                        plan['final_query'] = ""
                else:
                    plan['final_query'] = ""
            
            plan.setdefault('steps', ['Generated query'])
            return plan
            
        except Exception as e:
            logger.error(f"Error generating query plan: {e}")
            return {"steps": ["Error in query generation"], "final_query": ""}

    def _build_cypher_prompt(self, user_query: str, skills_str: str) -> str:
        """Build comprehensive Cypher generation prompt"""
        return f"""
You are a Neo4j Cypher expert. Generate a valid Cypher query according to these STRICT RULES:

    **BASIC QUERY RULES:**
    1. Query MUST start with MATCH.
    2. Use ONLY node labels: Course, Instructor, Level, Organization, Provider, Review, Skill, Subject
    3. Use ONLY relationships: HAS_LEVEL, HAS_REVIEW, HAS_SUBJECT, OFFERED_BY, PROVIDED_BY, TAUGHT_BY, TEACHES
    4. ALWAYS use single quotes for string values: 'value'.
    5. NEVER use double quotes "" inside the Cypher.
    6. If rating is involved, use toFloat(c.rating) instead of c.rating.
    7. MUST include RETURN clause.
    8. final_query MUST begin with MATCH (case insensitive).
    9. **NEVER use multiple WHERE clauses in one query!**
        MATCH (c:Course)-[:TAUGHT_BY]->(i:Instructor), (c)-[:TEACHES]->(s:Skill)
        WHERE i.name = 'Name' AND s.name IN ['skill1']

    **MULTIPLE CONDITIONS RULE:**
    If there are multiple conditions (skill, level, provider, etc.), you MUST use ONE of these patterns:

    Option A - Single MATCH with comma-separated patterns:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:TAUGHT_BY]->(i:Instructor)
    WHERE s.name IN ['skill1', 'skill2']

    Option B - Multiple MATCH lines with one WHERE:
    MATCH (c:Course)-[:TEACHES]->(s:Skill)
    MATCH (c)-[:TAUGHT_BY]->(i:Instructor)  
    WHERE s.name IN ['skill1', 'skill2']

    NEVER use this INVALID syntax:
    ❌ MATCH (c:Course)-[:TEACHES]->(s:Skill) WHERE condition, (other)-[:REL]->(node)
    ✅ MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:TAUGHT_BY]->(i:Instructor) WHERE condition

    **LEVEL HANDLING RULE:**
    If user query implies a course level (e.g., "Beginner", "Intermediate", "Advanced"), you MUST:
    a. First try to match via the HAS_LEVEL relationship:
    MATCH (c:Course)-[:HAS_LEVEL]->(l:Level {{name: '<LevelName>'}})
    b. ONLY if no such Level node exists, fallback to checking the course title:
    c.name CONTAINS '<level keyword>'
    c. Special rule for levels:
    - Detect keywords: Beginner, Introductory, Basics, Fundamentals... → "Beginner"
    - Intermediate, Mid-level... → "Intermediate" 
    - Advanced, Expert, Hard... → "Advanced"
    - Map keyword to exact Level.name and normalize skill name to Title Case

    **SKILL FILTERING RULE:**
    When CANDIDATE_SKILLS is provided, you MUST use ALL relevant skills from the list in the WHERE clause:
    - Analyze user query to determine which skills are relevant
    - Include ALL relevant skills: s.name IN ['skill1', 'skill2', ...]
    - Use EXACT skill names from CANDIDATE_SKILLS without modification

    **RELATIONSHIP PATTERNS:**
    For Course→X relationships (TEACHES, HAS_LEVEL, etc.), always match directly from Course:
    MATCH (c:Course)-[:REL_TYPE]->(x:NodeType {{…}}).

    **DATABASE SCHEMA:**
    Nodes:
    - Course: url, name, description, duration, rating, num_reviews, stars, has_app, has_assignment, has_discussion, has_no_enrol, has_peer, has_plugin, has_programming, has_quiz, has_reading, has_subject, has_teammate, has_ungraded, has_video, total_assignment, total_reading, total_video, type
    - Instructor: name, rating, description  
    - Organization: name, description
    - Provider: name
    - Review: comment, rating, stars
    - Skill: name, description
    - Level: name (Beginner, Intermediate, Advanced)
    - Subject: name, description

    Relationships:
    - (Course)-[:TEACHES]->(Skill)
    - (Course)-[:HAS_LEVEL]->(Level)
    - (Course)-[:OFFERED_BY]->(Organization)
    - (Course)-[:PROVIDED_BY]->(Provider)
    - (Course)-[:TAUGHT_BY]->(Instructor)
    - (Course)-[:HAS_REVIEW]->(Review)
    - (Course)-[:HAS_SUBJECT]->(Subject)

    **QUERY TYPE EXAMPLES:**

    **COURSE QUERIES:**
    USER: "Find intermediate Data Science courses from University of Michigan"
    STEPS:
    1. Match courses teaching 'Data Science' skill
    2. Match courses having Level 'Intermediate'  
    3. Match courses offered by 'University of Michigan'
    4. Return course properties

    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:HAS_LEVEL]->(l:Level {{name: 'Intermediate'}}), (c)-[:OFFERED_BY]->(o:Organization {{name: 'University of Michigan'}})
    WHERE s.name IN ['Data Science']
    RETURN c.url AS url, c.name AS name, c.duration AS duration, toFloat(c.rating) AS rating, c.description AS description

    **INSTRUCTOR QUERIES:**
    USER: "Find instructors teaching Python with rating above 4.8"
    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:TAUGHT_BY]->(i:Instructor)
    WHERE s.name IN ['Python'] AND toFloat(i.rating) > 4.8
    RETURN i.name AS instructor_name, i.rating AS instructor_rating, i.description AS instructor_description, count(c) AS courses_taught

    **ORGANIZATION QUERIES:**
    USER: "List organizations offering data science courses"
    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:OFFERED_BY]->(o:Organization)
    WHERE s.name IN ['Data Science']
    RETURN DISTINCT o.name AS organization_name, o.description AS organization_description, count(c) AS courses_offered

    **PROVIDER QUERIES:** (remember to lower the provider name before putting into query, e.g: EdX -> edx, Coursera -> coursera)
    USER: "Which providers have the most courses?"
    FINAL_QUERY:
    MATCH (c:Course)-[:PROVIDED_BY]->(p:Provider)
    RETURN p.name AS provider_name, count(c) AS total_courses
    ORDER BY total_courses DESC

    **REVIEW QUERIES:**
    USER: "Show reviews for machine learning courses"
    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:HAS_REVIEW]->(r:Review)
    WHERE s.name IN ['Machine Learning']
    RETURN c.name AS course_name, r.comment AS review_comment, r.rating AS review_rating, r.stars AS review_stars

    **SUBJECT QUERIES:**
    USER: "What subjects are available in programming?"
    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:HAS_SUBJECT]->(sub:Subject)
    WHERE s.name IN ['Programming']
    RETURN DISTINCT sub.name AS subject_name, sub.description AS subject_description, count(c) AS courses_count

    **SKILL QUERIES:**
    USER: "What programming skills can I learn?"
    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill)
    WHERE s.name CONTAINS 'Programming'
    RETURN DISTINCT s.name AS skill_name, s.description AS skill_description, count(c) AS courses_teaching

    **LEVEL QUERIES:**
    USER: "What difficulty levels are available?"
    FINAL_QUERY:
    MATCH (c:Course)-[:HAS_LEVEL]->(l:Level)
    RETURN DISTINCT l.name AS level_name, l.description AS level_description, count(c) AS courses_count

    **STATISTICAL QUERIES:**
    USER: "Average rating of Python courses by organization"
    FINAL_QUERY:
    MATCH (c:Course)-[:TEACHES]->(s:Skill), (c)-[:OFFERED_BY]->(o:Organization)
    WHERE s.name IN ['Python']
    RETURN o.name AS organization_name, avg(toFloat(c.rating)) AS avg_rating, count(c) AS course_count
    ORDER BY avg_rating DESC

    **RETURN CLAUSE GUIDELINES:** REMEMBER NOT TO RETURN ONE PROPERTY MORE THAN 1 TIME IN 1 CYPHER
    - Course queries: c.url, c.name, c.description, c.rating, c.duration, c.has_video, etc.
    - Instructor queries: i.name AS instructor_name, i.rating AS instructor_rating, i.description AS instructor_description
    - Organization queries: o.name AS organization_name, o.description AS organization_description
    - Provider queries: p.name AS provider_name
    - Review queries: r.comment AS review_comment, r.rating AS review_rating, r.stars AS review_stars
    - Subject queries: sub.name AS subject_name, sub.description AS subject_description
    - Skill queries: s.name AS skill_name, s.description AS skill_description
    - Level queries: l.name AS level_name, l.description AS level_description
    - Statistical queries: Use appropriate aggregations (count, avg, sum, etc.)

    **IMPORTANT:** MUST RETURN JSON with both "steps" and "final_query" keys.
    If you cannot build a valid Cypher, set "final_query" to empty string "".

    USER QUERY: {user_query}
    CANDIDATE_SKILLS: {skills_str}
    IMPORTANT: Use ALL relevant skills from this list in your WHERE clause.

    OUTPUT FORMAT (JSON):
    {{
        "steps": ["step1", "step2", ...],
        "final_query": "MATCH... RETURN..."
    }}
"""

    def extract_json_safe(self, text: str) -> Dict:
        """Safe JSON extraction with improved thinking tag handling"""
        try:
            logger.debug(f"🔍 RAW LLM RESPONSE:\n{text}...")
            
            # Remove thinking tags before finding JSON
            text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text_clean = text_clean.strip()
            
            # Prepare text
            text_clean = text_clean.replace("True", "true").replace("False", "false")
            text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)
            
            # Find JSON block - prefer last JSON
            json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
            matches = json_pattern.findall(text_clean)
            
            if not matches:
                logger.warning("No JSON found in cleaned LLM response")
                return {"steps": ["No valid JSON in response"], "final_query": ""}
            
            # Get last JSON match (usually the actual result)
            json_str = matches[-1]
            logger.debug(f"Extracted JSON candidate: {json_str[:200]}...")
            
            # Validate JSON has required keys
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and ("steps" in parsed or "final_query" in parsed):
                    if "final_query" not in parsed:
                        parsed["final_query"] = ""
                    if "steps" not in parsed:
                        parsed["steps"] = ["Generated query"]
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Fallback - fix common JSON issues
            json_str = self._fix_common_json_issues(json_str)
            
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    parsed.setdefault("final_query", "")
                    parsed.setdefault("steps", ["Fixed JSON parsing"])
                    return parsed
            except json.JSONDecodeError as e:
                logger.error(f"JSON still invalid after fixes: {e.msg}")
                logger.error(f"Final JSON attempt: {json_str[:300]}...")
            
            # Last resort - extract from thinking if available
            return self._extract_from_thinking_fallback(text)
                
        except Exception as e:
            logger.error(f"JSON extraction completely failed: {e}")
            return {"steps": ["JSON extraction failed"], "final_query": ""}

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        # Fix single quotes to double quotes
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # Fix unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str

    def _extract_from_thinking_fallback(self, original_text: str) -> Dict:
        """Fallback: extract info from thinking tags if JSON fails"""
        try:
            think_match = re.search(r'<think>(.*?)</think>', original_text, re.DOTALL)
            if think_match:
                thinking = think_match.group(1)
                
                steps = []
                if "step" in thinking.lower():
                    step_matches = re.findall(r'(\d+\..*?)(?=\d+\.|$)', thinking, re.DOTALL)
                    steps = [step.strip() for step in step_matches if step.strip()]
                
                if not steps:
                    steps = ["Extracted from thinking process"]
                
                query_matches = re.findall(r'MATCH.*?RETURN[^"]*', thinking, re.IGNORECASE | re.DOTALL)
                final_query = query_matches[-1] if query_matches else ""
                
                return {
                    "steps": steps,
                    "final_query": final_query,
                    "source": "thinking_fallback"
                }
        except:
            pass
        
        return {"steps": ["Fallback extraction failed"], "final_query": ""}

    def clean_cypher(self, query: str) -> str:
        """Clean and normalize Cypher query"""
        q = query.strip()
        q = re.sub(r"^```cypher|```$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r";+\s*$", "", q)
        q = re.sub(r'name:\s*"([^"]+)"', r"name: '\1'", q)
        q = re.sub(r'\s+', ' ', q).strip()
        q = q.rstrip('"')
        q = q.replace('\\n', ' ').strip()
        return q

    def validate_cypher(self, query: str):
        """Validate Cypher query structure and security"""
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