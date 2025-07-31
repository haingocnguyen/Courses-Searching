# all_in_one.py
import os, sys, shutil, atexit, logging, re, time, json, traceback
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import ollama
import streamlit as st
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Tránh warning
os.environ['OMP_NUM_THREADS'] = '1' 
st.set_page_config(
    page_title="CourseFinder",
    layout="wide",
    initial_sidebar_state="expanded"  
)
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import torch
from langgraph.graph import StateGraph, END
import asyncio
import faiss
import asyncio
import time
from typing import AsyncGenerator, Generator, Optional
import platform
import threading
import hashlib
import queue



if 'clear_requested' not in st.session_state:
    st.session_state.clear_requested = False


# Nếu vừa reload page mà clear_requested đã true (từ lần bấm hoặc reload),
# chúng ta cũng đảm bảo xóa history
if st.session_state.clear_requested:
    keys_to_clear = [
        'messages', 
        'last_results',
        'message_results',
        # ✅ CRITICAL: Clear button states
        'show_clarification_buttons',
        'clarification_query'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset flag
    st.session_state.clear_requested = False




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
logging.getLogger("neo4j.io").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def add_progress_css():
    """Add CSS for animated progress and text"""
    st.markdown("""
    <style>
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    @keyframes colorShift {
        0% { color: #ff6b6b; }
        20% { color: #ffa726; }
        40% { color: #66bb6a; }
        60% { color: #42a5f5; }
        80% { color: #ab47bc; }
        100% { color: #ff6b6b; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .animated-text {
        font-size: 14px;
        font-weight: 600;
        animation: colorShift 2s ease-in-out infinite, pulse 1.5s ease-in-out infinite;
        margin: 5px 0;
        /* FORCE animation restart */
        animation-fill-mode: both;
        animation-play-state: running;
    }
    
    .animated-text::after {
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
        color: #00d2ff;
        font-weight: bold;
        animation-fill-mode: both;
    }
    
    /* Force restart animations when content changes */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(0, 210, 255, 0.3);
        /* Key change: force re-render */
        contain: layout style paint;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 50%, #ff6b6b 100%);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .status-text {
        font-size: 13px;
        color: #666;
        margin-top: 5px;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

class ProgressTracker:
    def __init__(self, container):
        self.container = container
        self.progress_bar = None
        self.text_container = None
        self.current_progress = 0
        self.animation_counter = 0  # THÊM: Counter để force unique animations
        
    def initialize(self):
        """Initialize progress bar and text container"""
        with self.container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            self.progress_bar = st.progress(0)
            self.text_container = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
    
    def update(self, progress_percent, message, emoji="🔄"):
        """Update progress bar and animated text"""
        if self.progress_bar is None:
            self.initialize()
            
        # Update progress bar
        self.progress_bar.progress(progress_percent / 100)
        
        # FORCE animation restart với unique key
        self.animation_counter += 1
        unique_key = f"anim-{self.animation_counter}-{time.time()}"
        
        animated_html = f"""
        <div style="display: flex; align-items: center; margin: 5px 0;" key="{unique_key}">
            <span style="font-size: 16px; margin-right: 8px;">{emoji}</span>
            <span class="animated-text" style="
                animation: colorShift 2s ease-in-out infinite, pulse 1.5s ease-in-out infinite;
                animation-delay: 0s !important;
                animation-fill-mode: both !important;
                animation-play-state: running !important;
            ">{message}</span>
            <span style="margin-left: 10px; color: #888; font-size: 12px;">({progress_percent}%)</span>
        </div>
        
        <script>
        // Force restart animations on each update
        setTimeout(() => {{
            const elements = document.querySelectorAll('.animated-text');
            elements.forEach(el => {{
                // Force animation restart
                el.style.animation = 'none';
                el.offsetHeight; // Trigger reflow
                el.style.animation = 'colorShift 2s ease-in-out infinite, pulse 1.5s ease-in-out infinite';
            }});
        }}, 10);
        </script>
        """
        
        self.text_container.markdown(animated_html, unsafe_allow_html=True)
        self.current_progress = progress_percent
        time.sleep(0.1)
    
    def clear(self):
        """Clear progress display"""
        if self.container:
            self.container.empty()
            # Reset counter
            self.animation_counter = 0

def simulate_progress_step(tracker, start_percent, end_percent, message, emoji, duration=1.0):
    """Simulate progress trong 1 step với animation"""
    steps = 3  # Số bước nhỏ để tạo smooth animation
    step_size = (end_percent - start_percent) / steps
    step_duration = duration / steps
    
    for i in range(steps + 1):
        current_percent = start_percent + (i * step_size)
        tracker.update(int(current_percent), message, emoji)
        time.sleep(step_duration)



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else 0.0

class Neo4jConnection:
    def __init__(self, uri, user, pwd, max_connection_pool_size=25):
        self._driver = GraphDatabase.driver(
            uri, 
            auth=(user, pwd),
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=15.0,
            max_transaction_retry_time=15.0
        )
    
    @st.cache_data(ttl=600, max_entries=50)  # Giảm cache size để tránh memory bloat
    def execute_query_cached(_self, query_hash, q, p=None):
        """Optimized cache với read transaction"""
        try:
            # Sử dụng read transaction cho performance tốt hơn
            with _self._driver.session(default_access_mode="READ") as s:
                result = s.run(q, p or {})
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []
    
    def execute_query(self, q, p=None):
        # BỎ cache để tránh overhead
        try:
            with self._driver.session(default_access_mode="READ") as s:
                result = s.run(q, p or {})
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("Neo4j query failed: %s", e)
            return []
        
    def close(self): 
        self._driver.close()

class SBERTEmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.to('cpu')  # Ensure CPU usage for consistency
        self.model_name = model_name  # Thêm để cache dựa trên model name
        
    @st.cache_data(ttl=3600, max_entries=200)
    def get_embedding_cached(_self, text_hash: str, text: str) -> np.ndarray:
        """Cache embeddings với hash key đơn giản"""
        # Bỏ start_time và các logging không cần thiết để tăng tốc
        emb = _self.model.encode(text, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        return emb
    
    def get_embedding(self, text: str) -> np.ndarray:
        # Tạo hash đơn giản hơn
        text_hash = str(hash(text))
        return self.get_embedding_cached(text_hash, text)
    
# Global prompt cache để store mapping
PROMPT_CACHE = {}

@st.cache_data(ttl=1800, max_entries=100, hash_funcs={"builtins.str": lambda x: hashlib.md5(x.encode()).hexdigest()[:16]})
def cached_llm_invoke(content_hash: str, model_name: str, temperature: float):
    """Cache LLM responses với hash key optimized"""
    try:
        # Get original prompt from cache
        prompt = PROMPT_CACHE.get(content_hash, "")
        if not prompt:
            logger.warning(f"Prompt not found for hash {content_hash}")
            return {'response': 'Error: Prompt not found in cache'}
            
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={'temperature': temperature}
        )
        return response
    except Exception as e:
        logger.error(f"Cached LLM call failed: {e}")
        return {'response': 'Error generating response'}

class OllamaLLM:
    def __init__(self, model="qwen3:1.7b", small=True):
        self.model = model
        # Tăng temperature và thêm options để LLM nhanh hơn
        self.opts = {'temperature': 0.1 if small else 0.3}
        
    async def invoke_async(self, messages: List[Dict[str, str]]) -> str:
        start_time = time.perf_counter()
        prompt = "\n".join(m["content"] for m in messages)
        
        response = await asyncio.to_thread(ollama.generate, 
                        model=self.model, 
                        prompt=prompt, 
                        options=self.opts)
            
        logger.debug(f"LLM invocation took {time.perf_counter() - start_time:.2f} seconds")
        return response['response']
    
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join(m["content"] for m in messages)
        try:
            response = ollama.generate(model=self.model, prompt=prompt, options=self.opts)
            return response['response']
        except Exception as e:
            logger.error(f"LLM invoke failed: {e}")
            return "Sorry, I encountered an error. Please try again."
    # Giữ nguyên các methods khác...
    async def stream_async(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        prompt = "\n".join(m["content"] for m in messages)
        stream = ollama.generate(model=self.model, prompt=prompt, options=self.opts, stream=True)
        for chunk in stream:
            if chunk.get('response'):
                yield chunk['response']

    def stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        prompt = "\n".join(m["content"] for m in messages)
        stream = ollama.generate(model=self.model, prompt=prompt, options=self.opts, stream=True)
        for chunk in stream:
            if chunk.get('response'):
                yield chunk['response']
    def stream_with_patience(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Patient streaming - không cắt ngang LLM, chỉ báo status"""
        prompt = "\n".join(m["content"] for m in messages)
        
        try:
            stream = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={**self.opts, 'stream': True},
                stream=True
            )
            
            start_time = time.time()
            last_chunk_time = start_time
            has_content = False
            in_think_tag = False
            total_chunks = 0
            
            for chunk in stream:
                current_time = time.time()
                
                if chunk.get('response'):
                    content = chunk['response']
                    total_chunks += 1
                    
                    # Handle thinking tags
                    if '<think>' in content:
                        in_think_tag = True
                        content = content.split('<think>')[0]
                    if '</think>' in content:
                        in_think_tag = False
                        content = content.split('</think>')[-1]
                    if in_think_tag:
                        continue
                    
                    # First content check
                    if not has_content and content.strip():
                        initial_wait = current_time - start_time
                        if initial_wait > 20:
                            # CHỈ BÁO STATUS, KHÔNG CẮT
                            yield f"⏳ LLM took {initial_wait:.1f}s to start. Continuing...\n\n"
                        has_content = True
                    
                    # Long gap warning nhưng KHÔNG CẮT
                    if has_content:
                        gap = current_time - last_chunk_time
                        if gap > 5:  # 5s gap
                            yield f"⏳ Processing... "
                    
                    # Yield actual content
                    if content:
                        yield content
                        last_chunk_time = current_time
                
                        
        except Exception as e:
            yield f"\n\n❌ Stream error: {e}"
        
def get_llm(small=True): return OllamaLLM(small=small)

# ---------------------------------------------
class QueryProcessor:
    def __init__(self, model_name: str = "qwen3:1.7b"):
        self.model_name = model_name
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL | re.MULTILINE)

    def generate_query_plan(self, user_query: str, candidate_skills: list = None) -> Dict:
        start_time = time.perf_counter()
        
        # Tạo skills_str TRỰC TIẾP thay vì dùng cache key phức tạp
        skills_str = ""
        if candidate_skills:
            escaped = [s.replace("'", "\\'").replace('"', '\\"') for s in candidate_skills]
            quoted = [f"'{s}'" for s in escaped]
            joined = ",".join(quoted)
            skills_str = f"[{joined}]"

        prompt = f"""
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

    **PROVIDER QUERIES:** (remember to lower the provider name before putting into query)
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
            
            # Ensure required fields
            plan.setdefault('steps', ['Generated query'])
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating query plan: {e}")
            return {"steps": ["Error in query generation"], "final_query": ""}

    def extract_json_safe(self, text: str) -> Dict:
        """Safe JSON extraction với improved thinking tag handling"""
        try:
            # LOG RAW LLM RESPONSE
            logger.debug(f"🔍 RAW LLM RESPONSE:\n{text}...")
            # BƯỚC 1: Loại bỏ thinking tags TRƯỚC KHI tìm JSON
            text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text_clean = text_clean.strip()
            
            # BƯỚC 2: Chuẩn bị text
            text_clean = text_clean.replace("True", "true").replace("False", "false")
            text_clean = re.sub(r'/\*.*?\*/', '', text_clean, flags=re.DOTALL)
            
            # BƯỚC 3: Tìm JSON block - ưu tiên JSON cuối cùng
            # Pattern cải thiện để tìm JSON object hoàn chỉnh
            json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
            matches = json_pattern.findall(text_clean)
            
            if not matches:
                logger.warning("No JSON found in cleaned LLM response")
                return {"steps": ["No valid JSON in response"], "final_query": ""}
            
            # Lấy JSON match cuối cùng (thường là kết quả thực)
            json_str = matches[-1]
            logger.debug(f"Extracted JSON candidate: {json_str[:200]}...")
            
            # BƯỚC 4: Validate JSON có các keys cần thiết
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and ("steps" in parsed or "final_query" in parsed):
                    # JSON hợp lệ với structure đúng
                    if "final_query" not in parsed:
                        parsed["final_query"] = ""
                    if "steps" not in parsed:
                        parsed["steps"] = ["Generated query"]
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # BƯỚC 5: Fallback - fix common JSON issues
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
            
            # BƯỚC 6: Last resort - extract from thinking if available
            return self._extract_from_thinking_fallback(text)
                
        except Exception as e:
            logger.error(f"JSON extraction completely failed: {e}")
            return {"steps": ["JSON extraction failed"], "final_query": ""}

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        # Fix single quotes to double quotes (trong strings)
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # Fix unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix escaped quotes in final_query
        def fix_query_quotes(match):
            query_content = match.group(1)
            # Escape internal quotes properly
            query_content = query_content.replace('"', '\\"')
            return f'"final_query": "{query_content}"'
        
        json_str = re.sub(
            r'"final_query"\s*:\s*"([^"]*(?:\\"[^"]*)*)"',
            fix_query_quotes,
            json_str
        )
        
        return json_str

    def _extract_from_thinking_fallback(self, original_text: str) -> Dict:
        """Fallback: extract info from thinking tags if JSON fails"""
        try:
            # Tìm thinking content
            think_match = re.search(r'<think>(.*?)</think>', original_text, re.DOTALL)
            if think_match:
                thinking = think_match.group(1)
                
                # Extract steps from thinking
                steps = []
                if "step" in thinking.lower():
                    step_matches = re.findall(r'(\d+\..*?)(?=\d+\.|$)', thinking, re.DOTALL)
                    steps = [step.strip() for step in step_matches if step.strip()]
                
                if not steps:
                    steps = ["Extracted from thinking process"]
                
                # Try to extract query from thinking
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
@st.cache_data(ttl=3600)  # Cache 1 giờ
def load_course_embeddings_cached():
    """Load và cache course embeddings"""
    # Sử dụng connection từ cache
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

@st.cache_data(ttl=3600)  # Cache 1 giờ
def load_skill_embeddings_cached():
    """Load và cache skill embeddings"""
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
   """Build và cache FAISS indexes tối ưu cho tốc độ"""
   course_data = load_course_embeddings_cached()
   skill_data = load_skill_embeddings_cached()
   
   # Build course index
   course_index = None
   course_urls = []
   if course_data:
       embeddings = [data["embedding"] for url, data in course_data.items()]
       if embeddings:  # Kiểm tra không rỗng
           d = len(embeddings[0])
           embeddings_array = np.array(embeddings).astype('float32')
           
           # Luôn dùng FlatIP để tăng tốc - bỏ IVF phức tạp
           course_index = faiss.IndexFlatIP(d)
           faiss.normalize_L2(embeddings_array)
           course_index.add(embeddings_array)
           logger.info(f"Fast Flat index built for {len(embeddings)} courses")
               
           course_urls = list(course_data.keys())
   
   # Build skill index - tối ưu tương tự
   skill_index = None
   skill_names = []
   if skill_data:
       embeddings = list(skill_data.values())
       if embeddings:  # Kiểm tra không rỗng
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
    def __init__(self, neo4j_conn: Neo4jConnection, embedding_model: SBERTEmbeddingModel, top_skill_k: int = 5):
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
        start_time = time.perf_counter()
        desired_k = top_k or self.top_skill_k
        num_skills = len(self.skill_names)

        # never ask Faiss for more neighbors than you have vectors
        k = min(desired_k, num_skills)

        query_emb = self.embedding_model.get_embedding(query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_emb)

        if self.skill_emb_index is None or num_skills == 0:
            return []

        similarities, indices = self.skill_emb_index.search(query_emb, k)

        top_skills = []
        for sim_score, idx in zip(similarities[0], indices[0]):
            # idx should now always be < num_skills
            if sim_score > 0.5:
                top_skills.append(self.skill_names[idx])

        logger.debug(f"Skill similarity search took {time.perf_counter() - start_time:.2f} seconds")
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
        query_emb = self.embedding_model.get_embedding(query).astype(np.float32)
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # Tìm kiếm sử dụng FAISS
        similarities, indices = self.course_emb_index.search(query_emb.reshape(1, -1), top_n)
        return [self.course_urls[i] for i in indices[0]]

    def _get_fallback_courses_embedding_only(self, query: str, exclude_urls: List[str], top_k: int = 5) -> List[Dict]:
        query_emb = self.embedding_model.get_embedding(query).astype(np.float32)
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # Tìm top N + số lượng cần loại trừ
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
        """SIMPLIFIED processing - ít steps hơn, nhanh hơn"""
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
            
            # SIMPLIFIED FALLBACK: Chỉ dùng semantic search
            if query_type == "course" or not raw_results:
                logger.info(f"⚠️ Using simplified semantic fallback")
                return self._simple_semantic_search(user_query, top_k=10)
            else:
                logger.info(f"⚠️ No results for {query_type} query")
                return []

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            return []
        
    def _simple_semantic_search(self, user_query: str, top_k: int = 10) -> List[Dict]:
        """SIMPLIFIED semantic search - không cần refinement phức tạp"""
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
                
                # Build result với course details
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
            
            logger.info(f"✅ Semantic search returned {len(results)} results")
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
        """Process query with history-based context for LLM prompts but raw skill extraction."""
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

            # Ranking results by embedding similarity (as in process_query)
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
            # Fallback to embedding-only or refinement steps, mirroring process_query...
            # (You can reuse _get_candidate_urls_by_embedding, generate_refinement_query, etc.)
            return ranked  # or full fallback logic

        except Exception as e:
            logger.error(f"Error in process_query_with_context: {e}", exc_info=True)
            return []


    
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
    

def needs_clarification(query: str) -> bool:
    """Simple check - chỉ cần check intent là course_search là cần clarification"""
    return True
def extract_course_name(query: str) -> str:
    """Extract course name using LLM with simple prompt"""
    try:
        prompt = f"""
Extract the course name from this user query. Return ONLY the course name, nothing else.

User query: "{query}"

Examples:
- "tell me about Python programming course" → "Python programming"
- "find information about Introduction to Machine Learning" → "Introduction to Machine Learning" 
- "explain the Data Science Fundamentals course" → "Data Science Fundamentals"
- "java programming for beginners" → "Java programming for beginners"
- "web development course" → "Web development"
Advoid reasoning too much.
Course name:"""

        # Sử dụng LLM nhỏ và nhanh
        llm = OllamaLLM(model="qwen3:1.7b", small=True)
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # Clean response
        course_name = response.strip()
        
        # Remove quotes if present
        course_name = course_name.strip('"\'')
        
        # Remove "course" at the end if it's redundant
        if course_name.lower().endswith(' course'):
            course_name = course_name[:-7].strip()
        
        # Fallback to cleaned query if LLM returns empty or too short
        if len(course_name) < 3:
            # Simple manual cleanup as fallback
            cleaned = query.lower().strip()
            stop_words = ['tell me about', 'information about', 'explain', 'describe', 'find', 'search', 'show', 'the', 'a', 'an']
            for stop in stop_words:
                cleaned = cleaned.replace(stop, ' ')
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            course_name = cleaned if len(cleaned) > 2 else query
        
        logger.info(f"Extracted course name: '{course_name}' from query: '{query}'")
        return course_name
        
    except Exception as e:
        logger.error(f"LLM course name extraction failed: {e}")
        # Fallback to original query
        return query.strip()

def run_specific_course_search(course_name: str) -> List[Dict]:
    """Search for specific course by name with comprehensive info"""
    try:
        
        
        search_query = """
        MATCH (c:Course)
        WHERE toLower(c.name) CONTAINS toLower($course_name)
        OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
        OPTIONAL MATCH (c)-[:HAS_LEVEL]->(l:Level)
        OPTIONAL MATCH (c)-[:TAUGHT_BY]->(i:Instructor)
        OPTIONAL MATCH (c)-[:OFFERED_BY]->(o:Organization)
        OPTIONAL MATCH (c)-[:PROVIDED_BY]->(p:Provider)
        OPTIONAL MATCH (c)-[:HAS_REVIEW]->(r:Review)
        RETURN 
            c.url AS url,
            c.name AS name,
            c.description AS description,
            c.rating AS rating,
            c.duration AS duration,
            c.type AS course_type,
            l.name AS level,
            i.name AS instructor,
            o.name AS organization,
            p.name AS provider,
            collect(DISTINCT s.name) AS skills,
            collect(DISTINCT r.comment)[0..3] AS sample_reviews,
            avg(toFloat(r.rating)) AS avg_review_rating,
            count(DISTINCT r) AS review_count
        ORDER BY 
            CASE WHEN toLower(c.name) = toLower($course_name) THEN 0 ELSE 1 END,
            c.rating DESC
        LIMIT 5
        """
        
        neo4j_conn = get_neo4j_connection()
        results = neo4j_conn.execute_query(search_query, {"course_name": course_name})
        
        # Add metadata for each result
        for i, result in enumerate(results):
            result["similarity"] = 0.95 - (i * 0.05)  # High similarity for name matches
            result["source"] = "specific_course_search"
            result["query_type"] = "course"
            result["search_type"] = "specific"
            
            # Clean up None values
            for key, value in result.items():
                if value is None:
                    result[key] = "N/A"
        
        return results
        
    except Exception as e:
        logger.error(f"Specific course search failed: {e}")
        return []
def handle_specific_course_flow(original_query: str):
    """Handle specific course search flow - SUPER FIXED VERSION"""
    
    # ✅ CLEAR BUTTONS TRƯỚC KHI BẮT ĐẦU
    st.session_state.show_clarification_buttons = False
    if 'clarification_query' in st.session_state:
        del st.session_state.clarification_query
    
    with st.chat_message("assistant"):
        progress_container = st.empty()
        tracker = ProgressTracker(progress_container)
        response_container = st.empty()
        
        try:
            # Step 1: Extract course name using LLM
            simulate_progress_step(tracker, 0, 25, "Understanding course name", "🧠", 0.8)
            
            # ✅ ENHANCED course name extraction
            course_name = extract_course_name(original_query)
            
            # ✅ DOUBLE CHECK: Clean course name again
            course_name = clean_answer(course_name)
            
            # ✅ VALIDATION: Make sure course name is reasonable
            if len(course_name) < 3 or '<think>' in course_name.lower():
                logger.warning(f"Invalid course name extracted: {course_name}")
                # Use fallback extraction
                course_name = extract_course_name_fallback(original_query)
            
            tracker.update(30, f"Looking for: '{course_name}'", "🎯")
            time.sleep(0.5)
            
            # Step 2: Search database
            simulate_progress_step(tracker, 30, 60, "Searching course database", "🗄️", 1.2)
            results = run_specific_course_search(course_name)
            tracker.update(65, f"Found {len(results)} matching courses", "📊")
            time.sleep(0.3)
            
            # Step 3: Generate detailed analysis
            simulate_progress_step(tracker, 65, 95, "Analyzing course details", "🤖", 1.0)
            
            if results:
                # Get the best match (first result)
                best_match = results[0]
                
                # ✅ CLEAN extracted info display
                extracted_info = f"**Searching for:** {course_name}\n**Found:** {best_match.get('name', 'Unknown')}\n\n"
                
                # Create detailed analysis prompt - SIMPLIFIED to avoid thinking
                analysis_prompt = f"""
                Provide a comprehensive course overview for: {best_match.get('name', 'Unknown')}
                
                Key details:
                - Rating: {best_match.get('rating', 'N/A')}
                - Duration: {best_match.get('duration', 'N/A')}
                - Level: {best_match.get('level', 'N/A')}
                - Instructor: {best_match.get('instructor', 'N/A')}
                - Organization: {best_match.get('organization', 'N/A')}
                - Skills: {', '.join(best_match.get('skills', []))}
                
                Write 4-5 sentences about this course covering its key features, target audience, and value. Be direct and informative without reasoning steps.
                """
                
                tracker.update(100, "Analysis complete", "✅")
                time.sleep(0.5)
                tracker.clear()
                
                # Show extracted course name first
                response_container.markdown(extracted_info)
                
                # Then stream detailed analysis  
                analysis_container = st.empty()
                messages = [{"role":"user","content":analysis_prompt}]
                
                # ✅ ENHANCED streaming with better cleaning
                try:
                    analysis = ""
                    for chunk in st.session_state.llm.stream_with_patience(messages):
                        analysis += chunk
                        # Clean on the fly
                        display_text = clean_answer(analysis)
                        analysis_container.markdown(display_text + "▌")
                        time.sleep(0.01)
                    
                    # Final cleaning
                    analysis = clean_answer(analysis)
                    analysis_container.markdown(analysis)
                    
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    analysis = st.session_state.llm.invoke(messages)
                    analysis = clean_answer(analysis)
                    analysis_container.markdown(analysis)
                
                # Combine both parts for message history
                full_response = extracted_info + analysis
                
                # Save results with specific course flag
                st.session_state.messages.append({
                    "role":"assistant","type":"analysis_with_results",
                    "analysis":full_response,"results":results,"result_type":"course",
                    "search_type": "specific_course",
                    "extracted_course_name": course_name
                })
                st.session_state.last_results = results
                
            else:
                tracker.update(100, "No courses found", "❌")
                time.sleep(0.5)
                tracker.clear()
                
                # ✅ CLEAN no results message
                no_results_msg = f"""
**Searching for:** {course_name}

I couldn't find any courses matching '{course_name}'. This could mean:

- The exact course name might be different in our database
- Try using broader terms (e.g., "Python" instead of "Python 3.9 Advanced")
- The course might not be available in our current dataset

**Suggestions:**
- Use the "Search & Analyze Multiple Courses" option for broader results
- Try searching with just the main topic (e.g., "Python", "Data Science", "Web Development")
"""
                response_container.markdown(no_results_msg)
                st.session_state.messages.append({
                    "role":"assistant","type":"text","content":no_results_msg,
                    "extracted_course_name": course_name
                })
                st.session_state.last_results = []
                
        except Exception as e:
            tracker.clear()
            error_msg = f"❌ Error processing your request: {str(e)}"
            response_container.markdown(error_msg)
            logger.exception("Error in specific course search")
def extract_course_name_fallback(query: str) -> str:
    """Fallback extraction using pattern matching"""
    # Direct patterns for course name extraction
    patterns = [
        r'(?:course\s+)?["\']([^"\']+)["\']',  # Quoted names
        r'course\s+([A-Z][^.!?\n]*)',  # "course Name..."  
        r'about\s+(?:the\s+course\s+)?([A-Z][^.!?\n]*)',  # "about the course Name"
        r'know about\s+(?:the\s+course\s+)?([^.!?\n]*)',  # "know about Name"
        r'(?:the\s+course\s+)?([A-Z][A-Za-z\s,]+(?:Politics|Democracy|Manipulation|Programming|Science|Development)[A-Za-z\s,]*)',  # Topic-based
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Clean extracted name
            extracted = extracted.strip('.,!?')
            if len(extracted) > 3:
                return extracted
    
    # Last resort: remove common prefixes
    cleaned = query.lower()
    prefixes = ['i want to know about', 'tell me about', 'information about', 'explain', 'describe', 'find', 'search', 'show']
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # Remove "the course" prefix
    cleaned = re.sub(r'^(?:the\s+)?course\s+', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.title() if cleaned else query
def clean_answer(text: str) -> str:
    """SUPER ENHANCED clean_answer để loại bỏ thinking tags và debug info"""
    if not text:
        return text
    
    # ✅ STEP 1: Remove thinking tags (most aggressive)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # ✅ STEP 2: Remove any remaining thinking patterns
    text = re.sub(r'\*\*thinking\*\*.*?\*\*end thinking\*\*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # ✅ STEP 3: Remove debug/reasoning lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip debug/reasoning lines
        if any(skip in line.lower() for skip in [
            'let me think', 'let me see', 'i need to', 'first, i', 'okay,', 
            'the user wants', 'i should', 'looking at', 'the query',
            'the instruction says', 'i just need', 'make sure'
        ]):
            continue
        
        # Skip empty lines at start
        if not cleaned_lines and not line:
            continue
            
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # ✅ STEP 4: Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # ✅ STEP 5: Clean up whitespace
    text = text.strip()
    
    # ✅ STEP 6: If text starts with reasoning, extract the actual answer
    if any(text.lower().startswith(prefix) for prefix in ['okay', 'let me', 'first', 'the user']):
        # Try to find the actual answer after reasoning
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not any(word in sentence.lower() for word in ['let me', 'i need', 'user wants']):
                text = sentence + '.'
                break
    
    return text
# ---------------------------------------------
# 3. Agents
# ---------------------------------------------

class FlexibleIntentClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
        # Enhanced educational contexts - CHỈ academic/professional topics
        self.educational_contexts = [
            # REALISTIC course queries
            "find programming courses for beginners",
            "search for data science training programs", 
            "show me machine learning courses with high ratings",
            "I need Python courses for web development",
            "recommend beginner courses in artificial intelligence",
            "Java programming courses online",
            "web development bootcamp recommendations",
            "statistics courses for data analysis",
            "computer science degree programs",
            "cybersecurity certification courses",
            
            # Instructor queries
            "find instructors teaching machine learning",
            "who are the best Python instructors",
            "show me instructors with rating above 4.5",
            "find courses taught by specific instructor",
            "math courses taught by Jeffrey Chasnov",
            "instructors from top universities",
            
            # Organization queries
            "what organizations offer data science courses",
            "list top universities providing programming courses",
            "which universities have computer science programs",
            "show me courses from MIT or Stanford",
            "organizations offering cybersecurity training",
            "colleges with best programming courses",
            
            # Provider queries
            "which platforms have programming courses",
            "compare course providers like Coursera and edX",
            "what providers offer free programming courses",
            "show me all online course providers",
            "best platforms for learning coding",
            "coursera vs udemy for data science",
            
            # Review queries
            "show me reviews for Python programming courses",
            "what do students say about machine learning courses",
            "find programming courses with positive reviews",
            "display recent course reviews and ratings",
            "student feedback on web development courses",
            "reviews for data science bootcamps",
            
            # Subject queries
            "what subjects are covered in computer science",
            "list all programming topics available",
            "show me subjects related to AI and ML",
            "what programming topics are taught",
            "available subjects in data analysis",
            "computer science subjects and topics",
            
            # Skill queries
            "what skills can I learn from programming courses",
            "list all available programming skills",
            "show me skills taught in data science",
            "what coding skills are most popular",
            "skills needed for web development",
            "programming skills in demand",
            
            # Level queries
            "what difficulty levels are available",
            "show me beginner programming courses",
            "advanced courses in machine learning",
            "intermediate level data science courses",
            "course difficulty levels explained",
            "beginner vs advanced programming courses",
            
            # Statistical queries
            "average rating of programming courses",
            "how many data science courses are available",
            "statistics about course duration and ratings",
            "compare programming course offerings",
            "course enrollment statistics",
            "most popular programming languages in courses"
        ]
        
        # Enhanced general chat contexts - include fantasy/casual topics
        self.general_chat_contexts = [
            "hello how are you doing today",
            "good morning have a great day",
            "thank you so much for your help",
            "what time is it right now",
            "tell me a funny joke please",
            "I am feeling tired today",
            "what's the weather like outside today",
            "tell me an interesting story",
            "who are you exactly",
            "what is your name",
            "how are you feeling today",
            "nice to meet you there",
            "goodbye see you later",
            "have a wonderful day",
            "thanks for the assistance",
            # ADD fantasy/casual queries
            "is there any courses to become hero",
            "I am a super AI woman",
            "how to become a superhero",
            "courses to become a wizard",
            "training to be a ninja",
            "how to get superpowers",
            "become a dragon trainer",
            "magical powers course",
            "superhero training academy",
            "wizard school applications",
            "how to fly like superman"
        ]
        
        self.scope_inquiry_contexts = [
            "what do you do as an assistant",
            "what is your specific scope of work", 
            "tell me about your capabilities and features",
            "what are your main functions and abilities",
            "what can you help me with specifically",
            "what is this system designed for",
            "what services do you provide to users",
            "what is your primary function here",
            "explain your role and responsibilities",
            "what kind of assistance do you offer"
        ]
        
        self.edu_embs = self.model.encode(self.educational_contexts, convert_to_tensor=True)
        self.chat_embs = self.model.encode(self.general_chat_contexts, convert_to_tensor=True) 
        self.scope_embs = self.model.encode(self.scope_inquiry_contexts, convert_to_tensor=True)

    def classify_intent(self, query: str, chat_history: list = None) -> dict:
        query_lower = query.lower().strip()
        
        # ✅ PRIORITY 1: Detect fantasy/casual topics FIRST
        fantasy_patterns = [
            r'\b(hero|superhero|superpowers|magical|wizard|ninja|dragon|superman|batman|spiderman)\b',
            r'\b(super ai woman|become.*hero|fantasy|fictional|mythical|legendary)\b',
            r'\b(powers|magic|supernatural|enchanted|spell|potion)\b',
            r'\b(fly like|super speed|invisibility|telepathy|time travel)\b'
        ]
        
        for pattern in fantasy_patterns:
            if re.search(pattern, query_lower):
                return {"intent": "general_chat", "confidence": "high",
                        "details": {"edu": 0.0, "chat": 0.95, "scope": 0.0, "method": "fantasy_pattern"}}
        
        # ✅ PRIORITY 2: Academic course patterns - STRICTER criteria
        academic_course_patterns = [
            # Must have BOTH course keyword AND academic subject
            r'\b(find|search|show|get|list|display)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science|statistics|mathematics|physics|chemistry|biology|business|marketing|finance|accounting|cybersecurity|artificial intelligence)\b.*\b(course|courses|class|classes|training|program|programs|certification|bootcamp)\b',
            r'\b(course|courses|class|classes|training|program|programs|certification|bootcamp)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science|statistics|mathematics|physics|chemistry|biology|business|marketing|finance|accounting|cybersecurity|artificial intelligence)\b',
            
            # Instructor patterns with academic context
            r'\b(instructor|professor|teacher|prof|faculty)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science|statistics|mathematics|physics|chemistry|biology)\b',
            r'\b(taught by|who teaches|instructors teaching|teachers of)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science|statistics|mathematics)\b',
            
            # University/academic context
            r'\b(university|college|school|institution|academy)\b.*\b(course|courses|program|programs)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science|statistics|mathematics)\b',
            
            # Provider patterns with academic subjects
            r'\b(coursera|edx|udemy|khan academy|codecademy|pluralsight|udacity)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science)\b',
            
            # Review patterns with academic subjects
            r'\b(review|reviews|feedback|rating)\b.*\b(programming|python|java|javascript|data science|machine learning|web development|computer science)\b.*\b(course|courses)\b'
        ]
        
        # Check STRICT academic patterns
        for pattern in academic_course_patterns:
            if re.search(pattern, query_lower):
                return {"intent": "course_search", "confidence": "high",
                        "details": {"edu": 0.9, "chat": 0.1, "scope": 0.0, "method": "academic_pattern_match"}}
        
        # ✅ PRIORITY 3: Direct pattern matching for other cases
        identity_patterns = ["who are you", "what are you", "who r u", "what r u"]
        greeting_patterns = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey"]
        thanks_patterns = ["thank you", "thanks", "thx"]
        retrospective_patterns = ["what did i ask", "what was my question", "what did i say", "my previous question"]
        
        if any(pattern in query_lower for pattern in retrospective_patterns):
            return {"intent": "general_chat", "confidence": "high",
                    "details": {"edu": 0.0, "chat": 0.9, "scope": 0.0, "method": "pattern_match"}}
        
        if any(pattern in query_lower for pattern in identity_patterns):
            return {"intent": "scope_inquiry", "confidence": "high",
                    "details": {"edu": 0.0, "chat": 0.0, "scope": 0.9, "method": "pattern_match"}}
        
        if any(query_lower.startswith(pattern) for pattern in greeting_patterns):
            return {"intent": "general_chat", "confidence": "high", 
                    "details": {"edu": 0.0, "chat": 0.9, "scope": 0.0, "method": "pattern_match"}}
        
        if any(pattern in query_lower for pattern in thanks_patterns):
            return {"intent": "general_chat", "confidence": "high",
                    "details": {"edu": 0.0, "chat": 0.9, "scope": 0.0, "method": "pattern_match"}}
        
        # ✅ PRIORITY 4: Casual "course" mentions should be chat
        casual_course_patterns = [
            r'\bcourse.*\b(hero|superhero|magical|wizard|fantasy|dragon|ninja|superpowers)\b',
            r'\b(funny|weird|strange|silly|crazy|absurd|ridiculous)\b.*\bcourse\b',
            r'\bcourse.*\b(superpowers|magic|fictional|mythical|legendary)\b',
            r'\b(become|training).*\b(hero|superhero|wizard|ninja|dragon)\b'
        ]
        
        for pattern in casual_course_patterns:
            if re.search(pattern, query_lower):
                return {"intent": "general_chat", "confidence": "high",
                        "details": {"edu": 0.0, "chat": 0.9, "scope": 0.0, "method": "casual_course_pattern"}}
        
        # ✅ PRIORITY 5: Use embedding-based classification with HIGHER thresholds
        q_emb = self.model.encode(query, convert_to_tensor=True)
        edu_sim = util.pytorch_cos_sim(q_emb, self.edu_embs)[0].max().item()
        chat_sim = util.pytorch_cos_sim(q_emb, self.chat_embs)[0].max().item()
        scope_sim = util.pytorch_cos_sim(q_emb, self.scope_embs)[0].max().item()
        
        # ✅ HIGHER thresholds để tránh false positives
        if max(edu_sim, chat_sim, scope_sim) < 0.3:
            # Enhanced keyword fallback - CHỈ academic keywords + course context
            academic_keywords = [
                'programming', 'python', 'java', 'javascript', 'web development', 
                'data science', 'machine learning', 'artificial intelligence',
                'computer science', 'software engineering', 'cybersecurity',
                'statistics', 'mathematics', 'physics', 'chemistry', 'biology',
                'business', 'marketing', 'finance', 'accounting', 'economics'
            ]
            
            course_keywords = ['course', 'courses', 'class', 'classes', 'training', 'program', 'programs', 'certification', 'bootcamp']
            
            # Must have BOTH course AND academic keyword
            has_course = any(word in query_lower for word in course_keywords)
            has_academic = any(keyword in query_lower for keyword in academic_keywords)
            
            if has_course and has_academic:
                return {"intent": "course_search", "confidence": "medium",
                        "details": {"edu": 0.6, "chat": round(chat_sim, 3), "scope": round(scope_sim, 3), "method": "keyword_fallback"}}
            
            # Default to chat if ambiguous
            return {"intent": "general_chat", "confidence": "medium",
                    "details": {"edu": round(edu_sim, 3), "chat": round(chat_sim, 3), 
                               "scope": round(scope_sim, 3), "method": "embedding_fallback"}}

        # ✅ PRIORITY 6: Classification logic với HIGHER thresholds
        if scope_sim > 0.5 and scope_sim >= edu_sim and scope_sim >= chat_sim:
            intent = "scope_inquiry"
            conf = "high" if scope_sim > 0.7 else "medium"
        elif edu_sim > 0.4 and edu_sim > chat_sim * 1.2:  # edu must be significantly higher
            intent = "course_search"
            conf = "high" if edu_sim > 0.6 else "medium"
        else:
            intent = "general_chat"  # Default to chat for ambiguous cases
            conf = "high" if chat_sim > 0.5 else "medium"
        
        logger.debug(f"Intent: {intent} (edu:{edu_sim:.3f}, chat:{chat_sim:.3f}, scope:{scope_sim:.3f})")
        return {"intent": intent, "confidence": conf,
                "details": {"edu": round(edu_sim, 3), "chat": round(chat_sim, 3),
                           "scope": round(scope_sim, 3), "method": "embedding"}}

    def _build_context_query(self, current_query: str, chat_history: list = None) -> str:
        """Tạo query có context từ lịch sử chat"""
        if not chat_history:
            return current_query
            
        # Lấy 3 tin nhắn gần nhất để tạo context
        recent_messages = chat_history[-6:]  # 3 cặp user-assistant
        
        context_parts = []
        for msg in recent_messages:
            if msg["role"] == "user" and "content" in msg:
                context_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant" and msg.get("type") != "courses" and "content" in msg:
                context_parts.append(f"Assistant: {msg['content'][:100]}...")
                
        context = " ".join(context_parts[-4:])  # Giới hạn context
        return f"{context} Current query: {current_query}"


# ✅ TESTING FUNCTION
def test_fixed_intent_classifier():
    """Test cases để verify fixed intent classification"""
    classifier = FlexibleIntentClassifier()
    
    test_cases = [
        # Should be GENERAL CHAT (fantasy/casual)
        ("is there any courses to become hero?", "general_chat"),
        ("I am a super AI woman, can you provide me some course to become hero?", "general_chat"),
        ("how to become a superhero", "general_chat"),
        ("courses to become a wizard", "general_chat"),
        ("training to be a ninja", "general_chat"),
        ("superhero training academy", "general_chat"),
        ("magical powers course", "general_chat"),
        ("how to fly like superman", "general_chat"),
        
        # Should be GENERAL CHAT (regular chat)
        ("hello how are you", "general_chat"),
        ("thank you so much", "general_chat"),
        ("what time is it", "general_chat"),
        ("tell me a joke", "general_chat"),
        
        # Should be COURSE SEARCH (legitimate academic)
        ("find python programming courses", "course_search"),
        ("I need data science courses for beginners", "course_search"),
        ("show me machine learning courses", "course_search"),
        ("web development training programs", "course_search"),
        ("computer science degree programs", "course_search"),
        ("instructors teaching Java programming", "course_search"),
        ("coursera python courses", "course_search"),
        ("statistics courses for beginners", "course_search"),
        ("cybersecurity certification programs", "course_search"),
        
        # Should be SCOPE INQUIRY
        ("what can you help me with", "scope_inquiry"),
        ("what are your capabilities", "scope_inquiry"),
        ("who are you", "scope_inquiry"),
    ]
    
    print("🧪 Testing Fixed Intent Classifier:")
    print("="*50)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        result = classifier.classify_intent(query)
        actual = result["intent"]
        status = "✅" if actual == expected else "❌"
        confidence = result["confidence"]
        method = result["details"]["method"]
        
        print(f"{status} '{query}'")
        print(f"   → {actual} ({confidence}, {method})")
        if actual != expected:
            print(f"   ❌ Expected: {expected}, Got: {actual}")
            print(f"   📊 Scores: edu={result['details']['edu']}, chat={result['details']['chat']}, scope={result['details']['scope']}")
        else:
            correct += 1
        print()
    
    print(f"📊 Results: {correct}/{total} correct ({correct/total*100:.1f}%)")

class EnhancedResultHandler:
    def __init__(self, llm_model="qwen3:1.7b"):
        self.llm_model = llm_model

    def detect_query_type(self, query: str, results: List[Dict]) -> str:
        """Enhanced detection for all entity types"""
        if not results:
            return "empty"
            
        first_result = results[0]
        query_lower = query.lower()
        
        # Priority 1: Check query intent keywords
        intent_mapping = {
            "instructor": ["instructor", "teacher", "professor", "taught by", "who teaches"],
            "organization": ["organization", "university", "college", "school", "institution"],
            "provider": ["provider", "platform", "offered by", "coursera", "edx", "udemy"],
            "review": ["review", "feedback", "comment", "rating", "student says"],
            "subject": ["subject", "topic", "area", "field", "domain"],
            "skill": ["skill", "ability", "competency", "learn skill"],
            "level": ["level", "difficulty", "beginner", "intermediate", "advanced"],
            "statistical": ["average", "statistics", "how many", "count", "total", "percentage"]
        }
        
        for query_type, keywords in intent_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                # Verify results match intent
                if query_type == "instructor" and any(key in first_result for key in ["instructor_name", "instructor_rating", "i.name"]):
                    return "instructor"
                elif query_type == "organization" and any(key in first_result for key in ["organization_name", "organization_description", "o.name"]):
                    return "organization"
                elif query_type == "provider" and any(key in first_result for key in ["provider_name", "provider_description", "p.name"]):
                    return "provider"
                elif query_type == "review" and any(key in first_result for key in ["review_comment", "review_rating", "r.comment"]):
                    return "review"
                elif query_type == "subject" and any(key in first_result for key in ["subject_name", "subject_description", "sub.name"]):
                    return "subject"
                elif query_type == "skill" and any(key in first_result for key in ["skill_name", "skill_description", "s.name"]):
                    return "skill"
                elif query_type == "level" and any(key in first_result for key in ["level_name", "level_description", "l.name"]):
                    return "level"
                elif query_type == "statistical" and any(key in first_result for key in ["avg_rating", "total_courses", "course_count"]):
                    return "statistical"
        
        # Priority 2: Check result structure
        if any(key in first_result for key in ["instructor_name", "instructor_rating", "i.name"]):
            return "instructor"
        elif any(key in first_result for key in ["organization_name", "organization_description", "o.name"]):
            return "organization"
        elif any(key in first_result for key in ["provider_name", "provider_description", "p.name"]):
            return "provider"
        elif any(key in first_result for key in ["review_comment", "review_rating", "r.comment"]):
            return "review"
        elif any(key in first_result for key in ["subject_name", "subject_description", "sub.name"]):
            return "subject"
        elif any(key in first_result for key in ["skill_name", "skill_description", "s.name"]):
            return "skill"
        elif any(key in first_result for key in ["level_name", "level_description", "l.name"]):
            return "level"
        elif any(key in first_result for key in ["avg_rating", "total_courses", "course_count"]):
            return "statistical"
        elif "url" in first_result:
            return "course"
        else:
            return "mixed"

    def analyze_results(self, results: List[Dict], query: str, query_type: str = None) -> str:
        """Generate analysis based on result type"""
        if not results:
            return "I couldn't find any results matching your query. Try rephrasing or asking about a different topic."
            
        if query_type is None:
            query_type = self.detect_query_type(query, results)
            
        return self._generate_analysis_by_type(results, query, query_type)

    
    def _generate_analysis_by_type(self, results: List[Dict], query: str, query_type: str) -> str:
        """Generate specific analysis based on query type"""
        
        if query_type == "instructor":
            return self._analyze_instructor_results(results, query)
        elif query_type == "organization":
            return self._analyze_organization_results(results, query)
        elif query_type == "provider":
            return self._analyze_provider_results(results, query)
        elif query_type == "review":
            return self._analyze_review_results(results, query)
        elif query_type == "subject":
            return self._analyze_subject_results(results, query)
        elif query_type == "statistical":
            return self._analyze_statistical_results(results, query)
        elif query_type == "course":
            return self._analyze_course_results(results, query)
        else:
            return self._analyze_mixed_results(results, query)
    
    def _analyze_instructor_results(self, results: List[Dict], query: str) -> str:
        total = len(results)
        ratings = [float(r.get("instructor_rating", 0)) for r in results if r.get("instructor_rating")]
        avg_rating = np.mean(ratings) if ratings else 0
        
        top_instructors = sorted(results, key=lambda x: float(x.get("instructor_rating", 0)), reverse=True)[:3]
        
        context = f"""
User asked: "{query}"
Found {total} instructors, average rating {avg_rating:.2f}.
Top instructors: {', '.join([i.get('instructor_name', 'Unknown') for i in top_instructors])}.
Provide insights about these instructors in 3-4 sentences, focusing on their ratings and expertise areas.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_organization_results(self, results: List[Dict], query: str) -> str:
        total = len(results)
        # Get course counts if available
        course_counts = [int(r.get("courses_offered", r.get("course_count", 0))) for r in results]
        total_courses = sum(course_counts) if course_counts else 0
        
        context = f"""
User asked: "{query}"
Found {total} organizations offering {total_courses} total courses.
Top organizations by course count: {', '.join([r.get('organization_name', r.get('organization', 'Unknown')) for r in results[:3]])}.
Provide insights about these educational organizations in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_provider_results(self, results: List[Dict], query: str) -> str:
        total = len(results)
        course_counts = [int(r.get("total_courses", 0)) for r in results]
        total_courses = sum(course_counts) if course_counts else 0
        
        context = f"""
User asked: "{query}"
Found {total} course providers with {total_courses} total courses.
Major providers: {', '.join([r.get('provider_name', 'Unknown') for r in results[:5]])}.
Provide insights about the course provider landscape in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_review_results(self, results: List[Dict], query: str) -> str:
        total = len(results)
        ratings = [float(r.get("review_rating", 0)) for r in results if r.get("review_rating")]
        avg_rating = np.mean(ratings) if ratings else 0
        
        context = f"""
User asked: "{query}"
Found {total} course reviews, average rating {avg_rating:.2f}.
Reviews range from various courses. Provide insights about student feedback patterns in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_subject_results(self, results: List[Dict], query: str) -> str:
        total = len(results)
        subjects = [r.get('subject_name', 'Unknown') for r in results[:5]]
        
        context = f"""
User asked: "{query}"
Found {total} subjects: {', '.join(subjects)}.
Provide insights about these subject areas and their educational coverage in 3-4 sentences.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_statistical_results(self, results: List[Dict], query: str) -> str:
        context = f"""
User asked: "{query}"
Statistical analysis results: {len(results)} data points found.
Key metrics include ratings, course counts, and other quantitative measures.
Provide insights about these statistics in 3-4 sentences, highlighting key trends and patterns.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_course_results(self, results: List[Dict], query: str) -> str:
        # Keep existing course analysis logic
        ratings = []
        for c in results:
            r = c.get("rating", None)
            try:
                ratings.append(float(r))
            except (TypeError, ValueError):
                continue
        avg_rating = np.mean(ratings) if ratings else 0
        total = len(results)
        
        skills = sum([c.get("skills",[]) for c in results], [])
        from collections import Counter
        top_skills = [s for s,_ in Counter(skills).most_common(3)]
        
        context = f"""
User asked: "{query}"
Found {total} courses, average rating {avg_rating:.1f}.
Top skills covered: {', '.join(top_skills)}.
Provide an analytical overview in 3-4 sentences, without naming any single course.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}])
    
    def _analyze_mixed_results(self, results: List[Dict], query: str) -> str:
        total = len(results)
        context = f"""
User asked: "{query}"
Found {total} results with mixed data types including courses, instructors, organizations, and other educational entities.
Provide a comprehensive overview in 3-4 sentences about what was found.
"""
        llm = OllamaLLM(model=self.llm_model)
        return llm.invoke([{"role":"user","content":context}]) 
    
    def render_organization_results(self, results: List[Dict], query: str):
        """Render organization results"""
        st.markdown(f"### 🏛️ Organizations ({len(results)})")
        
        for i, result in enumerate(results):
            org_name = result.get("organization_name", result.get("o.name", "Unknown Organization"))
            org_description = result.get("organization_description", result.get("o.description", "No description"))
            courses_offered = result.get("courses_offered", result.get("course_count", ""))
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**🏛️ {org_name}**")
                    if courses_offered:
                        st.markdown(f"📚 Courses offered: {courses_offered}")
                    if org_description:
                        st.markdown(f"*{org_description}*")
                
                with col2:
                    st.metric("Match", f"{similarity:.1f}")
            
            if i < len(results) - 1:
                st.divider()

    def render_provider_results(self, results: List[Dict], query: str):
        """Render provider results"""
        st.markdown(f"### 🔗 Providers ({len(results)})")
        
        for i, result in enumerate(results):
            provider_name = result.get("provider_name", result.get("p.name", "Unknown Provider"))
            provider_description = result.get("provider_description", result.get("p.description", "No description"))
            courses_provided = result.get("courses_provided", result.get("course_count", ""))
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**🔗 {provider_name}**")
                    if courses_provided:
                        st.markdown(f"📚 Courses provided: {courses_provided}")
                    if provider_description:
                        st.markdown(f"*{provider_description}*")
                
                with col2:
                    st.metric("Match", f"{similarity:.1f}")
            
            if i < len(results) - 1:
                st.divider()

    def render_review_results(self, results: List[Dict], query: str):
        """Render review results"""
        st.markdown(f"### ⭐ Reviews ({len(results)})")
        
        for i, result in enumerate(results):
            review_comment = result.get("review_comment", result.get("r.comment", "No comment"))
            review_rating = result.get("review_rating", result.get("r.rating", ""))
            review_stars = result.get("review_stars", result.get("r.stars", ""))
            course_name = result.get("course_name", result.get("c.name", "Unknown Course"))
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if course_name:
                        st.markdown(f"**📖 {course_name}**")
                    
                    if review_rating:
                        try:
                            rating_val = float(review_rating)
                            star_display = "⭐" * int(rating_val)
                            st.markdown(f"Rating: {rating_val} {star_display}")
                        except:
                            st.markdown(f"Rating: {review_rating}")
                    
                    if review_comment:
                        st.markdown(f"💬 *{review_comment}*")
                
                with col2:
                    st.metric("Match", f"{similarity:.1f}")
            
            if i < len(results) - 1:
                st.divider()

    def render_subject_results(self, results: List[Dict], query: str):
        """Render subject results"""
        st.markdown(f"### 📚 Subjects ({len(results)})")
        
        for i, result in enumerate(results):
            subject_name = result.get("subject_name", result.get("sub.name", "Unknown Subject"))
            subject_description = result.get("subject_description", result.get("sub.description", "No description"))
            courses_count = result.get("courses_count", result.get("course_count", ""))
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**📚 {subject_name}**")
                    if courses_count:
                        st.markdown(f"📖 Courses available: {courses_count}")
                    if subject_description:
                        st.markdown(f"*{subject_description}*")
                
                with col2:
                    st.metric("Match", f"{similarity:.1f}")
            
            if i < len(results) - 1:
                st.divider()

    def render_skill_results(self, results: List[Dict], query: str):
        """Render skill results"""
        st.markdown(f"### 🎯 Skills ({len(results)})")
        
        for i, result in enumerate(results):
            skill_name = result.get("skill_name", result.get("s.name", "Unknown Skill"))
            skill_description = result.get("skill_description", result.get("s.description", "No description"))
            courses_teaching = result.get("courses_teaching", result.get("course_count", ""))
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**🎯 {skill_name}**")
                    if courses_teaching:
                        st.markdown(f"📖 Courses teaching this skill: {courses_teaching}")
                    if skill_description:
                        st.markdown(f"*{skill_description}*")
                
                with col2:
                    st.metric("Match", f"{similarity:.1f}")
            
            if i < len(results) - 1:
                st.divider()

    def render_level_results(self, results: List[Dict], query: str):
        """Render level results"""
        st.markdown(f"### 📊 Levels ({len(results)})")
        
        for i, result in enumerate(results):
            level_name = result.get("level_name", result.get("l.name", "Unknown Level"))
            level_description = result.get("level_description", result.get("l.description", "No description"))
            courses_count = result.get("courses_count", result.get("course_count", ""))
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**📊 {level_name}**")
                    if courses_count:
                        st.markdown(f"📖 Courses at this level: {courses_count}")
                    if level_description:
                        st.markdown(f"*{level_description}*")
                
                with col2:
                    st.metric("Match", f"{similarity:.1f}")
            
            if i < len(results) - 1:
                st.divider()

    def render_statistical_results(self, results: List[Dict], query: str):
        """Render statistical results"""
        st.markdown(f"### 📈 Statistics ({len(results)})")
        
        for result in results:
            with st.container():
                # Display all available metrics
                metrics_displayed = False
                
                # Common statistical fields
                stat_fields = {
                    "avg_rating": ("Average Rating", "⭐"),
                    "total_courses": ("Total Courses", "📚"),
                    "course_count": ("Course Count", "📖"),
                    "max_rating": ("Highest Rating", "🔝"),
                    "min_rating": ("Lowest Rating", "🔻"),
                    "total_instructors": ("Total Instructors", "👨‍🏫"),
                    "total_organizations": ("Total Organizations", "🏛️")
                }
                
                cols = st.columns(min(len([k for k in stat_fields.keys() if k in result]), 4))
                col_idx = 0
                
                for field, (label, icon) in stat_fields.items():
                    if field in result and col_idx < len(cols):
                        with cols[col_idx]:
                            value = result[field]
                            if isinstance(value, float):
                                st.metric(f"{icon} {label}", f"{value:.2f}")
                            else:
                                st.metric(f"{icon} {label}", str(value))
                        col_idx += 1
                        metrics_displayed = True
                
                # Display any other fields
                other_fields = [k for k in result.keys() if k not in stat_fields and k not in ["similarity", "source", "skills", "subjects", "query_type"]]
                if other_fields:
                    st.markdown("**Additional Data:**")
                    for field in other_fields:
                        st.write(f"• {field}: {result[field]}")

    def render_instructor_results(self, results: List[Dict], query: str):
        """Render instructor results with enhanced formatting"""
        total_count = len(results)
        
        st.markdown(f"### 👨‍🏫 Instructor Results ({total_count})")
        
        for i, result in enumerate(results):
            # Extract instructor data with fallbacks
            instructor_name = (
                result.get("instructor_name") or 
                result.get("i.name") or 
                result.get("name") or 
                "Unknown Instructor"
            )
            
            instructor_rating = (
                result.get("instructor_rating") or 
                result.get("i.rating") or 
                result.get("rating") or 
                "N/A"
            )
            
            instructor_description = (
                result.get("instructor_description") or 
                result.get("i.description") or 
                result.get("description") or 
                "No description available"
            )
            
            # Course count if available
            courses_taught = result.get("courses_taught", result.get("course_count", ""))
            
            # Similarity score
            similarity = result.get("similarity", 0)
            
            # Create instructor card
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Instructor name as header
                    st.markdown(f"**👨‍🏫 {instructor_name}**")
                    
                    # Rating display
                    if instructor_rating and instructor_rating != "N/A":
                        try:
                            rating_val = float(instructor_rating)
                            star_display = "⭐" * int(rating_val)
                            st.markdown(f"Rating: {rating_val} {star_display}")
                        except:
                            st.markdown(f"Rating: {instructor_rating}")
                    
                    # Courses taught count
                    if courses_taught:
                        st.markdown(f"📚 Courses taught: {courses_taught}")
                    
                    # Description
                    if instructor_description:
                        st.markdown(f"*{instructor_description}*")
                
                with col2:
                    # Similarity score
                    st.metric("Match", f"{similarity:.1f}", help="Relevance score")
                    
                    # Additional metadata
                    if result.get("source"):
                        st.caption(f"Source: {result['source']}")
            
            # Separator between instructors
            if i < len(results) - 1:
                st.divider()

    def render_course_results(self, results: List[Dict], query: str):
        """Enhanced course results rendering"""
        st.markdown(f"### 📚 Course Results ({len(results)})")
        
        for i, result in enumerate(results):
            course_name = result.get("name", "Unknown Course")
            course_url = result.get("url", "")
            course_description = result.get("description", "No description available")
            course_rating = result.get("rating", "N/A")
            course_duration = result.get("duration", "")
            skills = result.get("skills", [])
            similarity = result.get("similarity", 0)
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Course name with link if available
                    if course_url:
                        st.markdown(f"**📖 [{course_name}]({course_url})**")
                    else:
                        st.markdown(f"**📖 {course_name}**")
                    
                    # Skills display
                    if skills:
                        skills_display = ", ".join(skills[:3])  # Show first 3 skills
                        if len(skills) > 3:
                            skills_display += f" +{len(skills)-3} more"
                        st.markdown(f"🎯 **Skills:** {skills_display}")
                    
                    # Duration
                    if course_duration:
                        st.markdown(f"⏱️ **Duration:** {course_duration}")
                    
                    # Description (truncated)
                    if course_description:
                        desc_preview = course_description[:200] + "..." if len(course_description) > 200 else course_description
                        st.markdown(f"📝 *{desc_preview}*")
                
                with col2:
                    # Rating
                    if course_rating and course_rating != "N/A":
                        try:
                            rating_val = float(course_rating)
                            st.metric("Rating", f"{rating_val:.1f}⭐")
                        except:
                            st.metric("Rating", course_rating)
                    
                    # Similarity score
                    st.metric("Match", f"{similarity:.1f}", help="Relevance score")
            
            # Separator between courses
            if i < len(results) - 1:
                st.divider()


class CourseAnalyzer:
    def __init__(self, llm_model="qwen3:1.7b"):
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
def handle_general_search(original_query: str):
    """Handle general multi-course search - PLACEHOLDER"""
    
    # Clear buttons
    st.session_state.show_clarification_buttons = False
    if 'clarification_query' in st.session_state:
        del st.session_state.clarification_query
    
    with st.chat_message("assistant"):
        progress_container = st.empty()
        tracker = ProgressTracker(progress_container)
        response_container = st.empty()
        
        try:
            # Step 1: Process query
            simulate_progress_step(tracker, 0, 30, "Analyzing query for multiple courses", "🔍", 1.0)
            
            # Step 2: Search database
            simulate_progress_step(tracker, 30, 70, "Searching course database", "🗄️", 1.5)
            
            # Use existing process_query logic
            results = st.session_state.advisor.process_query(original_query)
            
            simulate_progress_step(tracker, 70, 100, "Analyzing results", "📊", 1.0)
            
            if results:
                # Generate analysis
                handler = EnhancedResultHandler()
                analysis = handler.analyze_results(results, original_query)
                analysis = clean_answer(analysis)
                
                tracker.clear()
                response_container.markdown(analysis)
                
                # Save results
                st.session_state.messages.append({
                    "role":"assistant","type":"analysis_with_results",
                    "analysis":analysis,"results":results,
                    "result_type":handler.detect_query_type(original_query, results),
                    "search_type": "general_search"
                })
                st.session_state.last_results = results
                
            else:
                tracker.clear()
                no_results_msg = f"I couldn't find any courses matching '{original_query}'. Try rephrasing your query or using different keywords."
                response_container.markdown(no_results_msg)
                st.session_state.messages.append({
                    "role":"assistant","type":"text","content":no_results_msg
                })
                
        except Exception as e:
            tracker.clear()
            response_container.markdown(f"❌ Error: {e}")
            logger.exception("Error in general search")
def handle_general_chat(state: dict) -> dict:
    user_query = state["question"]
    chat_history = state.get("chat_history", [])
    
    # Chỉ giữ một vài quick responses cơ bản nhất
    quick_responses = {
        "hello": "Hello! I'm Course Finder, ready to help you explore educational courses.",
        "hi": "Hi there! I'm Course Finder, your educational course analysis assistant.",
        "thank you": "You're welcome! Feel free to ask about any course topics.",
        "thanks": "You're welcome! Let me know if you need help with course analysis.",
    }
    
    query_lower = user_query.lower().strip()
    
    # Chỉ check exact matches cho những greeting cơ bản
    if query_lower in quick_responses:
        state["assistant_answer"] = quick_responses[query_lower]
        return state
    
    # Build context như cũ nhưng compact hơn
    context = ""
    if chat_history:
        recent = chat_history[-2:]  # Chỉ lấy 1 cặp gần nhất
        for msg in recent:
            if msg.get("role") == "user" and "content" in msg:
                context += f"User previously: {msg['content'][:50]}...\n"
            elif msg.get("role") == "assistant" and msg.get("type") != "courses" and "content" in msg:
                snippet = msg["content"][:50]
                context += f"Assistant: {snippet}...\n"
        if context:
            context = f"Recent context:\n{context}\n"

    # Prompt cải thiện để handle chat history
    prompt = f"""{context}You are Course Finder, a friendly course finding system designed to provide comprehensive overviews and summaries of educational courses to help users make informed decisions. You focus on offering insights and information about courses without recommending specific ones, empowering users to choose for themselves.

Current user message: "{user_query}"

Key points:
- If they ask about conversation history, chat history, or past queries, acknowledge that you can see recent messages in our conversation and show chat history
- Respond naturally and helpfully in a conversational, warm tone
- If they greet you, greet them back. If they ask how you are, respond appropriately  
- If the user asks about something outside your scope, politely decline and redirect back to course topics, say that you are just able to help for course searching only.
- Keep responses concise but warm (2-3 sentences max for general chat)
- Avoid long explanations and limit unless specifically needed
- Don't overthink responses - be direct and friendly
"""

    state["assistant_prompt"] = prompt
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

def run_query_analysis(state: dict) -> dict:
    """Run analysis for any type of educational query"""
    chat_history = state.get("chat_history", [])
    
    try:
        if chat_history:
            # Giới hạn chat history để tăng tốc
            limited_history = chat_history[-4:]  # Chỉ 2 cặp gần nhất
            raw_results = state["advisor"].process_query_with_context(
                state["question"], limited_history
            )
        else:
            raw_results = state["advisor"].process_query(state["question"])
        
        state["raw_results"] = raw_results
        return state
        
    except Exception as e:
        logger.error(f"Error in query analysis: {e}")
        state["raw_results"] = []
        state["error"] = f"Error processing query: {str(e)}"
        return state

def analyze_results(state: dict) -> dict:
    """Enhanced analysis for all types of results"""
    results = state.get("raw_results", [])
    query = state["question"]
    
    # Use enhanced result handler
    handler = EnhancedResultHandler()
    analysis = handler.analyze_results(results, query)
    
    state["assistant_answer"] = analysis
    state["result_type"] = handler.detect_query_type(query, results)
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

def create_enhanced_workflow():
    wf = StateGraph(dict)
    wf.add_node("check_intent", check_course_intent)  # Giữ nguyên tên để tương thích
    wf.add_node("educational_query", run_query_analysis)  # Renamed from course_query
    wf.add_node("analyze_results", analyze_results)  # Renamed from analyze_courses
    wf.add_node("handle_general_chat", handle_general_chat)
    wf.add_node("explain_system_scope", explain_system_scope)

    wf.set_entry_point("check_intent")
    
    def route_intent(state):
        intent = state.get("intent")
        if intent == "course_search":  # This now covers ALL educational queries
            return "educational_query"
        elif intent == "scope_inquiry": 
            return "explain_system_scope"
        else:
            return "handle_general_chat"
    
    wf.add_conditional_edges(
        "check_intent",
        route_intent,
        {
            "educational_query": "educational_query",
            "explain_system_scope": "explain_system_scope", 
            "handle_general_chat": "handle_general_chat"
        }
    )
    
    wf.add_edge("educational_query", "analyze_results")
    wf.add_edge("analyze_results", END)
    wf.add_edge("handle_general_chat", END)
    wf.add_edge("explain_system_scope", END)
    
    return wf.compile()

course_wf = create_enhanced_workflow()

# ---------------------------------------------
# 5. Streamlit App - Complete Optimized Version
# ---------------------------------------------

import re
import time
from typing import List, Dict

# Memory và configuration optimizations
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Tránh warning
os.environ['OMP_NUM_THREADS'] = '1'  # Limit CPU threads

SYSTEM_PROMPT = """You are Course Finder, a friendly course finding system designed to provide comprehensive overviews and summaries of educational courses to help users make informed decisions. You focus on offering insights and information about courses without recommending specific ones, empowering users to choose for themselves.

Key behaviors:
- When users ask about conversation history, chat history, or past queries, acknowledge that you can see recent messages and offer to help with course topics
- If they greet you, greet them back warmly
- If they ask how you are, respond appropriately 
- If the user asks about something outside your scope, politely decline and redirect back to course topics, say that you are just able to help for course searching only.
- Keep responses conversational, warm, and helpful
- Avoid long explanations unless specifically needed
- Don't overthink responses - be direct and friendly

Respond naturally without excessive reasoning or lengthy explanations."""

# ---------------------------------------------
# Cached Resource Functions
# ---------------------------------------------

@st.cache_resource
def get_neo4j_connection():
    """Cache Neo4j connection"""
    return Neo4jConnection("bolt://localhost:7687", "neo4j", "1234567890")

@st.cache_resource
def get_embedding_model():
    """Cache embedding model - chỉ load 1 lần"""
    return SBERTEmbeddingModel("all-MiniLM-L6-v2")

@st.cache_resource
def get_knowledge_base_qa():
    """Cache toàn bộ KnowledgeBaseQA system"""
    neo4j_conn = get_neo4j_connection()
    embedding_model = get_embedding_model()
    return KnowledgeBaseQA(neo4j_conn, embedding_model)

@st.cache_resource
def get_cached_llm():
    """Cache LLM instance"""
    return get_llm(small=True)

# ---------------------------------------------
# Session State Initialization
# ---------------------------------------------

# Clear chat history handling
if 'clear_requested' not in st.session_state:
    st.session_state.clear_requested = False


# Nếu vừa reload page mà clear_requested đã true, xóa history
if st.session_state.clear_requested:
    keys_to_clear = [
        'messages', 
        'last_results',
        'message_results',
        # ✅ CRITICAL: Clear button states
        'show_clarification_buttons',
        'clarification_query'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset flag
    st.session_state.clear_requested = False

# Initialize cached components
if 'advisor' not in st.session_state:
    with st.spinner("🚀 Initializing system..."):
        st.session_state.advisor = get_knowledge_base_qa()

if 'llm' not in st.session_state:
    st.session_state.llm = get_cached_llm()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'message_results' not in st.session_state:
    st.session_state.message_results = {}  # Dict để lưu results theo message_id
if 'show_clarification_buttons' not in st.session_state:
    st.session_state.show_clarification_buttons = False
if 'clarification_query' not in st.session_state:
    st.session_state.clarification_query = ""
# ---------------------------------------------
# UI Helper Functions
# ---------------------------------------------

def clear_duplicate_ui():
    duplicate_keys = [
        'current_results_displayed',
        'duplicate_results',
        'temp_analysis',
        'pending_results',
        'pending_query',
        'button_clicked',
        'processing_specific',
        'processing_general',
        'show_clarification_buttons',
        'clarification_query'
    ]
    
    for key in duplicate_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Force UI refresh nếu cần
    try:
        st.cache_data.clear()
    except:
        pass

def clean_answer(text: str) -> str:
    """Lọc bỏ các thẻ như <think>...</think> và strip khoảng trắng"""
    # Xóa phần trong <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def manage_chat_history():
    """Optimized chat history management"""
    MAX_MESSAGES = 6  
    
    if len(st.session_state.messages) > MAX_MESSAGES:
        clear_duplicate_ui()  # Clear UI khi trim messages
        # Chỉ giữ các tin nhắn gần nhất
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

def render_result_card(result, result_type):
    """Render different types of results based on type with improved layout"""
    if result_type == "course":
        col1, col2 = st.columns([3, 1])
        with col1:
            if result.get('url'):
                st.markdown(f"**[{result['name']}]({result['url']})**")
            else:
                st.markdown(f"**{result['name']}**")
            
            if result.get('skills'):
                st.caption(f"Skills: {', '.join(result.get('skills',[])[:3])}...")  # Limit skills shown
        
        with col2:
            if result.get('rating'):
                st.metric("Rating", result.get('rating','N/A'), label_visibility="collapsed")
            if 'similarity' in result:
                st.caption(f"Match: {result.get('similarity',0):.1f}")
                
        if result.get('description'):
            st.write("**Description**")
            st.write(result['description'])

    elif result_type == "instructor":
        st.markdown(f"**👨‍🏫 {result.get('instructor_name', 'Unknown Instructor')}**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Rating: {result.get('instructor_rating','N/A')}")
        with col2:
            st.caption(f"Courses: {result.get('courses_taught', 'N/A')}")
        
        if result.get('instructor_description'):
            st.write("**Description**")
            st.write(result['instructor_description'])
                
    elif result_type == "organization":
        st.markdown(f"**🏫 {result.get('organization_name', result.get('organization', 'Unknown'))}**")
        st.caption(f"Courses offered: {result.get('courses_offered', result.get('course_count', 'N/A'))}")
        
        if result.get('organization_description'):
            st.write("**Description**")
            st.write(result['organization_description'])
                
    elif result_type == "provider":
        st.markdown(f"**🌐 {result.get('provider_name', 'Unknown Provider')}**")
        st.metric("Total Courses", result.get('total_courses', 'N/A'), label_visibility="collapsed")
        
    elif result_type == "review":
        st.markdown(f"**💬 Review for {result.get('course_name', 'Course')}**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Rating: {result.get('review_rating','N/A')}")
        with col2:
            st.caption(f"Stars: {result.get('review_stars', 'N/A')}")
        
        if result.get('review_comment'):
            st.write("**Review Comment**")
            st.write(result['review_comment'])
                
    elif result_type == "subject":
        st.markdown(f"**📚 {result.get('subject_name', 'Unknown Subject')}**")
        if result.get('subject_description'):
            st.write("**Description**")
            st.write(result['subject_description'])
                
    elif result_type == "statistical":
        st.markdown("**📊 Statistics**")
        cols = st.columns(min(3, len([k for k in result.keys() if k not in ['subject_name', 'organization_name', 'provider_name']])))
        
        col_idx = 0
        for key, value in result.items():
            if key not in ['subject_name', 'organization_name', 'provider_name'] and col_idx < len(cols):
                with cols[col_idx]:
                    if isinstance(value, (int, float)):
                        st.metric(key.replace('_', ' ').title(), 
                                f"{value:.1f}" if isinstance(value, float) else value,
                                label_visibility="visible")
                    else:
                        st.caption(f"{key.replace('_', ' ').title()}")
                        st.write(str(value))
                col_idx += 1
    else:
        st.markdown("**📋 Result**")
        important_fields = ['name', 'title', 'rating', 'url']
        
        for field in important_fields:
            if field in result:
                if field == 'url':
                    st.markdown(f"[View Details]({result[field]})")
                else:
                    st.write(f"**{field.title()}:** {result[field]}")
        
        other_fields = {k: v for k, v in result.items() if k not in important_fields + ['similarity', 'source']}
        if other_fields:
            st.write("**Additional Details**")
            for key, value in other_fields.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
def display_chat():
    """Display chat history WITH results for each message - NO DUPLICATES"""
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        message_id = f"msg_{i}"
        
        with st.chat_message(role):
            if role == "user":
                st.markdown(message["content"])
                
            elif role == "assistant":
                msg_type = message.get("type", "text")
                
                if msg_type == "analysis_with_results":
                    # Show analysis text
                    analysis = message.get("analysis", "")
                    if analysis:
                        st.markdown(analysis)
                    
                    # ✅ CHỈ HIỂN THỊ RESULTS CHO MESSAGE CUỐI CÙNG
                    # Tránh duplicate results từ các query cũ
                    is_latest_result_message = (i == len(st.session_state.messages) - 1)
                    
                    if is_latest_result_message:
                        results = message.get("results", [])
                        result_type = message.get("result_type", "course")
                        
                        if results:
                            st.markdown("---")  # Separator
                            st.markdown(f"### 📊 Results ({len(results)})")
                            
                            # Display results in expandable format
                            for j, r in enumerate(results[:8]):  # Limit to 8 results per query
                                if result_type == "course":
                                    name = r.get('name', f'Course {j+1}')
                                    with st.expander(f"📖 {name}", expanded=(j==0)):
                                        render_result_card(r, result_type)
                                elif result_type == "instructor":
                                    name = r.get('instructor_name', r.get('i.name', f'Instructor {j+1}'))
                                    with st.expander(f"👨‍🏫 {name}", expanded=(j==0)):
                                        render_result_card(r, result_type)
                                elif result_type == "organization":
                                    name = r.get('organization_name', r.get('o.name', f'Organization {j+1}'))
                                    with st.expander(f"🏫 {name}", expanded=(j==0)):
                                        render_result_card(r, result_type)
                                elif result_type == "provider":
                                    name = r.get('provider_name', r.get('p.name', f'Provider {j+1}'))
                                    with st.expander(f"🌐 {name}", expanded=(j==0)):
                                        render_result_card(r, result_type)
                                else:
                                    with st.expander(f"📋 Result {j+1}", expanded=(j==0)):
                                        render_result_card(r, result_type)
                            
                            # Show "View more" if there are more than 8 results
                            if len(results) > 8:
                                st.info(f"... and {len(results) - 8} more results")
                    else:
                        # CHO CÁC MESSAGE CŨ, CHỈ HIỂN THỊ SUMMARY
                        results = message.get("results", [])
                        result_type = message.get("result_type", "course")
                        if results:
                            st.info(f"📊 Found {len(results)} {result_type}(s) - archived results")
                        
                elif msg_type in ["text", "courses"]:
                    content = message.get("content", "")
                    if content:
                        st.markdown(content)
                else:
                    # Fallback for other message types
                    content = message.get("content", "")
                    if content:
                        st.markdown(content)
def display_results(results, result_type="course"):
    # Dynamic headers
    if result_type == "instructor":
        st.markdown("**👨‍🏫 Instructor Results**")
        icon = "👨‍🏫"
        entity_name = "Instructor"
    elif result_type == "organization":
        st.markdown("**🏫 Organization Results**")
        icon = "🏫"
        entity_name = "Organization"
    elif result_type == "provider":
        st.markdown("**🌐 Provider Results**")
        icon = "🌐"
        entity_name = "Provider"
    else:
        st.markdown("**📚 Course Results**")
        icon = "📖"
        entity_name = "Course"
    
    for i, result in enumerate(results):
        st.markdown(f"{icon} {entity_name} {i+1}")
        
        if result_type == "instructor":
            # Instructor-specific display
            name = result.get('instructor_name', result.get('i.name', result.get('name', 'Unknown')))
            rating = result.get('instructor_rating', result.get('i.rating', 'N/A'))
            
            # Get courses taught
            courses_taught = result.get('courses_taught', result.get('course_names', []))
            if isinstance(courses_taught, list):
                courses_display = ', '.join(courses_taught[:3])  # Show first 3
                if len(courses_taught) > 3:
                    courses_display += f" (+{len(courses_taught)-3} more)"
            else:
                courses_display = str(courses_taught) if courses_taught else "N/A"
            
            st.markdown(f"**👨‍🏫 {name}**")
            st.markdown(f"Rating: {rating}")
            st.markdown(f"Teaches: {courses_display}")
            
        elif result_type == "organization":
            name = result.get('organization_name', result.get('o.name', result.get('name', 'Unknown')))
            courses_offered = result.get('courses_offered', result.get('course_count', 'N/A'))
            st.markdown(f"**🏫 {name}**")
            st.markdown(f"Courses Offered: {courses_offered}")
            
        else:
            # Course display (original)
            name = result.get('name', 'Unknown')
            rating = result.get('rating', 'N/A')
            st.markdown(f"**📖 {name}**")
            st.markdown(f"Rating: {rating}")

def add_multi_entity_debug_sidebar():
    """Add comprehensive debug functionality"""
    with st.sidebar:
        st.markdown("### 🔧 Multi-Entity Debug")
        
        entity_tests = {
            "Instructors": "MATCH (c:Course)-[:TAUGHT_BY]->(i:Instructor) RETURN i.name AS instructor_name, count(c) AS courses_taught LIMIT 3",
            "Organizations": "MATCH (c:Course)-[:OFFERED_BY]->(o:Organization) RETURN o.name AS organization_name, count(c) AS courses_offered LIMIT 3",
            "Providers": "MATCH (c:Course)-[:PROVIDED_BY]->(p:Provider) RETURN p.name AS provider_name, count(c) AS courses_provided LIMIT 3",
            "Reviews": "MATCH (c:Course)-[:HAS_REVIEW]->(r:Review) RETURN r.comment AS review_comment, r.rating AS review_rating LIMIT 3",
            "Subjects": "MATCH (c:Course)-[:HAS_SUBJECT]->(sub:Subject) RETURN sub.name AS subject_name, count(c) AS courses_count LIMIT 3",
            "Skills": "MATCH (c:Course)-[:TEACHES]->(s:Skill) RETURN s.name AS skill_name, count(c) AS courses_teaching LIMIT 3",
            "Levels": "MATCH (c:Course)-[:HAS_LEVEL]->(l:Level) RETURN l.name AS level_name, count(c) AS courses_count LIMIT 3"
        }
        
        selected_test = st.selectbox("Test Entity:", list(entity_tests.keys()))
        
        if st.button("Run Test"):
            try:
                neo4j_conn = get_neo4j_connection()
                results = neo4j_conn.execute_query(entity_tests[selected_test])
                
                st.write(f"**{selected_test} Test Results:**")
                if results:
                    for result in results:
                        st.json(result)
                else:
                    st.write(f"No {selected_test.lower()} found")
                    
            except Exception as e:
                st.error(f"Test failed: {e}")
# SỬA stream_llm_response để handle flexible message formats
def stream_llm_response(llm, user_messages: List[Dict[str, str]], placeholder):
    """Stream với timeout protection"""
    
    # Format messages như cũ
    if len(user_messages) == 1 and "role" not in user_messages[0]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                   {"role": "user", "content": user_messages[0]["content"]}]
    elif all("role" in msg for msg in user_messages):
        if user_messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
        else:
            messages = user_messages
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in user_messages:
            if isinstance(msg, dict) and "content" in msg:
                messages.append({"role": "user", "content": msg["content"]})

    full_response = ""
    
    try:
        for chunk in llm.stream_with_patience(messages):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        placeholder.markdown("⚠️ Falling back to standard response...")
        full_response = llm.invoke(messages)
    
    # Final cleanup
    full_response = clean_answer(full_response)
    placeholder.markdown(full_response)
    return full_response
# ---------------------------------------------
# Main Query Handler
# ---------------------------------------------

def handle_query(q: str):
    """Process user question với clarification buttons - FIXED VERSION with immediate button display"""
    
    # Clear any duplicate UI elements
    clear_duplicate_ui()
    
    # ========= FIX 1: XỬ LÝ BUTTON RESPONSE ĐÚNG CÁCH =========
    
    # Check if this is a button response
    if q.startswith("CLARIFY_"):
        choice = q.replace("CLARIFY_", "").split("_")[0]
        original_query = q.replace(f"CLARIFY_{choice}_", "")
        
        # ✅ QUAN TRỌNG: Clear clarification buttons NGAY LẬP TỨC
        st.session_state.show_clarification_buttons = False
        if 'clarification_query' in st.session_state:
            del st.session_state.clarification_query
        
        # ✅ KHÔNG echo user message cho button responses
        # Chỉ process và show kết quả
        
        if choice == "SPECIFIC":
            # Handle specific course search TRỰC TIẾP
            handle_specific_course_flow(original_query)
            manage_chat_history()
            return
            
        elif choice == "GENERAL": 
            # Handle general search TRỰC TIẾP
            handle_general_search(original_query)
            manage_chat_history()
            return

    # ========= FIX 2: CHỈ ECHO USER CHO REAL USER INPUT =========
    # Echo user CHỈ cho real user input, KHÔNG phải button responses
    with st.chat_message("user"):
        st.markdown(q)
    st.session_state.messages.append({'role': 'user', 'content': q})
    
    # ========= FIX 3: FORCE UI UPDATE NGAY LẬP TỨC =========
    # Đảm bảo user message hiển thị trước khi bắt đầu processing
    
    # Follow‑up details check (giữ nguyên)
    last = st.session_state.get('last_results', [])
    matched = None
    for course in last:
        name = course.get('name', '') or ''
        url = course.get('url', '') or ''
        if (name and name.lower() in q.lower()) or (url and url in q):
            matched = course; break

    if matched:
        with st.chat_message("assistant"):
            progress_container = st.empty()
            tracker = ProgressTracker(progress_container)
            
            simulate_progress_step(tracker, 0, 50, "Retrieving course details", "🔍", 0.8)
            details = st.session_state.advisor.get_course_details(matched['url'])
            simulate_progress_step(tracker, 50, 100, "Details loaded successfully", "✅", 0.5)
            
            time.sleep(0.3)
            tracker.clear()
            
            content = (
                f"**Details for {details['name']}**\n\n"
                f"- **Level:** {details.get('level','N/A')}\n"
                f"- **Skills:** {', '.join(details.get('skills',[]))}\n"
                f"- **Instructor:** {details.get('instructor','N/A')}\n"
                f"- **Description:** {details.get('description','')}"
            )
            st.markdown(content)
        st.session_state.messages.append({'role':'assistant','type':'text','content':content})
        st.session_state.last_results = []
        manage_chat_history()
        return

    # Main processing với intent check - NGAY DƯỚI USER MESSAGE
    with st.chat_message("assistant"):
        progress_container = st.empty()
        tracker = ProgressTracker(progress_container)
        response_container = st.empty()
        
        try:
            # Quick intent check
            simulate_progress_step(tracker, 0, 20, "Understanding your question", "🧠", 0.5)
            
            cls = FlexibleIntentClassifier()
            res = cls.classify_intent(q)
            intent = res["intent"]
            
            tracker.update(25, f"Intent: {intent.replace('_', ' ').title()}", "🎯")
            time.sleep(0.3)
            
            # Check if course search needs clarification
            if intent == "course_search":
                tracker.update(100, "Need clarification", "❓")
                time.sleep(0.5)
                tracker.clear()
                
                # Show clarification message
                response_container.markdown(f"""
I can help you with **"{q}"** in two ways:

Choose your preferred approach below:
""")
                
                # ✅ Set flags for buttons
                st.session_state.show_clarification_buttons = True
                st.session_state.clarification_query = q
                
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'type': 'clarification', 
                    'content': f'I can help you with "{q}" in two ways. Please choose your approach below.',
                    'query': q
                })
                
                # ✅ CRITICAL FIX: Force immediate rerun để show buttons
                st.rerun()
                return
            
            # Non-course search flows
            elif intent == "general_chat":
                simulate_progress_step(tracker, 25, 60, "Preparing response", "💭", 0.8)
                
                # Build context
                chat_history = st.session_state.messages[:-1]
                context = ""
                if chat_history:
                    recent = chat_history[-2:]
                    for msg in recent:
                        if msg.get("role") == "user" and "content" in msg:
                            context += f"User previously: {msg['content'][:50]}...\n"
                        elif msg.get("role") == "assistant" and msg.get("type") != "courses" and "content" in msg:
                            snippet = msg["content"][:50]
                            context += f"Assistant: {snippet}...\n"
                    if context:
                        context = f"Recent context:\n{context}\n"

                prompt = f"""{context}You are Course Finder, a friendly course finding system. 

                        Current user message: "{q}"

                        Respond naturally, warm, and helpful. Keep it concise but friendly (2-3 sentences max for general chat).
                        """
                
                tracker.update(95, "Generating response", "💬")
                time.sleep(0.3)
                tracker.clear()
                
                # Stream LLM response
                messages = [{"role":"user","content":prompt}]
                analysis = stream_llm_response(st.session_state.llm, messages, response_container)
                st.session_state.messages.append({
                    "role":"assistant","type":"text","content":analysis
                })
                
            elif intent == "scope_inquiry":
                simulate_progress_step(tracker, 25, 90, "Generating explanation", "📝", 0.6)
                
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
                tracker.update(100, "Explanation ready", "✅")
                time.sleep(0.5)
                tracker.clear()
                response_container.markdown(scope_explanation)
                st.session_state.messages.append({
                    "role":"assistant","type":"text","content":scope_explanation
                })
            
        except Exception as e:
            tracker.clear()
            response_container.markdown(f"❌ Error: {e}")
            logger.exception("Error in handle_query")
            return

    manage_chat_history()

# ---------------------------------------------
# Main Streamlit Interface
# ---------------------------------------------

# ===== MAIN INTERFACE =====
add_progress_css()
with st.sidebar:
    st.markdown("### 🛠️ Debug & Tools")
    
    # Clear chat history button
    if st.button("🔄 Clear Chat History"):
        st.session_state.clear_requested = True
        
        # ✅ Clear ALL session state related to chat và UI
        keys_to_clear = [
            'messages', 
            'last_results', 
            'message_results',
            # ✅ IMPORTANT: Clear clarification button states
            'show_clarification_buttons',
            'clarification_query',
            # Clear any other UI states
            'current_results_displayed',
            'duplicate_results',
            'temp_analysis',
            'pending_results',
            'pending_query',
            'button_clicked',
            'processing_specific',
            'processing_general'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # ✅ Also clear duplicate UI
        clear_duplicate_ui()
        
        st.rerun()

    
    st.markdown("---")  # Separator
    
    # Multi-entity debug section
    st.markdown("### 🔧 Entity Tests")
    
    entity_tests = {
        "Instructors": "Find instructors teaching machine learning",
        "Organizations": "What organizations offer programming courses", 
        "Providers": "Which providers have the most courses",
        "Reviews": "Show me reviews for Python courses",
        "Subjects": "What subjects are available in programming",
        "Skills": "What programming skills can I learn",
        "Levels": "What difficulty levels are available"
    }
    
    selected_test = st.selectbox("Test Entity:", list(entity_tests.keys()))
    
    if st.button("🧪 Run Entity Test"):
        # Instead of processing here, add the query to main interface
        test_query = entity_tests[selected_test]
        # Trigger the query in main interface
        st.session_state.pending_query = test_query
        st.rerun()
    
    st.markdown("---")  # Separator
    
    # Example queries section  
    st.markdown("### 🧪 Quick Examples")
    
    example_queries = [
        "find python programming courses",
        "instructors teaching machine learning", 
        "organizations offering data science"
    ]
    
    selected_query = st.selectbox("Try an example:", example_queries)
    
    if st.button("▶️ Test Query"):
        # Same approach - send to main interface
        st.session_state.pending_query = selected_query
        st.rerun()
st.title("🎓 CourseFinder")
st.markdown("Ask me about educational courses and I'll provide comprehensive information!")
display_chat()

if 'pending_query' in st.session_state and st.session_state.pending_query:
    pending = st.session_state.pending_query
    del st.session_state.pending_query  # Clear it
    handle_query(pending)
# ===== CLARIFICATION BUTTONS =====
if st.session_state.get('show_clarification_buttons', False):
    st.markdown("### 🤔 How would you like me to help?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📖 Find a Specific Course", 
            key="btn_specific", 
            use_container_width=True,
            help="Get detailed information about a particular course"
        ):
            st.session_state.show_clarification_buttons = False
            original_query = st.session_state.get('clarification_query', '')
            handle_specific_course_flow(original_query)
            st.rerun()
    
    with col2:
        if st.button(
            "🔍 Search & Analyze Multiple Courses", 
            key="btn_general", 
            use_container_width=True,
            help="Find multiple courses and get landscape overview"
        ):
            st.session_state.show_clarification_buttons = False
            original_query = st.session_state.get('clarification_query', '')
            handle_general_search(original_query)
            st.rerun()
    
    st.markdown("---")

# ===== CHAT INPUT - ĐẶT CUỐI CÙNG =====
if prompt := st.chat_input("Ask about courses...", key="unique_chat_input"):
    handle_query(prompt)

# ===== FOOTER =====
st.markdown("---")
st.caption("💡 Tip: Try asking about specific topics like 'Python courses' or 'Data science landscape'")