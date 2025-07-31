import logging
import sys
import ollama
import numpy as np
import json
import re
import streamlit as st
from neo4j import GraphDatabase
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

class LSTMEmbeddingModel:
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, max_length: int = 200):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Build model
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=False),
            Dense(embedding_dim, activation='tanh')
        ])
        self.model.compile(loss='cosine_proximity', optimizer='adam')

    def train(self, texts: List[str]):
        """Train the embedding model on given texts"""
        if not texts:
            logger.warning("No texts provided for training")
            return
            
        try:
            # Tokenization and sequencing
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length)
            
            # Dummy training (since we're using pre-trained embeddings)
            # For real training, you'd need labeled data
            dummy_labels = np.random.rand(len(texts), self.embedding_dim)
            self.model.fit(padded, dummy_labels, epochs=10, verbose=0)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_length)
            return self.model.predict(padded, verbose=0)[0]
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return np.zeros(self.embedding_dim)

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
class ConversationManager:
    def __init__(self, window_size=3):
        self.history = []
        self.window_size = window_size
    
    def add_interaction(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]
    
    def get_context(self) -> str:
        return "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in self.history
        )
class QueryProcessor:
    def __init__(self, model_name: str = "qwen3:4b"): #qwen2.5-coder:3b    llama3.2:3b        qwen2.5:3b         deepseek-r1:7b   qwen3:4b
        self.model_name = model_name
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL | re.MULTILINE)

    def generate_query_plan(self, user_query: str) -> Dict:
        start_time = time.perf_counter()
        prompt = f"""
        You are a Neo4j Cypher expert. Generate a valid Cypher query following these STRICT RULES:
        
        MANDATORY RULES:
        1. Query MUST start with MATCH
        2. Use ONLY these node labels: Course, Skill, Level, Organization, Instructor, Career
        3. Use ONLY these relationships: TEACHES, HAS_LEVEL, OFFERED_BY, TAUGHT_BY, REQUIRES
        4. ALWAYS use single quotes for string values: 'value'
        5. Combine conditions using WHERE clause when needed
        6. MUST include RETURN clause with course properties
        7. The value of final_query MUST begin with the keyword MATCH (case insensitive)
        8. NEVER use double quotes "" inside the Cypher
        9. Special rule for levels:
            - Detect any of these keywords in the query (case-insensitive):
                Beginner, Introductory, Basics, Fundamentals...
                Intermediate, Mid-level... 
                Advanced, Expert, Hard... 
            - Map the matched keyword to Level.name exactly (“Beginner”, “Intermediate” or “Advanced”).
            - Remove that keyword (and any prepositions like “of”) from the skill phrase.
            - **Normalize the skill name** by converting it to Title Case (capitalize the first letter of each word).
            - Then generate one step to MATCH the Skill{{name: $skill_name}} and one step to MATCH courses with HAS_LEVEL → Level{{name: $level}}.
        10. For ANY of these Course→X relationships (TEACHES, HAS_LEVEL, OFFERED_BY, TAUGHT_BY), always match directly from 'Course':
            MATCH (c:Course)-[:REL_TYPE]->(x:NodeType {{…}}),
            do not write (c)-[:TEACHES]->(s)-[:HAS_LEVEL]->(l).

        11. NEVER put relationship patterns inside WHERE. 
            Always include Course→X relationships in a MATCH clause 
            or comma-separated in the first MATCH.
        12. Rating is in string format, if any query about rating, change to toFloat(c.rating) instead of c.rating for the whole Cypher.
        SCHEMA:
        - Course properties: url, name, duration, rating, description
        - Relationships directions: 
          (Course)-[:TEACHES]->(Skill)
          (Course)-[:HAS_LEVEL]->(Level)
          (Course)-[:OFFERED_BY]->(Organization)
          (Course)-[:TAUGHT_BY]->(Instructor)
          (Career)-[:REQUIRES]->(Skill)

        EXAMPLES:[
        {{
                "natural_query": "Find courses about machine learning",
                "steps": ["Match courses teaching Machine Learning skill", "Return results"],
                "final_query": "MATCH (c:Course)-[:TEACHES]->(s:Skill {{name: 'Machine Learning'}}) RETURN c.name, c.url, c.rating"
            }},

        {{
                "natural_query": "Advanced Cybersecurity courses from Google",
                "steps": [
                    "Find Cybersecurity skill courses",
                    "Filter by Advanced level",
                    "Check offered by Google",
                    "Return results"
                ],
                "final_query": "MATCH (c:Course)-[:TEACHES]->(s:Skill {{name: 'Cybersecurity'}}), (c)-[:HAS_LEVEL]->(l:Level {{name: 'Advanced'}}), (c)-[:OFFERED_BY]->(o:Organization {{name: 'Google'}}) RETURN c.name, c.url, c.duration"
            }},
        {{  
                "natural_query": Find me some Python beginner courses
                "steps": ["Find Python skill courses", "Filter by Beginner level"],
                "final_query": "MATCH (c:Course)-[:TEACHES]->(s:Skill {{name: 'Python'}}), (c)-[:HAS_LEVEL]->(l:Level {{name: 'Beginner'}}) RETURN c.name, c.url, c.duration"
            }},
        {{
                "natural_query": "Data science courses with rating > 4.5",
                "steps": [
                    "Match courses teaching Data Science skill",
                    "Filter by courses with rating greater than 4.5",
                    "Return results"
                ],
                "final_query": "MATCH (c:Course)-[:TEACHES]->(s:Skill {{name: 'Data Science'}}) WHERE toFloat(c.rating) > 4.5 RETURN c.name, c.url, c.rating"
        }}
        ]
        USER QUERY: {user_query}

        OUTPUT FORMAT (JSON):
        {{
            "steps": ["step1", "step2", ...],
            "final_query": "MATCH...RETURN..."
        }}
        """
        
        try:
            logger.debug(f"Generating query plan with prompt:\n{prompt[:500]}...")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1},
                format='json'
            )
            gen_time = time.perf_counter() - start_time
            logger.info(f"Query generated in {gen_time:.2f}s")
            
            
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
            logger.error(f"Query generation failed after {gen_time:.2f}s: {str(e)}")
            return {
                "steps": ["Error generating query plan"],
                "final_query": "",
                "debug_info": {
                    'error': str(e),
                    'gen_time': gen_time
                }
            }
    def validate_cypher(self, query: str):
        """Enhanced Cypher validation"""
        # Check for empty query
        if not query:
            raise ValueError("Empty query")
            
        q = re.sub(r"^```cypher|```$", "", query.strip(), flags=re.IGNORECASE).strip()
        if not re.match(r'(?i)^MATCH\s', q):
            raise ValueError("Query must start with MATCH clause")
            
        # Check contains RETURN clause
        if "RETURN" not in query.upper():
            raise ValueError("Query missing RETURN clause")
            
        # Check for forbidden keywords
        forbidden = ["CREATE", "DELETE", "SET", "REMOVE", "MERGE"]
        for word in forbidden:
            if word in query.upper():
                raise ValueError(f"Forbidden keyword detected: {word}")


    def extract_json(self, text: str) -> Dict:
        # Preprocessing steps
        text = text.replace("True", "true").replace("False", "false")
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Find JSON candidate block
        match = self.json_pattern.search(text)
        if not match:
            raise ValueError("No JSON found in response")
        json_str = match.group()
        
        # Special handling for final_query string values
        def replace_inner_quotes(m):
            inner = m.group(1)
            # Replace all unescaped double quotes with single quotes
            inner = re.sub(r'(?<!\\)"', "'", inner)
            # Escape any remaining backslashes
            inner = inner.replace('\\', '\\\\')
            return f'"final_query": "{inner}"'
        
        # Process final_query field
        json_str = re.sub(
            r'"final_query"\s*:\s*"((?:\\"|[^"])*)"',
            replace_inner_quotes,
            json_str,
            flags=re.DOTALL
        )
        
        # Convert smart quotes to regular quotes
        json_str = json_str.replace('“', '"').replace('”', '"')
        
        # Handle missing commas in JSON arrays
        json_str = re.sub(r'\s*]\s*{', '], {', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e.msg}")
            logger.error(f"Problematic JSON content: {json_str}")
            raise


    
    def clean_cypher(self, query: str) -> str:
        # 1) Strip markdown fences & semicolons
        q = query.strip()
        q = re.sub(r"^```cypher|```$", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r";+\s*$", "", q)

        # 2) Force single quotes around properties
        #    e.g. replace name: "Python" → name: 'Python'
        q = re.sub(r'name:\s*"([^"]+)"', r"name: '\1'", q)

        # 3) Correct any chained Skill→Level patterns to separate MATCHes
        #    If it sees (s:Skill)…-[:HAS_LEVEL]->(l:Level), fix it:
        q = re.sub(
            r'\(c:Course\)-\[:TEACHES\]->\(s:Skill\s*\{[^\}]+\}\)-\[:HAS_LEVEL\]->\(l:Level\s*\{[^\}]+\}\)',
            "(c:Course)-[:TEACHES]->(s:Skill {name: 'Python'}), (c)-[:HAS_LEVEL]->(l:Level {name: 'Beginner'})",
            q
        )

        # 4) Normalize whitespace and return
        q = re.sub(r'\s+', ' ', q).strip()
        return q
    # def validate_cypher(self, query: str):
    #     """Enhanced Cypher validation and cleaning"""
    #     # Clean the query first
    #     query = query.strip().replace('"', "'").replace('\\n', ' ').replace('\\t', ' ')
        
    #     # Check for empty query
    #     if not query:
    #         raise ValueError("Empty query")
            
    #     # Check starting with MATCH (case insensitive)
    #     if not re.match(r'^\s*MATCH\s', query, re.IGNORECASE):
    #         raise ValueError("Query must start with MATCH clause")
            
    #     # Check for RETURN clause
    #     if "RETURN" not in query.upper():
    #         raise ValueError("Query missing RETURN clause")

class KnowledgeBaseQA:
    def __init__(self, neo4j_conn: Neo4jConnection, embedding_model: LSTMEmbeddingModel):
        self.neo4j_conn = neo4j_conn
        self.embedding_model = embedding_model
        self.query_processor = QueryProcessor()
        self.conversation = ConversationManager()
    def generate_natural_response(self, query: str, results: List[Dict]) -> str:
        prompt = f"""
        You are a course advisor. Generate a friendly response based on these guidelines:
        
        1. Acknowledge the user's query
        2. Mention number of results
        3. Highlight key filters (if any)
        4. Offer next steps
        
        Query: {query}
        Results Found: {len(results)}
        Example Response: "I found {len(results)} courses matching your request for X. 
        These include options from Y providers at Z level. Would you like me to compare 
        any of these or provide more details on specific courses?"
        """
        
        response = ollama.generate(
            model="qwen3:4b",
            prompt=prompt,
            options={'temperature': 0.5}
        )
        return response['response'].strip()
    
    def process_query(self, user_query: str) -> Dict:
        response = {"type": "text", "content": "", "results": []}
        
        try:
            # Check if question is conversational
            if self.is_general_question(user_query):
                response["content"] = self.answer_general_question(user_query)
                return response
                
            # Otherwise process as course search
            plan = self.query_processor.generate_query_plan(user_query)
            if not plan.get('final_query'):
                response["content"] = "Could not generate valid search query"
                return response
                
            raw_results = self.neo4j_conn.execute_query(plan['final_query'])
            filtered = self.semantic_filter(raw_results, user_query)
            
            response.update({
                "type": "courses",
                "content": self.generate_natural_response(user_query, filtered),
                "results": filtered,
                "metadata": {
                    "query_plan": plan,
                    "processed_at": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            response["content"] = f"Sorry, I encountered an error: {str(e)}"
        
        self.conversation.add_interaction("user", user_query)
        self.conversation.add_interaction("system", response["content"])
        return response

    def is_general_question(self, query: str) -> bool:
        prompt = f"""
        Classify if this query requires course search or is a general question:
        Query: {query}
        Respond ONLY with 'course' or 'general'
        """
        response = ollama.generate(model="llama3", prompt=prompt)
        return "general" in response['response'].lower()

    def answer_general_question(self, query: str) -> str:
        prompt = f"""
        You're a course advisor assistant. Answer this question using 
        simple, friendly language. If unsure, ask for clarification.
        
        Context from previous conversation:
        {self.conversation.get_context()}
        
        Question: {query}
        """
        response = ollama.generate(
            model="qwen3:4b", 
            prompt=prompt,
            options={'temperature': 0.5}
        )
        return response['response'].strip()
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

    def process_query(self, user_query: str) -> List[Dict]:
        try:
            plan = self.query_processor.generate_query_plan(user_query)
            if not plan.get('final_query'):
                return []
                
            raw_results = self.neo4j_conn.execute_query(plan['final_query'])
            return self.semantic_filter(raw_results, user_query)
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return []

    def semantic_filter(self, results: List[Dict], query: str) -> List[Dict]:
        """Lọc và xếp hạng kết quả dựa trên ngữ nghĩa"""
        if not results:
            return []

        filtered = []
        query_embedding = self.embedding_model.get_embedding(query)
        
        for record in results:
            try:
                # Tính similarity cho description
                desc = record.get('description', '')
                desc_embedding = self.embedding_model.get_embedding(desc)
                desc_sim = np.dot(query_embedding, desc_embedding)
                
                # Tính similarity cho tên khóa học
                name = record.get('name', '')
                name_sim = self.embedding_model.get_semantic_similarity(name, query)
                
                # Kết hợp trọng số
                total_sim = 0.6 * desc_sim + 0.4 * name_sim
                
                filtered.append({
                    **record,
                    "similarity": float(total_sim),
                    "name_similarity": float(name_sim),
                    "desc_similarity": float(desc_sim)
                })
                
            except Exception as e:
                logger.warning(f"Skipping invalid record: {str(e)}")
                continue
        
        return sorted(filtered, key=lambda x: x['similarity'], reverse=True)[:5]


    def rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        if not results:
            return []

        query_embedding = self.embedding_model.get_embedding(query)
        ranked = []
        
        for record in results:
            try:
                desc = record.get('description', '')
                course_embedding = self.embedding_model.get_embedding(desc)
                similarity = np.dot(query_embedding, course_embedding)
                
                ranked.append({
                    **record,
                    "similarity": float(similarity)
                })
                
            except Exception as e:
                logger.warning(f"Skipping invalid record: {str(e)}")
                continue
        
        return sorted(ranked, key=lambda x: x['similarity'], reverse=True)[:5]
    def get_course_level(self, course_url: str) -> str:
        """Fetch course level từ Neo4j nếu không có trong kết quả"""
        query = """
        MATCH (c:Course {url: $url})-[:HAS_LEVEL]->(l:Level)
        RETURN l.name AS level
        """
        result = self.neo4j_conn.execute_query(query, {"url": course_url})
        return result[0].get('level', 'N/A') if result else 'N/A'

# Streamlit UI Components
def render_course_card(course: Dict):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = course.get('name', course.get('c.name', 'Unnamed Course'))
            url = course.get('url', course.get('c.url', '#'))
            rating = course.get('rating', course.get('c.rating', 0))
            details = st.session_state.advisor.get_course_details(url)
            level = details.get('level', 'N/A') if details else 'N/A'
            # Thêm popover cho chi tiết khóa học
            with st.popover(f"📚 {name}", use_container_width=True):
                if details := st.session_state.advisor.get_course_details(url):
                    st.markdown(f"**Description:** {details.get('description', 'N/A')}")
                    st.divider()
                    
                    cols = st.columns(3)

                    try:
                        rating = float(rating)
                        rating_display = f"{rating:.1f}/5.0"
                    except ValueError:
                        rating_display = "No rating"

                    cols[0].metric("Rating", rating_display)

                    cols[1].metric("Level", details.get('level', 'Unknown'))
                    cols[2].metric("Skills", len(details.get('skills', [])))
                    
                    if details.get('skills'):
                        st.markdown("**Related Skills:**")
                        st.write(", ".join(details['skills']))
            
            st.markdown(f"### [{name}]({url})")
            st.caption(f"**Level:** {level} | **Rating:** {course.get('rating', "No rating")}")
            skills = course.get('skills', [])
            if skills:
                st.write("**Skills:** " + ", ".join(skills))
        
        with col2:
            similarity = course.get('similarity', 0)
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
            ("AWS Advanced", "Show advanced cloud computing courses with AWS"),
            ("Top Data Science", "Data science courses with rating > 4.5")
        ]
        
        for col, (title, query) in zip(cols, sample_queries):
            if col.button(title, help=query, use_container_width=True):
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
            response = st.session_state.advisor.process_query(query)
            
            system_msg = {
                "role": "assistant",
                "type": response["type"],
                "content": response["content"],
                "results": response.get("results", []),
                "metadata": response.get("metadata", {})
            }
            
            st.session_state.messages.append(system_msg)
            st.rerun()
            
    except Exception as e:
        handle_processing_error(e)

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
            st.markdown(msg["content"])
            
            if msg["type"] == "courses":
                st.markdown(f"**Found {len(msg['results'])} relevant courses:**")
                for course in msg['results']:
                    render_course_card(course)
                
            if "metadata" in msg:
                with st.expander("🔍 Execution Details"):
                    st.json(msg["metadata"])

def main():
    st.set_page_config(
        page_title="CourseFinder",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Thêm custom CSS
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

    # Khởi tạo session state
    if 'neo4j' not in st.session_state:
        try:
            st.session_state.neo4j = Neo4jConnection(
                "bolt://localhost:7687",
                "neo4j",
                "12345678"
            )
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")
            st.stop()
    
    if 'model' not in st.session_state:
        st.session_state.model = LSTMEmbeddingModel()
    
    if 'advisor' not in st.session_state:
        st.session_state.advisor = KnowledgeBaseQA(
            st.session_state.neo4j,
            st.session_state.model
        )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Render UI components
    render_sidebar_settings()
    show_quick_actions()
    display_chat_history()
    
    # Xử lý input
    if prompt := st.chat_input("Ask about courses..."):
        handle_query_submission(prompt)
    
    # Auto-scroll
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
    main()