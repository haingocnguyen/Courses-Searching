import logging, sys

# Xoá toàn bộ handler hiện có
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# Tạo handler mới ghi ra stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s"))

# Gắn vào root logger, và bật mức DEBUG
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(handler)
logger = logging.getLogger(__name__)
import streamlit as st
from neo4j import GraphDatabase
import ollama
import numpy as np
from functools import lru_cache
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from neo4j import GraphDatabase, WRITE_ACCESS
from neo4j.graph import Node as GraphNode
import time
from datetime import datetime
from neo4j import GraphDatabase
from functools import lru_cache
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import ClassVar
from typing import Any
from langchain.chains.base import Chain
from pydantic import Field
from pydantic import PrivateAttr

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_query(self, query, parameters=None):
        try:
            with self._driver.session(
                default_access_mode=WRITE_ACCESS
            ) as session:
                result = session.run(query, parameters or {})
                return [dict(r.items()) for r in result]
        except Exception as e:
            logging.error(f"Neo4j query failed: {e}")
            logging.debug(f"Failed query: {query}")
            logging.debug(f"Parameters: {parameters}")
            return []

    def close(self):
        self.driver.close()

    # def run_query(self, query, parameters=None):
    #     with self.driver.session() as session:
    #         result = session.run(query, parameters or {})
    #         records = [r.data() for r in result]


    #     cleaned = []
    #     for record in records:
    #         clean = {}
    #         for k, v in record.items():
    #             new_key = k.strip().strip('"').strip()
    #             clean[new_key] = v
    #         cleaned.append(clean)

    #     return cleaned
    # def run_query(self, query, parameters=None):
    #     try:
    #         with self.driver.session() as session:
    #             return session.execute_write(
    #                 lambda tx: list(tx.run(query, parameters or {})))
    #     except Exception as e:
    #         logging.error(f"Query failed: {str(e)}")
    #         return []

    def store_embedding(self, node_id, embedding):
        query = """
        MATCH (c:Course {url: $node_id})
        SET c.embedding = $embedding
        """
        self.run_query(query, {"node_id": node_id, "embedding": embedding.tolist()})

# LSTM Embedding Model
class LSTMEmbeddingModel:
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.max_length = max_length
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=False),
            Dense(embedding_dim, activation='tanh')
        ])

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        self.model.compile(optimizer='adam', loss='mse')
        dummy_labels = np.zeros((len(texts), 128))
        self.model.fit(padded, dummy_labels, epochs=1, verbose=0)

    def get_embedding(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        return self.model.predict(padded, verbose=0)[0]
class SubqueryOutputParser(BaseOutputParser):
    def parse(self, text: str):
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if json_match:
                text = json_match.group(1)
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
# Query Processor with Qwen/deepseek
class QueryDecomposer(Chain):
    prompt: ClassVar[PromptTemplate] = PromptTemplate(
        input_variables=["question"],
        template="""
        You are a Neo4j Cypher expert. Decompose this query into sequential subqueries:

        Graph Schema:
        - Nodes:
          Course {{url, name, duration, rating, description}}
          Skill {{name}}
          Level {{name}}
          Organization {{name}}
          Instructor {{name}}
          Career {{name}}
        - Relationships:
          TEACHES (Course -> Skill)
          HAS_LEVEL (Course -> Level)
          OFFERED_BY (Course -> Organization)
          TAUGHT_BY (Course -> Instructor)
          REQUIRES (Career -> Skill)

        Rules:
        1. Break complex queries into atomic steps
        2. Each step must return either primitive values or node IDs
        3. Subsequent steps must reference prior results via $bind variables
        4. Use parameterized queries with $placeholders
        5. Handle skill-level combinations using Level nodes

        Output JSON format:
        [
          {{
            "description": "step purpose",
            "cypher": "MATCH...RETURN...",
            "parameters": {{"$param": value}},
            "return": "intermediate|final",
            "bind": "variable_name" (if intermediate)
          }}
        ]

        Query: {question}
        """
    )
    output_parser: ClassVar[BaseOutputParser] = SubqueryOutputParser()

    _llm: Any = PrivateAttr()

    def __init__(self, llm: Any):
        super().__init__()
        # Gán vào private attr, không chạm vào ClassVar
        self._llm = llm
        #self.output_parser = SubqueryOutputParser()

    @property
    def input_keys(self):
        return ["question"]

    @property
    def output_keys(self):
        return ["subqueries"]

    def _call(self, inputs):
        # Dùng self._llm, không self.llm
        prompt_text = self.prompt.format(question=inputs["question"])
        # Generate từ private llm
        response = self._llm.generate([prompt_text])
        raw_output = response.generations[0][0].text
        # Parse JSON
        queries = self.output_parser.parse(raw_output)
        return {"subqueries": queries}

    
# Knowledge Base QA System
class KnowledgeBaseQA:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, llm_model, embedding_model):
        self.conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.llm = Ollama(model=llm_model, temperature=0.3)
        self.decomposer = QueryDecomposer(llm=self.llm)
        self.embedding_model = embedding_model

    def process_query(self, user_query):
        try:
            # Step 1: Query decomposition
            decomposition = self.decomposer({"question": user_query})
            steps = decomposition["subqueries"]
            
            if not steps:
                return []

            # Step 2: Execute subqueries
            context = {}
            results = []
            
            for step in steps:
                processed_params = {
                    k: context.get(v.lstrip('$')) if isinstance(v, str) and v.startswith('$') else v
                    for k, v in step.get("parameters", {}).items()
                }
                
                records = self.conn.run_query(step["cypher"], processed_params)
                
                if step.get("return") == "intermediate":
                    context[step["bind"]] = [list(r.values())[0] for r in records]
                else:
                    results.extend([
                        self._format_result(r) 
                        for r in records
                        if r not in results
                    ])

            # Step 3: Semantic ranking
            query_embed = self.embedding_model.get_embedding(user_query)
            results.sort(
                key=lambda x: self._cosine_sim(
                    query_embed,
                    self.embedding_model.get_embedding(x["description"])
                ),
                reverse=True
            )
            
            return results[:10]

        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return []

    def _format_result(self, record):
        return {
            "title": record.get("name", "Untitled Course"),
            "url": record.get("url", "#"),
            "rating": record.get("rating", 0.0),
            "skills": record.get("skills", []),
            "level": record.get("level", "Unknown"),
            "description": record.get("description", "")
        }

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)



# ====================== Streamlit UI Integration ======================
def render_courses(courses):
    """Display course cards with rich formatting"""
    st.json(courses)
    return

def show_course_details(url):
    """Display course details popover"""
    query = """
    MATCH (c:Course {url: $url})
    OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
    OPTIONAL MATCH (c)-[:HAS_LEVEL]->(l:Level)
    RETURN c.name AS name, 
           c.description AS description,
           c.rating AS rating,
           collect(DISTINCT s.name) AS skills,
           l.name AS level
    """
    result = st.session_state.advisor.neo4j_conn.run_query(query, {"url": url})
    
    if result:
        details = result[0]
        with st.popover(f"📚 {details['name']}", use_container_width=True):
            st.markdown(f"**Description:** {details.get('description', 'N/A')}")
            st.divider()
            
            cols = st.columns(3)
            cols[0].metric("Rating", f"{details.get('rating', 0):.1f}/5.0")
            cols[1].metric("Level", details.get('level', 'Unknown'))
            cols[2].metric("Skills", len(details.get('skills', [])))
            
            if details.get('skills'):
                st.markdown("**Related Skills:**")
                st.write(", ".join(details['skills']))

def handle_query_submission(query: str):
    """Handle query submission and UI updates"""
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
            
            # Execute query
            results = st.session_state.advisor.process_query(query)
            
            # Format system response
            system_msg = {
                "role": "assistant",
                "type": "courses" if results else "text",
                "content": results if results else "No matching courses found",
                "metadata": {
                    "processing_time": f"{time.time()-start_time:.2f}s",
                    "query_type": "course_search"
                }
            }
            st.session_state.messages.append(system_msg)
            
            # Auto-refresh UI
            st.rerun()
            
    except Exception as e:
        handle_processing_error(e)

def handle_processing_error(error: Exception):
    """Display error messages in UI"""
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
def render_sidebar_settings():
    """User settings panel"""
    with st.sidebar.expander("⚙️ SETTINGS", expanded=True):
        current_theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "System"],
            index=2,
            help="Select display theme"
        )
        
        
        st.caption(f"Version: 1.0.0 | Mode: {current_theme}")

def display_chat_history():
    """Display chat message history"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=("👤" if msg["role"] == "user" else "🤖")):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "courses":
                render_courses(msg["content"])
            
            if "metadata" in msg:
                with st.expander("Technical details"):
                    st.json(msg["metadata"])

def show_quick_actions():
    """Sample query quick action buttons"""
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

def process_chat_input():
    """Handle main chat input"""
    if prompt := st.chat_input("Ask about courses..."):
        handle_query_submission(prompt)

def main():
    st.set_page_config(
        page_title="EduAssistant", 
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
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

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "advisor" not in st.session_state:
        st.session_state.advisor = KnowledgeBaseQA(
        neo4j_uri="bolt://localhost:7687?address_resolver=ipv4",
        neo4j_user="neo4j",
        neo4j_password="12345678",
        llm_model="deepseek-r1:7b",          
        embedding_model=LSTMEmbeddingModel()
    )

    # Render UI components
    render_sidebar_settings()
    show_quick_actions()
    display_chat_history()
    process_chat_input()
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
    main()
                