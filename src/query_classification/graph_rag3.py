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
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

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

class VectorStore:
    """Vector store for semantic search using FAISS"""
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        self.index = None
        self.course_metadata = []
        
    def build_index(self, courses: List[Dict]):
        """Build FAISS index from course data"""
        texts = []
        self.course_metadata = []
        
        for course in courses:
            # Combine name, description, and skills for rich context
            text_parts = []
            
            if course.get('name'):
                text_parts.append(f"Course: {course['name']}")
            
            if course.get('description'):
                text_parts.append(f"Description: {course['description']}")
                
            if course.get('skills') and isinstance(course['skills'], list):
                text_parts.append(f"Skills: {', '.join(course['skills'])}")
            
            combined_text = " | ".join(text_parts)
            texts.append(combined_text)
            self.course_metadata.append(course)
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Built vector index with {len(texts)} courses")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Semantic search in vector space"""
        if not self.index:
            return []
            
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid result
                results.append((self.course_metadata[idx], float(score)))
        
        return results

class HybridQueryProcessor:
    """Combines structured queries with semantic search"""
    def __init__(self, model_name: str = "qwen3:4b"):
        self.model_name = model_name
        self.json_pattern = re.compile(r'\{.*\}', re.DOTALL | re.MULTILINE)

    def analyze_query_intent(self, user_query: str) -> Dict:
        """Analyze whether query needs structured search, semantic search, or both"""
        prompt = f"""
        Analyze this educational query and determine the best search strategy:
        
        Query: "{user_query}"
        
        Classify into one of these strategies:
        1. STRUCTURED: Query has clear filters (level, organization, rating, specific skills in knowledge graph)
        2. SEMANTIC: Query is conceptual/descriptive (learning goals, broad topics, job roles)  
        3. HYBRID: Query needs both structured filtering AND semantic matching
        
        Also extract:
        - Key skills/topics mentioned
        - Specific filters (level, provider, rating)
        - Learning objectives or context
        
        Return JSON:
        {{
            "strategy": "STRUCTURED|SEMANTIC|HYBRID",
            "confidence": 0.0-1.0,
            "structured_elements": {{
                "skills": ["skill1", "skill2"],
                "level": "Beginner|Intermediate|Advanced|null",
                "organization": "provider_name|null", 
                "rating_filter": ">=4.5|null"
            }},
            "semantic_elements": {{
                "learning_goals": ["goal1", "goal2"],
                "job_context": "job_role|null",
                "broad_topics": ["topic1", "topic2"]
            }},
            "reasoning": "explanation of strategy choice"
        }}
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1},
                format='json'
            )
            
            return self.extract_json(response['response'])
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {str(e)}")
            return {
                "strategy": "SEMANTIC",
                "confidence": 0.5,
                "structured_elements": {},
                "semantic_elements": {"broad_topics": [user_query]},
                "reasoning": "Fallback to semantic search due to analysis error"
            }

    def generate_cypher_query(self, structured_elements: Dict) -> str:
        """Generate Cypher for structured search"""
        conditions = []
        match_clauses = ["(c:Course)"]
        
        if skills := structured_elements.get('skills'):
            for skill in skills:
                match_clauses.append(f"(c)-[:TEACHES]->(s:Skill {{name: '{skill}'}})")
        
        if level := structured_elements.get('level'):
            match_clauses.append(f"(c)-[:HAS_LEVEL]->(l:Level {{name: '{level}'}})")
        
        if org := structured_elements.get('organization'):
            match_clauses.append(f"(c)-[:OFFERED_BY]->(o:Organization {{name: '{org}'}})")
        
        if rating := structured_elements.get('rating_filter'):
            conditions.append(f"toFloat(c.rating) {rating}")
        
        match_clause = "MATCH " + ", ".join(match_clauses)
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        return_clause = " RETURN c.name, c.url, c.description, c.rating, c.duration"
        
        return match_clause + where_clause + return_clause

    def extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        text = text.replace("True", "true").replace("False", "false")
        match = self.json_pattern.search(text)
        if not match:
            raise ValueError("No JSON found in response")
        return json.loads(match.group())

class HybridKnowledgeBaseQA:
    """Main system combining structured and semantic search"""
    def __init__(self, neo4j_conn: Neo4jConnection):
        self.neo4j_conn = neo4j_conn
        self.query_processor = HybridQueryProcessor()
        self.vector_store = VectorStore()
        self.initialize_vector_store()
    
    def initialize_vector_store(self):
        """Load all courses and build vector index"""
        try:
            query = """
            MATCH (c:Course)
            OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
            OPTIONAL MATCH (c)-[:HAS_LEVEL]->(l:Level)
            OPTIONAL MATCH (c)-[:OFFERED_BY]->(o:Organization)
            RETURN 
                c.name as name,
                c.url as url,
                c.description as description,
                c.rating as rating,
                c.duration as duration,
                l.name as level,
                o.name as organization,
                collect(DISTINCT s.name) as skills
            """
            
            courses = self.neo4j_conn.execute_query(query)
            self.vector_store.build_index(courses)
            logger.info(f"Initialized vector store with {len(courses)} courses")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")

    def process_query(self, user_query: str) -> Dict:
        """Main query processing pipeline"""
        start_time = time.time()
        
        try:
            # 1. Analyze query intent
            intent = self.query_processor.analyze_query_intent(user_query)
            
            # 2. Execute appropriate search strategy
            if intent['strategy'] == 'STRUCTURED':
                results = self._structured_search(intent['structured_elements'])
                method = "structured_only"
                
            elif intent['strategy'] == 'SEMANTIC':
                results = self._semantic_search(user_query)
                method = "semantic_only"
                
            else:  # HYBRID
                structured_results = self._structured_search(intent['structured_elements'])
                semantic_results = self._semantic_search(user_query)
                results = self._merge_results(structured_results, semantic_results, user_query)
                method = "hybrid"
            
            # 3. Re-rank final results
            final_results = self._final_ranking(results, user_query, intent)
            
            processing_time = time.time() - start_time
            
            return {
                "results": final_results[:10],  # Top 10
                "metadata": {
                    "strategy": intent['strategy'],
                    "method": method,
                    "processing_time": f"{processing_time:.2f}s",
                    "total_found": len(final_results),
                    "intent_analysis": intent
                }
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "results": [],
                "metadata": {
                    "error": str(e),
                    "processing_time": f"{time.time() - start_time:.2f}s"
                }
            }

    def _structured_search(self, structured_elements: Dict) -> List[Dict]:
        """Execute structured Neo4j search"""
        if not any(structured_elements.values()):
            return []
            
        cypher = self.query_processor.generate_cypher_query(structured_elements)
        return self.neo4j_conn.execute_query(cypher)

    def _semantic_search(self, query: str) -> List[Dict]:
        """Execute semantic vector search"""
        vector_results = self.vector_store.search(query, top_k=20)
        return [{"semantic_score": score, **course} for course, score in vector_results]

    def _merge_results(self, structured: List[Dict], semantic: List[Dict], query: str) -> List[Dict]:
        """Intelligently merge structured and semantic results"""
        # Create URL-based lookup for structured results
        structured_lookup = {r['url']: r for r in structured}
        
        merged = []
        seen_urls = set()
        
        # First, add structured results with semantic scores
        for struct_result in structured:
            url = struct_result['url']
            seen_urls.add(url)
            
            # Find semantic score for this course
            semantic_score = 0.0
            for sem_result in semantic:
                if sem_result.get('url') == url:
                    semantic_score = sem_result.get('semantic_score', 0.0)
                    break
            
            merged.append({
                **struct_result,
                "semantic_score": semantic_score,
                "source": "structured+semantic"
            })
        
        # Then add high-scoring semantic results not in structured
        for sem_result in semantic:
            url = sem_result.get('url')
            if url not in seen_urls and sem_result.get('semantic_score', 0) > 0.3:  # Threshold
                merged.append({
                    **sem_result,
                    "source": "semantic_only"
                })
                seen_urls.add(url)
        
        return merged

    def _final_ranking(self, results: List[Dict], query: str, intent: Dict) -> List[Dict]:
        """Final ranking combining multiple signals"""
        if not results:
            return []
        
        for result in results:
            # Base semantic score
            semantic_score = result.get('semantic_score', 0.0)
            
            # Rating boost (if available)
            rating_boost = 0.0
            if result.get('rating'):
                try:
                    rating = float(result['rating'])
                    rating_boost = (rating - 3.0) / 10.0  # Scale 3-5 to 0-0.2
                except:
                    rating_boost = 0.0
            
            # Structured match boost
            structured_boost = 0.1 if result.get('source') == 'structured+semantic' else 0.0
            
            # Title relevance boost
            title_boost = 0.0
            if result.get('name'):
                # Simple keyword matching for title
                query_words = set(query.lower().split())
                title_words = set(result['name'].lower().split())
                overlap = len(query_words.intersection(title_words))
                title_boost = min(overlap * 0.05, 0.2)
            
            # Combined score
            final_score = semantic_score + rating_boost + structured_boost + title_boost
            result['final_score'] = final_score
        
        return sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)

# Updated Streamlit UI Components
def render_enhanced_course_card(course: Dict):
    """Enhanced course card showing multiple relevance signals"""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            name = course.get('name', 'Unnamed Course')
            url = course.get('url', '#')
            
            # Source indicator
            source = course.get('source', 'unknown')
            source_emoji = {
                'structured+semantic': '🎯',
                'semantic_only': '🔍',
                'structured_only': '📊'
            }.get(source, '❓')
            
            st.markdown(f"### {source_emoji} [{name}]({url})")
            
            # Description preview
            if desc := course.get('description'):
                with st.expander("Description"):
                    st.write(desc[:300] + "..." if len(desc) > 300 else desc)
            
            # Skills and metadata
            info_cols = st.columns(3)
            with info_cols[0]:
                if rating := course.get('rating'):
                    st.metric("Rating", f"{rating}/5.0")
            
            with info_cols[1]:
                if level := course.get('level'):
                    st.metric("Level", level)
            
            with info_cols[2]:
                if org := course.get('organization'):
                    st.metric("Provider", org)
            
            if skills := course.get('skills'):
                st.caption(f"**Skills:** {', '.join(skills[:5])}")
        
        with col2:
            # Relevance scoring
            final_score = course.get('final_score', 0)
            semantic_score = course.get('semantic_score', 0)
            
            st.metric("Relevance", f"{final_score:.3f}")
            st.caption(f"Semantic: {semantic_score:.3f}")
            
            # Source badge (đã loại bỏ tham số `type`)
            st.badge(source.replace('_', ' ').title())


def handle_enhanced_query_submission(query: str):
    """Enhanced query handling with hybrid search"""
    user_msg = {
        "role": "user",
        "type": "text", 
        "content": query,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_msg)
    
    try:
        with st.spinner("🔄 Processing with hybrid search..."):
            result = st.session_state.advisor.process_query(query)
            
            system_msg = {
                "role": "assistant",
                "type": "enhanced_courses",
                "content": result['results'],
                "metadata": result['metadata']
            }
            st.session_state.messages.append(system_msg)
            st.rerun()
            
    except Exception as e:
        logger.error(f"Enhanced query failed: {str(e)}")
        st.error(f"Search failed: {str(e)}")

def display_enhanced_chat_history():
    """Display chat history with enhanced course cards"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=("👤" if msg["role"] == "user" else "🤖")):
            if msg["type"] == "text":
                st.markdown(msg["content"])
                
            elif msg["type"] == "enhanced_courses":
                courses = msg['content']
                metadata = msg.get('metadata', {})
                
                # Results summary
                st.markdown(f"🎯 Found **{len(courses)}** courses using **{metadata.get('strategy', 'unknown')}** strategy")
                
                if metadata.get('total_found', 0) > len(courses):
                    st.caption(f"Showing top {len(courses)} of {metadata['total_found']} total results")
                
                # Display courses
                for course in courses:
                    render_enhanced_course_card(course)
                
                # Execution details
                with st.expander("🔍 Search Analysis"):
                    st.json(metadata)

def main():
    st.set_page_config(
        page_title="Enhanced EduAssistant with RAG",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Enhanced EduAssistant")
    st.caption("Hybrid Knowledge Graph + RAG Search System")
    
    # Initialize session state
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
    
    if 'advisor' not in st.session_state:
        with st.spinner("🔄 Initializing hybrid search system..."):
            st.session_state.advisor = HybridKnowledgeBaseQA(st.session_state.neo4j)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Search strategies info
    with st.expander("🔍 Search Strategies", expanded=False):
        st.markdown("""
        - **STRUCTURED**: Exact matching with filters (level, provider, skills in knowledge graph)
        - **SEMANTIC**: AI-powered understanding of learning goals and context
        - **HYBRID**: Combines both for comprehensive results
        
        The system automatically chooses the best strategy for your query!
        """)
    
    # Quick examples
    st.markdown("**🚀 Try these examples:**")
    cols = st.columns(3)
    examples = [
        ("I want to become a data scientist", "Career-focused semantic search"),
        ("Advanced Python from Google", "Structured search with filters"), 
        ("Machine learning for beginners", "Hybrid approach")
    ]
    
    for col, (query, desc) in zip(cols, examples):
        if col.button(f"{query}", help=desc, use_container_width=True):
            handle_enhanced_query_submission(query)
    
    # Chat interface
    display_enhanced_chat_history()
    
    if prompt := st.chat_input("Ask about courses (try natural language!)"):
        handle_enhanced_query_submission(prompt)

if __name__ == "__main__":
    main()