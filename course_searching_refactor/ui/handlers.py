import re
import time
import logging
import streamlit as st
from typing import List, Dict
from models.llm import OllamaLLM
from ui.components import (
    ProgressTracker, simulate_progress_step, clean_answer, 
    manage_chat_history, stream_llm_response
)

logger = logging.getLogger(__name__)

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
Avoid reasoning too much.
Course name:"""

        # Use small and fast LLM
        llm = OllamaLLM(model="qwen3:4b", small=True)
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
        
        from database.neo4j_client import get_neo4j_connection
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
    """Handle specific course search flow"""
    
    # Clear buttons before starting
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
            
            # Enhanced course name extraction
            course_name = extract_course_name(original_query)
            
            # Double check: Clean course name again
            course_name = clean_answer(course_name)
            
            # Validation: Make sure course name is reasonable
            if len(course_name) < 3 or '<think>' in course_name.lower():
                logger.warning(f"Invalid course name extracted: {course_name}")
                # Use fallback extraction
                course_name = extract_course_name_fallback(original_query)
            
            tracker.update(30, f"Looking for: '{course_name}'", "🎯")
            time.sleep(0.5)
            
            # Step 2: Search database
            simulate_progress_step(tracker, 30, 60, "Searching database", "🗄️", 1.2)
            results = run_specific_course_search(course_name)
            tracker.update(65, f"Found {len(results)} matching courses", "📊")
            time.sleep(0.3)
            
            # Step 3: Generate detailed analysis
            simulate_progress_step(tracker, 65, 95, "Analyzing course details", "🤖", 1.0)
            
            if results:
                # Get the best match (first result)
                best_match = results[0]
                
                # Clean extracted info display
                extracted_info = f"**Searching for:** {course_name}\n**Found:** {best_match.get('name', 'Unknown')}\n\n"
                
                # Create detailed analysis prompt - simplified to avoid thinking
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
                
                # Enhanced streaming with better cleaning
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
                
                # Clean no results message
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
            error_msg = f"Error processing your request: {str(e)}"
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
    
    # Remove thinking tags (most aggressive)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining thinking patterns
    text = re.sub(r'\*\*thinking\*\*.*?\*\*end thinking\*\*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove debug/reasoning lines
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
    
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up whitespace
    text = text.strip()
    
    # If text starts with reasoning, extract the actual answer
    if any(text.lower().startswith(prefix) for prefix in ['okay', 'let me', 'first', 'the user']):
        # Try to find the actual answer after reasoning
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not any(word in sentence.lower() for word in ['let me', 'i need', 'user wants']):
                text = sentence + '.'
                break
    
    return text
def handle_query(q: str):
    """Process user question with clarification buttons"""
    
    from ui.components import clear_duplicate_ui
    # Clear any duplicate UI elements
    clear_duplicate_ui()
    
    # Check if this is a button response
    if q.startswith("CLARIFY_"):
        choice = q.replace("CLARIFY_", "").split("_")[0]
        original_query = q.replace(f"CLARIFY_{choice}_", "")
        
        # Clear clarification buttons immediately
        st.session_state.show_clarification_buttons = False
        if 'clarification_query' in st.session_state:
            del st.session_state.clarification_query
        
        # Don't echo user message for button responses
        if choice == "SPECIFIC":
            handle_specific_course_flow(original_query)
            manage_chat_history()
            return
            
        elif choice == "GENERAL": 
            handle_general_search(original_query)
            manage_chat_history()
            return

    # Echo user only for real user input, not button responses
    with st.chat_message("user"):
        st.markdown(q)
    st.session_state.messages.append({'role': 'user', 'content': q})
    
    # Follow-up details check
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

    # Main processing with intent check
    with st.chat_message("assistant"):
        progress_container = st.empty()
        tracker = ProgressTracker(progress_container)
        response_container = st.empty()
        
        try:
            # Quick intent check
            simulate_progress_step(tracker, 0, 20, "Understanding your question", "🧠", 0.5)
            
            from core.intent_classifier import FlexibleIntentClassifier
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
                
                # Set flags for buttons
                st.session_state.show_clarification_buttons = True
                st.session_state.clarification_query = q
                
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'type': 'clarification', 
                    'content': f'I can help you with "{q}" in two ways. Please choose your approach below.',
                    'query': q
                })
                
                # Force immediate rerun to show buttons
                st.rerun()
                return
            
            # General chat flow (includes scope inquiry)
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

                # Enhanced prompt with system information
                prompt = f"""{context}You are Course Finder, a friendly course finding system. Here's your system information:

=== SYSTEM INFORMATION ===
You are backed by a Neo4j knowledge graph containing:
• 169,046 total nodes
• 282,507 total relationships

Node breakdown (8 types):
- Course: 13,793
- Instructor: 6,027
- Level: 4 (Beginner, Intermediate, Advance, Mix)
- Organization: 526
- Provider: 2
- Review: 136,767
- Skill: 11,897
- Subject: 30

Relationship breakdown (7 types):
- HAS_LEVEL: 13,982
- HAS_REVIEW: 136,767
- HAS_SUBJECT: 1,537
- OFFERED_BY: 14,118
- PROVIDED_BY: 13,793
- TAUGHT_BY: 15,749
- TEACHES: 86,561

=== YOUR ROLE & CAPABILITIES ===
You are a smart course discovery assistant designed to help users find and understand learning opportunities from your database of 13,793 courses.

**What you can help with:**
- Course discovery and search
- Educational insights and course landscapes
- Understanding course characteristics, levels, and learning paths
- Providing information about instructors, organizations, and course reviews
- General conversation about learning and education topics

**Your approach:**
- Provide comprehensive overviews without making specific recommendations
- Empower users with insights so they can choose courses themselves
- Be warm, helpful, and conversational
- When asked about your capabilities or database, share the system metrics above

Current user message: "{q}"

RESPONSE GUIDELINES:
- If they greet you, greet them back warmly
- If they ask how you are, respond appropriately 
- If they ask about your capabilities, system, or database, explain using the system information above
- If they ask about conversation history, acknowledge you can see recent messages
- If they ask about something outside your scope, politely decline and redirect to course topics
- Keep responses conversational, warm, and helpful
- Don't overthink - be direct and friendly
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
                
        except Exception as e:
            tracker.clear()
            response_container.markdown(f"❌ Error: {e}")
            logger.exception("Error in handle_query")
            return

    manage_chat_history()
def handle_general_search(original_query: str):
    """Handle general multi-course search"""
    
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
            simulate_progress_step(tracker, 0, 30, "Analyzing query for database", "🔍", 1.0)
            
            # Step 2: Search database
            simulate_progress_step(tracker, 30, 70, "Searching database", "🗄️", 1.5)
            
            # Use existing process_query logic
            results = st.session_state.advisor.process_query(original_query)
            
            simulate_progress_step(tracker, 70, 100, "Analyzing results", "📊", 1.0)
            
            if results:
                # Generate analysis
                from utils.helpers import EnhancedResultHandler
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