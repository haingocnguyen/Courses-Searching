import atexit
import logging
import streamlit as st
from config import MAX_CHAT_MESSAGES
from core.knowledge_base import get_knowledge_base_qa
from models.llm import get_llm
from ui.components import add_progress_css, display_chat, clear_duplicate_ui
from ui.handlers import handle_query, handle_specific_course_flow, handle_general_search
from utils.helpers import clear_all_caches

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    force=True
)
logging.getLogger("neo4j.io").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.INFO)

# Register cleanup
atexit.register(clear_all_caches)

# Streamlit config
st.set_page_config(
    page_title="CourseFinder",
    layout="wide",
    initial_sidebar_state="expanded"  
)

# Clear chat history handling
if 'clear_requested' not in st.session_state:
    st.session_state.clear_requested = False

# If page reloaded and clear_requested is true, clear history
if st.session_state.clear_requested:
    keys_to_clear = [
        'messages', 
        'last_results',
        'message_results',
        'show_clarification_buttons',
        'clarification_query'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset flag
    st.session_state.clear_requested = False


# Initialize session state
if 'advisor' not in st.session_state:
    with st.spinner("🚀 Initializing system..."):
        st.session_state.advisor = get_knowledge_base_qa()

if 'llm' not in st.session_state:
    st.session_state.llm = get_llm(small=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'last_results' not in st.session_state:
    st.session_state.last_results = []
    
if 'message_results' not in st.session_state:
    st.session_state.message_results = {}
    
if 'show_clarification_buttons' not in st.session_state:
    st.session_state.show_clarification_buttons = False
    
if 'clarification_query' not in st.session_state:
    st.session_state.clarification_query = ""

# Add CSS for progress animations
add_progress_css()

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ Debug & Tools")
    
    # Clear chat history button
    if st.button("🔄 Clear Chat History"):
        st.session_state.clear_requested = True
        
        # Clear all session state related to chat and UI
        keys_to_clear = [
            'messages', 
            'last_results', 
            'message_results',
            'show_clarification_buttons',
            'clarification_query',
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
        
        # Also clear duplicate UI
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
        # Trigger the query in main interface
        test_query = entity_tests[selected_test]
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
        # Send to main interface
        st.session_state.pending_query = selected_query
        st.rerun()

# Main interface
st.title("🎓 CourseFinder")
st.markdown("Ask me about educational courses and I'll provide comprehensive information!")

# Display chat history
display_chat(llm=st.session_state.llm)

# Handle pending query from sidebar
if 'pending_query' in st.session_state and st.session_state.pending_query:
    pending = st.session_state.pending_query
    del st.session_state.pending_query  # Clear it
    handle_query(pending)

# Clarification buttons
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
            "🔍 Search & Analyze Detail Information", 
            key="btn_general", 
            use_container_width=True,
            help="Find detail information and get landscape overview"
        ):
            st.session_state.show_clarification_buttons = False
            original_query = st.session_state.get('clarification_query', '')
            handle_general_search(original_query)
            st.rerun()
    
    st.markdown("---")

# Chat input - placed at the end
if prompt := st.chat_input("Ask about courses...", key="unique_chat_input"):
    handle_query(prompt)

# Footer
st.markdown("---")
st.caption("💡 Tip: Try asking about specific topics like 'Python courses' or 'Data science landscape'")