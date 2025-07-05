import re
import time
import streamlit as st
from typing import List, Dict
from config import MAX_CHAT_MESSAGES, MAX_RESULTS_DISPLAY

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
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(0, 210, 255, 0.3);
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
    """Progress tracking component with animated UI"""
    
    def __init__(self, container):
        self.container = container
        self.progress_bar = None
        self.text_container = None
        self.current_progress = 0
        self.animation_counter = 0
        
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
        
        # Force animation restart with unique key
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
        setTimeout(() => {{
            const elements = document.querySelectorAll('.animated-text');
            elements.forEach(el => {{
                el.style.animation = 'none';
                el.offsetHeight;
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
            self.animation_counter = 0

def simulate_progress_step(tracker, start_percent, end_percent, message, emoji, duration=1.0):
    """Simulate progress in one step with smooth animation"""
    steps = 3  # Number of small steps for smooth animation
    step_size = (end_percent - start_percent) / steps
    step_duration = duration / steps
    
    for i in range(steps + 1):
        current_percent = start_percent + (i * step_size)
        tracker.update(int(current_percent), message, emoji)
        time.sleep(step_duration)

def clean_answer(text: str) -> str:
    """Remove thinking tags and debug info from LLM responses"""
    if not text:
        return text
    
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove remaining thinking patterns
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
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not any(word in sentence.lower() for word in ['let me', 'i need', 'user wants']):
                text = sentence + '.'
                break
    
    return text

def clear_duplicate_ui():
    """Clear duplicate UI elements and session state"""
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
    
    # Force UI refresh if needed
    try:
        st.cache_data.clear()
    except:
        pass

def manage_chat_history():
    """Optimized chat history management"""
    if len(st.session_state.messages) > MAX_CHAT_MESSAGES:
        clear_duplicate_ui()  # Clear UI when trimming messages
        # Keep only recent messages
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_MESSAGES:]

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
    """Display chat history with results for each message - no duplicates"""
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
                    
                    # Only display results for latest message to avoid duplicates
                    is_latest_result_message = (i == len(st.session_state.messages) - 1)
                    
                    if is_latest_result_message:
                        results = message.get("results", [])
                        result_type = message.get("result_type", "course")
                        
                        if results:
                            st.markdown("---")  # Separator
                            st.markdown(f"### 📊 Results ({len(results)})")
                            
                            # Display results in expandable format
                            for j, r in enumerate(results[:MAX_RESULTS_DISPLAY]):
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
                            
                            # Show "View more" if there are more results
                            if len(results) > MAX_RESULTS_DISPLAY:
                                st.info(f"... and {len(results) - MAX_RESULTS_DISPLAY} more results")
                    else:
                        # For old messages, only show summary
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

def stream_llm_response(llm, user_messages: List[Dict[str, str]], placeholder):
    """Stream LLM response with timeout protection"""
    from config import SYSTEM_PROMPT
    
    # Format messages
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
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Streaming failed: {e}")
        placeholder.markdown("⚠️ Falling back to standard response...")
        full_response = llm.invoke(messages)
    
    # Final cleanup
    full_response = clean_answer(full_response)
    placeholder.markdown(full_response)
    return full_response