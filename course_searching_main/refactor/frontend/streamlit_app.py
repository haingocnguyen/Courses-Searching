# frontend/streamlit_app.py
import streamlit as st
import requests, uuid, json
from datetime import datetime

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="EduAssistant", layout="wide")

if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Placeholder for selected course details
if "detail" not in st.session_state:
    st.session_state.detail = None

def render_course_card(c):
    with st.container():
        cols = st.columns([4,1])
        with cols[0]:
            st.markdown(f"**[{c['name']}]({c['url']})**")
            st.caption(f"{c.get('rating','N/A')}/5 • {c.get('duration','N/A')} • {c.get('provider','N/A')}")
            if c.get("skills"):
                st.markdown(f"**Skills:** {', '.join(c['skills'][:3])}")
        with cols[1]:
            if st.button("Details", key=c['url']):
                st.session_state.detail = c


def chat_interface():
    st.title("🎓 EduAssistant")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")
            if msg.get("follow_up_questions"):
                for i, q in enumerate(msg["follow_up_questions"]):
                    key = f"followup_{st.session_state.conv_id}_{i}"
                    if st.button(q, key=key):
                        send(q)
            if msg.get("courses"):
                with st.expander(f"📚 {len(msg['courses'])} courses"):
                    for c in msg["courses"]:
                        render_course_card(c)

    query = st.chat_input("Ask about courses...")
    if query:
        send(query)


def send(text):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": text})
    
    # Send request to backend
    resp = requests.post(f"{BACKEND}/chat", json={
        "user_query": text,
        "conversation_id": st.session_state.conv_id
    })
    
    # Handle response
    if resp.status_code == 200:
        data = resp.json()
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["analysis"],
            "courses": data["courses"],
            "follow_up_questions": data["follow_up_questions"]
        })
    else:
        st.error("Error: " + resp.text)
    
    # Rerun the app to update the UI
    st.rerun()  


def detail_expander():
    c = st.session_state.detail
    if c:
        with st.expander(f"Details: {c['name']}", expanded=True):
            st.markdown(f"**Description:** {c.get('description', 'N/A')}")
            st.write(f"**Instructor:** {c.get('instructor','N/A')}")
            st.write(f"**Provider:** {c.get('provider','N/A')}")
            if c.get("reviews"):
                st.markdown("**User Reviews:**")
                for rev, score in zip(c.get("reviews", []), c.get("review_ratings", [])):
                    st.write(f"- ⭐ {score}: {rev}")
        # Clear detail after showing
        st.session_state.detail = None


if __name__ == "__main__":
    chat_interface()
    detail_expander()
