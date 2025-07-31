from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models.schemas import ChatMessage, CourseAnalysisResponse
from backend.services.conversation_manager import ConversationManager
from backend.services.course_analyzer import CourseAnalyzer
from backend.services.llm_service import LLMService
from backend.knowledgebase import KnowledgeBaseQA
from shared.database import Neo4jConnector
from shared.embedding import SBERTEmbeddingModel
import time
import signal
import sys
import os
import shutil
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="EduAssistant Backend", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

conv_mgr = ConversationManager()
llm = LLMService()
analyzer = CourseAnalyzer(llm)
neo4j_conn = Neo4jConnector()
embedding_model = SBERTEmbeddingModel()
qa = KnowledgeBaseQA(neo4j_conn, embedding_model, llm)

def generate_followups(user_query, courses, context):
    qs = []
    if courses:
        qs += [
            "Would you like a detailed comparison between these courses?",
            "Any specific course you want to dive deeper into?"
        ]
        skills = {s for c in courses for s in c.get("skills", [])}
        if len(skills) > 1:
            sample = ", ".join(list(skills)[:3])
            qs.append(f"Interested in learning more about skills like {sample}?")
    if context.get("previous_queries"):
        qs.append("Would you like to change or expand your search criteria?")
    return qs[:3]

def clear_caches():
    """Clear Hugging Face cache for SBERT model."""
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    try:
        if os.path.exists(hf_cache_dir):
            shutil.rmtree(hf_cache_dir)
            logger.info("Cleared Hugging Face cache.")
    except Exception as e:
        logger.error(f"Failed to clear Hugging Face cache: {str(e)}")

def signal_handler(sig, frame):
    """Handle termination signals to clean up caches."""
    logger.info("Received termination signal. Cleaning up caches...")
    clear_caches()
    neo4j_conn.close()  # Close Neo4j connection
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.post("/chat", response_model=CourseAnalysisResponse)
async def chat(message: ChatMessage):
    ctx = conv_mgr.get(message.conversation_id or "default")
    preferences = ctx.get("preferences", {})
    raw = await qa.process_query(message.user_query, preferences=preferences)
    analysis = await analyzer.analyze(raw, message.user_query, ctx)
    followups = generate_followups(message.user_query, raw, ctx)
    conv_id = message.conversation_id or f"conv_{int(time.time())}"
    conv_mgr.update(conv_id, message.user_query, analysis, raw)
    return CourseAnalysisResponse(
        analysis=analysis,
        courses=raw,
        conversation_id=conv_id,
        follow_up_questions=followups
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)