# backend/models/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class ChatMessage(BaseModel):
    user_query: str
    conversation_id: Optional[str] = None
    context: Optional[Dict] = None

class CourseAnalysisResponse(BaseModel):
    analysis: str
    courses: List[Dict]
    conversation_id: str
    follow_up_questions: List[str]
