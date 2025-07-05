import re
import logging
from sentence_transformers import SentenceTransformer, util
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class FlexibleIntentClassifier:
    """Classifies user intent for educational vs general chat queries"""
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        
        # Educational contexts - academic/professional topics only
        self.educational_contexts = [
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
            "find instructors teaching machine learning",
            "who are the best Python instructors",
            "show me instructors with rating above 4.5",
            "what organizations offer data science courses",
            "list top universities providing programming courses",
            "which platforms have programming courses",
            "compare course providers like Coursera and edX",
            "show me reviews for Python programming courses",
            "what subjects are covered in computer science",
            "what skills can I learn from programming courses",
            "what difficulty levels are available",
            "average rating of programming courses"
        ]
        
        # General chat contexts including fantasy/casual topics
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
        """Classify user intent with pattern matching and embedding similarity"""
        query_lower = query.lower().strip()
        
        # Priority 1: Detect fantasy/casual topics first
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
        
        # Priority 2: Academic course patterns - stricter criteria
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
        
        # Check strict academic patterns
        for pattern in academic_course_patterns:
            if re.search(pattern, query_lower):
                return {"intent": "course_search", "confidence": "high",
                        "details": {"edu": 0.9, "chat": 0.1, "scope": 0.0, "method": "academic_pattern_match"}}
        
        # Priority 3: Direct pattern matching for other cases
        identity_patterns = ["who are you", "what are you", "who r u", "what r u"]
        greeting_patterns = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey"]
        thanks_patterns = ["thank you", "thanks", "thx"]
        retrospective_patterns = ["what did i ask", "what was my question", "what did i say", "my previous question"]
        
        if any(pattern in query_lower for pattern in identity_patterns):
            return {"intent": "scope_inquiry", "confidence": "high",
                    "details": {"edu": 0.0, "chat": 0.0, "scope": 0.9, "method": "pattern_match"}}
        
        if any(query_lower.startswith(pattern) for pattern in greeting_patterns):
            return {"intent": "general_chat", "confidence": "high", 
                    "details": {"edu": 0.0, "chat": 0.9, "scope": 0.0, "method": "pattern_match"}}
        
        if any(pattern in query_lower for pattern in thanks_patterns):
            return {"intent": "general_chat", "confidence": "high",
                    "details": {"edu": 0.0, "chat": 0.9, "scope": 0.0, "method": "pattern_match"}}
        
        # Priority 4: Casual "course" mentions should be chat
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
        
        # Priority 5: Use embedding-based classification with higher thresholds
        q_emb = self.model.encode(query, convert_to_tensor=True)
        edu_sim = util.pytorch_cos_sim(q_emb, self.edu_embs)[0].max().item()
        chat_sim = util.pytorch_cos_sim(q_emb, self.chat_embs)[0].max().item()
        scope_sim = util.pytorch_cos_sim(q_emb, self.scope_embs)[0].max().item()
        
        # Higher thresholds to avoid false positives
        if max(edu_sim, chat_sim, scope_sim) < 0.3:
            # Enhanced keyword fallback - only academic keywords + course context
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

        # Priority 6: Classification logic with higher thresholds
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