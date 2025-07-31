import logging
from models.llm import OllamaLLM
from config import LLM_MODEL
import json
import re

logger = logging.getLogger(__name__)

class FlexibleIntentClassifier:
    """Classifies user intent using LLM with consolidated categories"""
    
    def __init__(self, llm_model=LLM_MODEL):
        self.llm = OllamaLLM(model=llm_model)

    def classify_intent(self, query: str, chat_history: list = None) -> dict:
        """Classify user intent using LLM with comprehensive examples in prompt"""
        
        context_info = self._build_context_info(chat_history) if chat_history else ""
        
        prompt = f"""You are an intent classification system. Analyze the user's query and classify it into exactly ONE of these two categories based on the examples below:

=== GENERAL_CHAT EXAMPLES ===
• "hello how are you doing today"
• "hi"
• "hello"
• "good morning have a great day"
• "thank you so much for your help"
• "tell me a funny joke please"
• "I am feeling tired today"
• "tell me an interesting story"
• "nice to meet you there"
• "goodbye see you later"
• "thanks for the assistance"
• "beginner"
• "intermediate"
• "advanced"
• "courses"
• "course"
• "training"
• "learn something"
• "I want to study"
• "show me courses"
• "what can I learn"
• "help me learn"
• "recommend me courses"
• "what should I learn"
• "show me some reviews"
• "reviews"
• "recommendations"
• "suggest something"
• "courses to become a superhero"
• "training to be a ninja"
• "how to get superpowers"
• "become a dragon trainer"
• "magical powers course"
• "superhero training academy"
• "wizard school applications"
• "how many courses do you have"
• "tell me about your system capability"
• "show details about your database"
• "what do you do as an assistant"
• "what is your specific scope of work"
• "tell me about your capabilities and features"
• "what are your main functions and abilities"
• "what can you help me with specifically"
• "what is this system designed for"
• "what services do you provide to users"
• "what is your primary function here"
• "explain your role and responsibilities"
• "what kind of assistance do you offer"
• "who are you exactly"
• "what is your name"
• "any courses to become a superhero?"

=== COURSE_SEARCH EXAMPLES ===
• "find programming courses for beginners"
• "search for data science training programs"
• "show me machine learning courses with high ratings"
• "I need Python courses for web development"
• "recommend beginner courses in artificial intelligence"
• "Java programming courses online"
• "web development bootcamp recommendations"
• "statistics courses for data analysis"
• "find instructors teaching machine learning"
• "who are the best Python instructors"
• "show me instructors with rating above 4.5"
• "what organizations offer data science courses"
• "show me reviews for Python programming courses"
• "JavaScript courses for beginners"
• "advanced Python programming courses"
• "React development training programs"
• "SQL database courses for data analysts"
• "find information about Obsidian productivity courses"
• "search for Second Brain methodology training"

CLASSIFICATION GUIDELINES:
1. Compare the user's query with the examples above
2. Choose the category that contains the most similar examples
3. Notice patterns:
   - GENERAL_CHAT: greetings, single words, vague requests, casual conversation, fantasy topics, system capability questions, identity questions
   - COURSE_SEARCH: specific educational requests with clear topics and action words

{context_info}

USER QUERY TO CLASSIFY: "{query}"

Analyze the query against the examples above and respond with ONLY this JSON format:
{{"intent": "general_chat|course_search", "confidence": "high|medium|low", "reason": "explanation based on similarity to examples"}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            
            # Parse response
            result = self._parse_llm_response(response, query)
            result["details"] = {"method": "llm_comprehensive_examples"}
            
            logger.info(f"Intent classified: {result['intent']} (confidence: {result['confidence']}) for query: '{query}'")
            return result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to simple pattern matching
            return self._pattern_fallback(query)

    def _build_context_info(self, chat_history: list) -> str:
        """Build context information from chat history"""
        if not chat_history:
            return ""
            
        recent_messages = chat_history[-4:] 
        context_parts = []
        
        for msg in recent_messages:
            if msg["role"] == "user" and "content" in msg:
                content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                context_parts.append(f"User: {content}")
            elif msg["role"] == "assistant" and msg.get("type") != "courses":
                content = msg.get('content', '')[:60] + "..." if len(msg.get('content', '')) > 60 else msg.get('content', '')
                if content:
                    context_parts.append(f"Assistant: {content}")
        
        if context_parts:
            context = "\n".join(context_parts[-3:])  
            return f"\nCONVERSATION CONTEXT:\n{context}\n"
        
        return ""

    def _parse_llm_response(self, response: str, query: str) -> dict:
        """Parse LLM response to extract intent and confidence"""
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                if "intent" in parsed:
                    intent = parsed.get("intent", "general_chat")
                    # Validate intent
                    if intent not in ["general_chat", "course_search"]:
                        intent = "general_chat"
                    return {
                        "intent": intent,
                        "confidence": parsed.get("confidence", "medium"),
                        "reason": parsed.get("reason", "LLM classification")
                    }
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        # Fallback parsing nếu JSON fails
        response_lower = response.lower()
        
        # Extract intent
        if 'general_chat' in response_lower:
            intent = 'general_chat'
        elif 'course_search' in response_lower:
            intent = 'course_search'
        else:
            # Default to general_chat
            intent = 'general_chat'
        
        # Extract confidence
        if 'high' in response_lower:
            confidence = 'high'
        elif 'low' in response_lower:
            confidence = 'low'
        else:
            confidence = 'medium'
        
        # Extract reason if available
        reason = "LLM classification"
        try:
            if '"reason"' in response_lower or 'reason:' in response_lower:
                reason_patterns = [
                    r'"reason":\s*"([^"]+)"',
                    r'reason:\s*"([^"]+)"',
                    r'reason:\s*([^\n\}]+)'
                ]
                for pattern in reason_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        reason = match.group(1).strip(' "')
                        break
        except:
            pass
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason
        }

    def _pattern_fallback(self, query: str) -> dict:
        """Simple pattern-based fallback when LLM fails"""
        query_lower = query.lower().strip()
        
        greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if query_lower in greeting_words:
            return {
                "intent": "general_chat", 
                "confidence": "high", 
                "reason": "greeting word",
                "details": {"method": "pattern_fallback"}
            }
        
        # Single vague words
        vague_words = ["course", "courses", "training", "beginner", "intermediate", "advanced", "learn", "study"]
        if query_lower in vague_words:
            return {
                "intent": "general_chat", 
                "confidence": "high", 
                "reason": "vague single word",
                "details": {"method": "pattern_fallback"}
            }
        
        # Thanks patterns
        if any(p in query_lower for p in ["thank you", "thanks", "thx"]):
            return {
                "intent": "general_chat", 
                "confidence": "high", 
                "reason": "gratitude expression",
                "details": {"method": "pattern_fallback"}
            }
            
        # Identity/capability questions (now part of general chat)
        identity_patterns = ["who are you", "what are you", "your capabilities", "what can you do", "what is your role", "system capability", "your database"]
        if any(p in query_lower for p in identity_patterns):
            return {
                "intent": "general_chat", 
                "confidence": "high", 
                "reason": "identity/capability question",
                "details": {"method": "pattern_fallback"}
            }
        
        # Specific course search patterns
        search_patterns = ["find", "search for", "show me", "i need", "recommend"]
        tech_topics = ["python", "javascript", "java", "react", "sql", "data science", "machine learning", "programming", "web development"]
        
        has_search_pattern = any(pattern in query_lower for pattern in search_patterns)
        has_tech_topic = any(topic in query_lower for topic in tech_topics)
        
        if has_search_pattern and has_tech_topic and len(query.split()) > 3:
            return {
                "intent": "course_search", 
                "confidence": "medium", 
                "reason": "specific educational search detected",
                "details": {"method": "pattern_fallback"}
            }
        
        # Default to general chat
        return {
            "intent": "general_chat", 
            "confidence": "low", 
            "reason": "default fallback",
            "details": {"method": "pattern_fallback"}
        }