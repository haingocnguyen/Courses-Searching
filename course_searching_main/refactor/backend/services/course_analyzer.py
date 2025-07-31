from shared.database import Neo4jConnector
from .llm_service import LLMService
from typing import List, Dict

class CourseAnalyzer:
    def __init__(self, llm: LLMService):
        self.db = Neo4jConnector()
        self.llm = llm

    async def analyze(self, courses: List[Dict], user_query: str, context: Dict=None) -> str:
        if not courses:
            return "I couldn’t find any courses matching your request. Could you try rephrasing?"
        enriched = self._enrich(courses)
        prompt = self._build_prompt(enriched, user_query, context)
        return await self.llm.generate(
            prompt=prompt,       # <-- gọi prompt dưới dạng keyword
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )

    def _enrich(self, courses: List[Dict]) -> List[Dict]:
        detail_cypher = """
        MATCH (c:Course {url: $url})
        OPTIONAL MATCH (c)-[:HAS_REVIEW]->(r:Review)
        OPTIONAL MATCH (c)-[:TEACHES]->(s:Skill)
        OPTIONAL MATCH (c)-[:HAS_LEVEL]->(l:Level)
        OPTIONAL MATCH (c)-[:TAUGHT_BY]->(i:Instructor)
        OPTIONAL MATCH (c)-[:OFFERED_BY]->(o:Organization)
        RETURN
          c.name AS name,
          c.description AS description,
          c.duration AS duration,
          c.rating AS rating,
          collect(r.comment)[0..3]   AS reviews,
          collect(r.rating)[0..3]    AS review_ratings,
          collect(DISTINCT s.name)   AS skills,
          collect(DISTINCT l.name)   AS levels,
          i.name                     AS instructor,
          o.name                     AS provider
        """
        out = []
        for c in courses:
            res = self.db.run(detail_cypher, {"url": c["url"]})
            if res:
                c.update(res[0])
            out.append(c)
        return out

    def _build_prompt(self, courses, user_query, context) -> str:
        ctx = ""
        if context and context.get("previous_queries"):
            last3 = context["previous_queries"][-3:]
            ctx = f"Conversation history: {last3}\n"
        sections = []
        for idx, course in enumerate(courses[:5], 1):
            sec = f"""
Course {idx}: {course['name']}
- Provider: {course['provider']}
- Instructor: {course['instructor']}
- Duration: {course['duration']}
- Rating: {course['rating']}/5
- Skills: {', '.join(course['skills'])}
- Description: {course['description'][:150]}...
- Reviews: {'; '.join(course['reviews']) if course.get('reviews') else 'N/A'}
"""
            sections.append(sec)
        course_block = "\n".join(sections)
        return f"""
You are a friendly education advisor. Analyze these courses for the user.

User question: "{user_query}"
{ctx}

Course details:
{course_block}

Please respond with:
1. A brief summary of what you found.
2. Strengths & weaknesses per course.
3. Comparison if there’s more than one.
4. An overview to help the user decide.
5. Do NOT recommend one specific course.
6. End with an open question to continue the chat.

Style: consultative, non-prescriptive, encouraging user autonomy.
"""
