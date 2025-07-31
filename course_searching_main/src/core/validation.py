import json
import networkx as nx
from typing import Dict
from .llm_integration import LLMAssistant
from .knowledge_base import KnowledgeBase

class ResultValidator:
    VALIDATION_PROMPT = """
    Assess the relevance of a learning resource against the original query:
    
    Original Query: {query}
    Course Details: {course_info}
    
    Evaluation Criteria:
    1. Alignment with stated learning goals
    2. Appropriate difficulty level
    3. Relevance to mentioned career paths
    4. Compatibility with preferred learning style
    5. Certification requirements (if any)
    
    Output JSON:
    {{
        "relevance_score": 0.0-1.0,
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }}
    
    Good Match Example:
    Query: "Project management course for software teams"
    Course: {{
    "title": "Agile Team Leadership",
    "topics": ["scrum", "team coordination"],
    "career_paths": ["tech project manager"],
    "learning_style": "case-study based"
    }}
    Validation:
    {{
    "relevance_score": 0.92,
    "strengths": [
        "Direct alignment with team management focus",
        "Practical case studies suitable for software teams"
    ],
    "weaknesses": [
        "Limited coverage of traditional PM methodologies"
    ],
    "recommendations": [
        "Supplement with waterfall methodology resources"
    ]
    }}

    Partial Match Example:  
    Query: "Python for financial analysis"
    Course: {{
    "title": "Data Science Basics",
    "topics": ["python", "statistics"],
    "career_paths": ["general data analysis"]
    }}
    Validation:
    {{
    "relevance_score": 0.65,
    "strengths": [
        "Strong Python fundamentals",
        "Statistical analysis coverage"
    ],
    "weaknesses": [
        "No financial domain-specific content",
        "Lacks time series analysis modules"
    ],
    "recommendations": [
        "Combine with financial modeling specialization course"
    ]
    }}

    Now evaluate:
    Query: {query}
    Course Details: {course_info}

    Output JSON:
    {{
        "relevance_score": 0.0-1.0,
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }}"""

    # In LearningAssistant summary generation
    SUMMARY_PROMPT = """
    Generate summaries using these examples:

    Example 1:
    Query: "UI/UX design courses with portfolio projects"
    Summary:
    {{
    "summary": "Top courses focus on practical design systems and portfolio development, though some lack advanced prototyping coverage.",
    "key_insights": [
        "Best match: 'Design Studio' course (4.8★) offers real client projects",
        "Consider adding complementary prototyping workshops"
    ],
    "alternatives": [
        "Digital Design Fundamentals + Advanced Prototyping bundle"
    ],
    "next_steps": [
        "Compare mentorship options",
        "Review portfolio requirements"
    ]
    }}

    Example 2:
    Query: "Ethical hacking certification prep under 6 months"
    Summary:
    {{
    "summary": "Certification-aligned programs found with hands-on labs, but require existing networking knowledge.",
    "key_insights": [
        "Top pick: 'Cybersecurity Bootcamp' includes exam voucher",
        "Check prerequisite networking modules"
    ],
    "alternatives": [
        "Self-paced CEH study path"
    ],
    "next_steps": [
        "Verify certification exam dates",
        "Assess lab environment requirements"
    ]
    }}

    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        
    def validate_result(self, course_id: str, query: str) -> dict:
        course = self.kb.data[self.kb.data['course_id'] == course_id].iloc[0]
        graph_context = self._get_graph_context(course_id)
        prompt = self.VALIDATION_PROMPT.format(
            query=query,
            course_info=json.dumps(course['features']),
            graph_context=graph_context
        )
        return self.llm.generate(prompt)
    
    def _get_graph_context(self, course_id: str) -> str:
        """Extract relevant graph neighborhood"""
        neighbors = []
        for neighbor in nx.neighbors(self.kb.graph, course_id):
            neighbors.append({
                'node': neighbor,
                'type': self.kb.graph.nodes[neighbor].get('type'),
                'relationship': self.kb.graph.edges[(course_id, neighbor)].get('relationship')
            })
        return json.dumps(neighbors[:5])