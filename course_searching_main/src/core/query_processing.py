from typing import Dict, List
import numpy as np
import pandas as pd
import networkx as nx
import time
from core.knowledge_base import KnowledgeBase
from core.llm_integration import LLMAssistant
class QueryProcessor:
    PROMPT_TEMPLATE = """
    Analyze the learning-related query and identify core components:
    
    Query: {query}
    
    Available Context Types:
    - Topics (e.g., programming, business)
    - Skills (e.g., Python, project management)
    - Difficulty Levels (beginner, intermediate, advanced)
    - Learning Styles (self-paced, instructor-led)
    - Career Paths (data science, web development)
    - Certifications (AWS, PMP)
    - Duration Constraints
    - Prerequisite Requirements
    
    Output JSON structure:
    {{
        "components": {{
            "learning_goals": [],
            "constraints": {{
                "difficulty": "",
                "duration_max": null,
                "required_prerequisites": [],
                "excluded_topics": []
            }},
            "preferences": {{
                "learning_style": "",
                "certifications": []
            }},
            "career_connections": []
        }},
        "search_strategy": {{
            "primary_approach": "", # "graph" should be prioritized, else if result is poor, hybrid and sematic be applied
            "fallback_approaches": [],
            "reasoning": "Step-by-step explanation of the analysis"
        }}
    }}
    Analyze the learning-related query and identify core components using these examples:

    Example 1:
    Query: "I want to switch to data science but only have basic math skills"
    Output:
    {{
    "components": {{
        "learning_goals": ["data science fundamentals"],
        "constraints": {{
        "difficulty": "beginner",
        "required_prerequisites": ["basic math"],
        "excluded_topics": ["advanced statistics"]
        }},
        "preferences": {{
        "learning_style": "hands-on"
        }},
        "career_connections": ["data science career transition"]
    }},
    "search_strategy": {{
        "primary_approach": "graph",
        "fallback_approaches": ["semantic", "constraint-based"],
        "reasoning": "Focus on career transition paths while filtering out advanced math requirements"
    }}
    }}

    Example 2:
    Query: "Advanced cloud security courses with AWS certification prep under 3 months"
    Output:
    {{
    "components": {{
        "learning_goals": ["cloud security", "AWS certification"],
        "constraints": {{
        "duration_max": 3,
        "required_prerequisites": ["basic cloud computing"]
        }},
        "preferences": {{
        "certifications": ["AWS Certified Security"]
        }},
        "career_connections": ["cloud security engineer"]
    }},
    "search_strategy": {{
        "primary_approach": "graph",
        "fallback_approaches": ["duration-filtered", "skill-graph"],
        "reasoning": "Prioritize certification-aligned content with strict timeline constraints"
    }}
    }}

    Now analyze this new query:
    Query: {query}

    Output JSON structure:
    {{
        "components": {{
            "learning_goals": [],
            "constraints": {{
                "difficulty": "",
                "duration_max": null,
                "required_prerequisites": [],
                "excluded_topics": []
            }},
            "preferences": {{
                "learning_style": "",
                "certifications": []
            }},
            "career_connections": []
        }},
        "search_strategy": {{
            "primary_approach": "",
            "fallback_approaches": [],
            "reasoning": "Step-by-step explanation of the analysis"
        }}
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        
    def analyze_query(self, query: str) -> dict:
        """Add this missing method implementation"""
        prompt = self.PROMPT_TEMPLATE.format(query=query)
        analysis = self.llm.generate(prompt)
        return self._normalize_analysis(analysis)
    def _normalize_analysis(self, raw: dict) -> dict:
        """Ensure consistent structure for downstream processing"""
        normalized = {
            'components': raw.get('components', {}),
            'strategy': raw.get('search_strategy', {})
        }
        
        # Normalize constraint values
        constraints = normalized['components'].get('constraints', {})
        if 'duration_max' in constraints:
            try:
                constraints['duration_max'] = int(constraints['duration_max'])
            except:
                constraints['duration_max'] = None
                
        return normalized

# ======================
# 4. Flexible Search System
# ======================
class AdaptiveSearcher:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.llm = LLMAssistant()
        
    def execute_search(self, query_analysis: dict) -> list:
        strategy = query_analysis['strategy']['primary_approach']
        
        if strategy == 'semantic':
            results = self._semantic_search(query_analysis)
        elif strategy in ('graph', 'topic-based'):
            results = self._graph_search(query_analysis)
        else:
            results = self._hybrid_search(query_analysis)
            
        return self._rerank_results(results, query_analysis)
        
    def _hybrid_search(self, analysis: dict) -> list:
        # Existing semantic search
        semantic_results = self._semantic_search(analysis)
        
        # Boost graph search parameters
        graph_results = self._graph_search(analysis)
        
        # Add keyword fallback
        keyword_results = self._keyword_fallback(analysis)
        
        # Combine all results
        combined = semantic_results + graph_results + keyword_results
        
        # Deduplicate
        return list({c['course_id']: c for c in combined}.values())
 
    def _keyword_fallback(self, analysis: dict) -> list:
        """Fallback to simple keyword search"""
        terms = analysis['components']['learning_goals']
        return self.kb.data[
            self.kb.data['search_text'].str.contains('|'.join(terms), case=False)
        ].to_dict('records')
    def _semantic_search(self, analysis: dict) -> list:
        search_terms = ' '.join(
            analysis['components']['learning_goals'] +
            analysis['components']['career_connections']
        )
        return self.kb.semantic_search(search_terms, top_k=50)

    def _graph_search(self, analysis: dict) -> list:
        components = analysis['components']
        results = []
        
        search_terms = components['learning_goals'] + components['career_connections']
        
        for term in search_terms:
            try:
                # Find matching concept nodes
                matches = [n for n in self.kb.graph.nodes 
                        if term.lower() in n.lower() 
                        and self.kb.graph.nodes[n].get('type') in ['concept', 'career']]
                
                # Find courses within 3 hops
                for match in matches:
                    # Check multiple hop distances
                    for distance in [1, 2, 3]:
                        for course_node in nx.descendants_at_distance(self.kb.graph, match, distance):
                            if self.kb.graph.nodes[course_node].get('type') == 'course':
                                course = self.kb.data[self.kb.data['course_id'] == course_node].iloc[0].to_dict()
                                results.append(course)
            except Exception as e:
                print(f"Graph search error: {str(e)}")
        
        return results
    
    def _apply_filters(self, results: list, components: dict) -> list:
        # Dynamic constraint application
        filtered = []
        constraints = components['constraints']
        prefs = components['preferences']
        
        for course in results:
            # Check hard constraints
            if not self._meets_constraints(course, constraints):
                continue
                
            # Check preferences
            course['score'] = self._calculate_preference_score(course, prefs)
            
            filtered.append(course)
            
        return filtered
    
    def _calculate_preference_score(self, course: dict, prefs: dict) -> float:
        # Score based on preference matches
        score = 0.0
        if course['learning_style'] == prefs.get('learning_style', ''):
            score += 0.3
        if any(cert in course.get('certifications', []) for cert in prefs.get('certifications', [])):
            score += 0.2
        return score
    def _rerank_results(self, results: list, analysis: dict) -> list:
        """Multi-factor ranking with KG context"""
        ranked = []
        for course in results:
            features = course['features']
            course_id = course['course_id']
            
            # Calculate score components
            semantic_score = self._calculate_semantic_match(course, analysis)
            graph_score = self.kb.graph.nodes[course_id].get('centrality', 0)
            freshness_score = np.log(time.time() - pd.to_datetime(course.get('created', 0)).timestamp())
            
            # Combine scores
            total_score = (
                0.4 * semantic_score +
                0.3 * graph_score +
                0.2 * freshness_score +
                0.1 * course.get('rating', 0)/5.0
            )
            
            ranked.append((total_score, course))
            
        return [x[1] for x in sorted(ranked, reverse=True)]

    def _calculate_semantic_match(self, course, analysis):
        """Context-aware similarity scoring"""
        query_embed = self.kb.encoder.encode(
            ' '.join(analysis['components']['learning_goals'])
        )
        course_embed = self.kb.encoder.encode(course['search_text'])
        return np.dot(query_embed, course_embed)
