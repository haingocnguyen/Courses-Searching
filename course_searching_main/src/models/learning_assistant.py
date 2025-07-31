import time
import traceback
from core.knowledge_base import KnowledgeBase
from core.llm_integration import LLMAssistant
from core.query_processing import QueryProcessor, AdaptiveSearcher
from core.validation import ResultValidator
import json

class LearningAssistant:
    def __init__(self, data_path: str, use_caching=True):
        self.kb = KnowledgeBase(data_path)
        self.llm = LLMAssistant()
        self.processor = QueryProcessor(self.kb)
        self.searcher = AdaptiveSearcher(self.kb)
        self.validator = ResultValidator(self.kb)
        
    def process_query(self, query: str) -> dict:
        response = {
            "query": query,
            "analysis": {},
            "results": {
                "courses": [],
                "summary": {
                    "overview": "",
                    "top_recommendations": [],
                    "considerations": []
                }
            },
            "metrics": {
                "processing_time": 0,
                "results_count": 0
            }
        }
        
        try:
            start_time = time.time()
            # Step 1: Query Understanding
            analysis = self.processor.analyze_query(query)
            response['analysis'] = analysis
            
            # Step 2: Search Execution
            results = self.searcher.execute_search(analysis)
            
            # Step 3: Result Validation
            # In LearningAssistant.process_query
            # Step 3: Result Validation
            validated = []
            initial_results_count = len(results)

            # Dynamic threshold calculation based on result quantity
            threshold = max(0.15, 0.4 - (0.05 * initial_results_count / 10))  # Becomes more lenient with fewer results

            # Validate top 50 results with progressive scoring
            for idx, course in enumerate(results[:50]):
                validation = self.validator.validate_result(course['course_id'], query)
                
                # Progressive threshold: first 10 results get 0.8*threshold, next 20 get threshold, rest get 1.2*threshold
                current_threshold = threshold * (0.8 if idx < 10 else 1.0 if idx < 30 else 1.2)
                
                if validation.get('relevance_score', 0) >= current_threshold:
                    validated.append({**course, "validation": validation})
                    
                # Early exit if we have enough high-quality matches
                if len(validated) >= 10:
                    break

            # Fallback: Take top 3 by semantic score if no validations
            if not validated:
                sorted_results = sorted(results, 
                                    key=lambda x: x.get('score', 0), 
                                    reverse=True)
                validated = sorted_results[:3]
                print("Using fallback results")
            # Step 4: Generate Summary
            response['results']['courses'] = validated
            response['results']['summary'] = self._generate_summary(validated, query)
            # Final formatting
            response['metrics'] = {
                "processing_time": time.time() - start_time,
                "results_count": len(validated)
            }
            
            # Ensure summary structure
            if 'summary' not in response['results']:
                response['results']['summary'] = {
                    "overview": "No summary generated",
                    "top_recommendations": [],
                    "considerations": []
                }
            print(f"\nSearch Stats: {len(results)} initial results, {len(validated)} after validation")
                
        except Exception as e:
            response['error'] = str(e)
            response['traceback'] = traceback.format_exc()
        
        return response

    def _generate_summary(self, results: list, query: str) -> dict:
        summary_prompt = f"""Generate a natural language summary for search results:
        
        Query: {query}
        Top Results: {json.dumps(results[:3])}
        
        Include:
        - Overall match quality
        - Key strengths of top results
        - Potential limitations to note
        - Alternative suggestions if relevant
        
        Output JSON:
        {{
            "summary": "...",
            "key_insights": [],
            "alternatives": [],
            "next_steps": []
        }}"""
        
        try:
            summary = self.validator.llm.generate(summary_prompt)
            return summary if isinstance(summary, dict) else {}
        except Exception as e:
            print(f"Summary generation failed: {str(e)}")
            return {}