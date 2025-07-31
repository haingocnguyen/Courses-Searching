import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import requests
import openai
from openai import OpenAI

# Setup relative imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.intent_classifier import FlexibleIntentClassifier
from core.knowledge_base import KnowledgeBaseQA, get_knowledge_base_qa
from database.neo4j_client import get_neo4j_connection
from models.embedding import get_embedding_model, SBERTEmbeddingModel
from core.query_processor import QueryProcessor
from utils.helpers import EnhancedResultHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_judge_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_test_cases_from_file(file_path: str) -> List[Dict]:
    """Load test cases from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        # Validate test case structure
        required_fields = ['id', 'query', 'type', 'expected_intent', 'expected_outcomes']
        for i, test_case in enumerate(test_cases):
            missing_fields = [field for field in required_fields if field not in test_case]
            if missing_fields:
                logger.warning(f"Test case {i+1} missing fields: {missing_fields}")
        
        logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        return test_cases
    
    except FileNotFoundError:
        logger.error(f"Test cases file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in test cases file: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading test cases: {e}")
        return []
class GPTJudge:
    """GPT-3.5 Turbo Judge for evaluating course advisor responses"""
    
    def __init__(self, api_key: str = None):
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY')
        )
        self.model = "gpt-3.5-turbo"
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def query_gpt(self, prompt: str, max_tokens: int = 800) -> str:
        """Query GPT-3.5 Turbo via OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for conversational AI systems. Provide objective, detailed evaluations in the exact format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent judgments
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            return "Error: Rate limit exceeded"
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "Error: OpenAI API error"
        except Exception as e:
            logger.error(f"Error querying GPT: {e}")
            return "Error: Connection failed"
    
    def evaluate_intent_classification(self, query: str, predicted_intent: str, 
                                     expected_intent: str, confidence: float) -> Dict:
        """Evaluate intent classification accuracy using GPT-3.5 Turbo"""
        
        prompt = f"""You are an expert evaluator for intent classification systems. Please evaluate the following case:

USER QUERY: "{query}"
EXPECTED INTENT: {expected_intent}
PREDICTED INTENT: {predicted_intent}
CONFIDENCE SCORE: {confidence}

Please evaluate based on these criteria:

1. Intent Accuracy (0-10): Is the predicted intent semantically correct for this query?
   - Consider edge cases and ambiguous queries
   - 10 = perfect match, 7-9 = good match, 4-6 = partially correct, 0-3 = incorrect

2. Confidence Appropriateness (0-10): Is the confidence level reasonable given the query complexity?
   - High confidence should match clear, unambiguous queries
   - Lower confidence is appropriate for ambiguous cases

3. Overall Intent Quality (0-10): Holistic assessment considering both accuracy and confidence

Please respond with ONLY a valid JSON object in this exact format:
{{
    "intent_accuracy": <integer_0_to_10>,
    "confidence_appropriateness": <integer_0_to_10>,
    "overall_intent_quality": <integer_0_to_10>,
    "reasoning": "<detailed_explanation_of_your_evaluation>",
    "is_correct": <true_or_false>
}}"""
        
        response = self.query_gpt(prompt)
        return self._parse_json_response(response, {
            "intent_accuracy": 5,
            "confidence_appropriateness": 5,
            "overall_intent_quality": 5,
            "reasoning": "Failed to parse GPT response",
            "is_correct": predicted_intent.lower() == expected_intent.lower()
        })
    
    def evaluate_response_quality(self, query: str, response: str, 
                                results_count: int, expected_type: str,
                                found_results: List[Dict] = None) -> Dict:
        """Evaluate the quality of the final response using GPT-3.5 Turbo"""
        
        # Include sample results context if available
        results_context = ""
        if found_results and len(found_results) > 0:
            results_context = f"\nACTUAL RESULTS FOUND: {len(found_results)} courses\n"
            for i, result in enumerate(found_results[:3], 1):
                results_context += f"{i}. {result.get('name', 'N/A')} (Rating: {result.get('rating', 'N/A')})\n"
        
        prompt = f"""You are an expert evaluator for educational course recommendation systems. Evaluate the following interaction:

USER QUERY: "{query}"
QUERY TYPE: {expected_type}
SYSTEM RESPONSE: "{response}"
RESULTS FOUND: {results_count}{results_context}

Evaluate the response quality on these 6 dimensions (each scored 0-10):

1. RELEVANCE: How well does the response address the user's specific query?
   - 10 = perfectly addresses the query, 7-9 = mostly relevant, 4-6 = partially relevant, 0-3 = off-topic

2. HELPFULNESS: How useful is this response for the user's goal?
   - Consider actionable information, next steps, guidance provided

3. CLARITY: How clear, understandable, and well-structured is the response?
   - Consider language complexity, organization, readability

4. COMPLETENESS: Does the response provide sufficient information?
   - Consider whether key details are included or missing

5. PROFESSIONAL_TONE: Is the tone appropriate, friendly, and professional?
   - Suitable for an educational assistant context

6. HALLUCINATION: Does the response contain false, invented, or unsupported information?
   - 10 = no hallucinations, completely factual, 7-9 = minor inaccuracies, 4-6 = some false info, 0-3 = major hallucinations
   - Consider if response mentions courses/instructors/details not in actual results

Please respond with ONLY a valid JSON object:
{{
    "relevance": <integer_0_to_10>,
    "helpfulness": <integer_0_to_10>,
    "clarity": <integer_0_to_10>,
    "completeness": <integer_0_to_10>,
    "professional_tone": <integer_0_to_10>,
    "hallucination": <integer_0_to_10>,
    "overall_score": <average_of_all_scores>,
    "reasoning": "<detailed_explanation_of_evaluation>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "suggestions": ["<improvement1>", "<improvement2>"]
}}"""
        
        response_eval = self.query_gpt(prompt, max_tokens=1000)
        return self._parse_json_response(response_eval, {
            "relevance": 5,
            "helpfulness": 5,
            "clarity": 5,
            "completeness": 5,
            "professional_tone": 7,
            "hallucination": 8,
            "overall_score": 5.83,
            "reasoning": "Failed to parse GPT response",
            "strengths": ["Response was provided"],
            "weaknesses": ["Evaluation failed"],
            "suggestions": ["Check GPT connection"]
        })

    def _parse_json_response(self, response: str, fallback: Dict) -> Dict:
        """Parse JSON response from GPT with fallback"""
        try:
            # Find JSON content
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields exist
                if isinstance(parsed, dict):
                    return parsed
                    
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
        except Exception as e:
            logger.warning(f"Error parsing GPT response: {e}")
        
        logger.info("Using fallback evaluation values")
        return fallback

    def test_connection(self) -> bool:
        """Test GPT API connection"""
        try:
            test_response = self.query_gpt("Respond with just 'OK' to test connection.")
            return "OK" in test_response.upper()
        except:
            return False

# Enhanced test cases (removed evaluation_focus for consistent evaluation)
ENHANCED_TEST_CASES = [
    # {
    #     "id": "TC1",
    #     "query": "Tell me about Introduction to Machine Learning courses",
    #     "type": "course_specific",
    #     "expected_intent": "course_search",
    #     "expected_outcomes": {
    #         "should_find_results": True,
    #         "expected_keywords": ["machine learning", "ML", "introduction", "beginner"],
    #         "minimum_results": 1
    #     }
    # },
    # {
    #     "id": "TC2", 
    #     "query": "Find beginner courses in data science",
    #     "type": "course_general",
    #     "expected_intent": "course_search",
    #     "expected_outcomes": {
    #         "should_find_results": True,
    #         "expected_keywords": ["data science", "beginner", "introduction"],
    #         "minimum_results": 1
    #     }
    # },
    # {
    #     "id": "TC3",
    #     "query": "Who are the best instructors for machine learning?",
    #     "type": "instructor_query",
    #     "expected_intent": "course_search",
    #     "expected_outcomes": {
    #         "should_find_results": True,
    #         "expected_keywords": ["instructor", "teacher", "machine learning"],
    #         "minimum_results": 1
    #     }
    # },
    # {
    #     "id": "TC4",
    #     "query": "What organizations offer web development courses?",
    #     "type": "organization_query",
    #     "expected_intent": "course_search",
    #     "expected_outcomes": {
    #         "should_find_results": True,
    #         "expected_keywords": ["organization", "provider", "web development"],
    #         "minimum_results": 1
    #     }
    # },
    # {
    #     "id": "TC5",
    #     "query": "Hello, how are you today?",
    #     "type": "greeting",
    #     "expected_intent": "general_chat",
    #     "expected_outcomes": {
    #         "should_find_results": False,
    #         "expected_keywords": ["help", "assist", "courses"],
    #         "minimum_results": 0
    #     }
    # },
    # {
    #     "id": "TC6",
    #     "query": "Thank you for your help!",
    #     "type": "thanks",
    #     "expected_intent": "general_chat", 
    #     "expected_outcomes": {
    #         "should_find_results": False,
    #         "expected_keywords": ["welcome", "help", "more"],
    #         "minimum_results": 0
    #     }
    # },
    # {
    #     "id": "TC7",
    #     "query": "What can you help me with?",
    #     "type": "scope_inquiry",
    #     "expected_intent": "general_chat",
    #     "expected_outcomes": {
    #         "should_find_results": False,
    #         "expected_keywords": ["courses", "help", "search", "find"],
    #         "minimum_results": 0
    #     }
    # },
    # {
    #     "id": "TC8",
    #     "query": "Find courses to become a superhero",
    #     "type": "fantasy",
    #     "expected_intent": "general_chat",
    #     "expected_outcomes": {
    #         "should_find_results": False,
    #         "expected_keywords": ["sorry", "help", "available courses"],
    #         "minimum_results": 0
    #     }
    # },
    # {
    #     "id": "TC9",
    #     "query": "Show me Java programming courses with rating above 4.5",
    #     "type": "course_specific",
    #     "expected_intent": "course_search",
    #     "expected_outcomes": {
    #         "should_find_results": True,
    #         "expected_keywords": ["Java", "programming", "rating", "4.5"],
    #         "minimum_results": 1
    #     }
    # },
    {
        "id": "TC181",
        "query": "Ever feel like you’re in a never-ending job interview?",
        "type": "general_chat",
        "expected_intent": "general_chat",
        "expected_outcomes": { "should_find_results": False, "expected_keywords": ["interview", "never-ending", "find", "courses"], "minimum_results": 0 }
    },
    {
        "id": "TC182",
        "query": "What do you do when you're not answering questions?",
        "type": "general_chat",
        "expected_intent": "general_chat",
        "expected_outcomes": { "should_find_results": False, "expected_keywords": ["not answering", "questions", "find", "courses"], "minimum_results": 0 }
  },
]

class EnhancedEvaluationSystem:
    """Enhanced evaluation system with GPT-3.5 Turbo judge"""
    
    def __init__(self, openai_api_key: str = None):
        # Initialize system components
        self.neo4j_conn = get_neo4j_connection()
        self.embedding_model = SBERTEmbeddingModel("all-MiniLM-L6-v2")
        self.advisor = KnowledgeBaseQA(self.neo4j_conn, self.embedding_model)
        self.intent_classifier = FlexibleIntentClassifier("qwen3:4b")  # Local LLM
        self.result_handler = EnhancedResultHandler("qwen3:4b")  # Local LLM
        self.gpt_judge = GPTJudge(openai_api_key)  # GPT-3.5 Turbo Judge
        
    def run_enhanced_test_case(self, test_case: Dict) -> Dict:
        """Run enhanced test case with comprehensive evaluation"""
        logger.info(f"Running enhanced test case {test_case['id']}: {test_case['query']}")
        
        response = {
            "test_id": test_case["id"],
            "query": test_case["query"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": test_case["type"],
            "expected_intent": test_case["expected_intent"],
            "expected_outcomes": test_case["expected_outcomes"]
        }
        
        # Step 1: Intent classification
        intent_start = time.perf_counter()
        intent_result = self.intent_classifier.classify_intent(test_case["query"])
        intent_time = time.perf_counter() - intent_start
        
        response["intent_analysis"] = {
            "detected_intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "processing_time": f"{intent_time:.4f}s",
            "reason": intent_result.get("reason", "N/A"),
            "method": intent_result["details"].get("method", "N/A")
        }
        
        # Step 2: GPT Judge evaluation of intent classification
        intent_evaluation = self.gpt_judge.evaluate_intent_classification(
            test_case["query"],
            intent_result["intent"],
            test_case["expected_intent"],
            intent_result["confidence"]
        )
        response["intent_evaluation"] = intent_evaluation
        
        # Step 3: Knowledge base processing (only for course_search intent)
        final_response_text = ""
        results_count = 0
        found_results = []
        
        if intent_result["intent"] == "course_search":
            kb_start = time.perf_counter()
            try:
                results = self.advisor.process_query(test_case["query"])
                kb_time = time.perf_counter() - kb_start
                results_count = len(results)
                found_results = results  # Store for hallucination checking
                
                # Prepare sample results
                sample_results = []
                for r in results[:3]:  # Get first 3 results
                    sample = {
                        "name": r.get("name", "N/A"),
                        "url": r.get("url", "N/A"),
                        "rating": r.get("rating", "N/A"),
                        "skills": r.get("skills", [])
                    }
                    if "instructor_name" in r:
                        sample["instructor_name"] = r["instructor_name"]
                    if "organization_name" in r:
                        sample["organization_name"] = r["organization_name"]
                    sample_results.append(sample)
                
                response["processing_details"] = {
                    "results_found": results_count,
                    "processing_time": f"{kb_time:.4f}s",
                    "sample_results": sample_results
                }
                
                # Generate final response
                if results:
                    analysis_start = time.perf_counter()
                    final_response_text = self.result_handler.analyze_results(results, test_case["query"])
                    analysis_time = time.perf_counter() - analysis_start
                    
                    response["system_response"] = {
                        "generation_time": f"{analysis_time:.4f}s",
                        "content": final_response_text
                    }
                else:
                    final_response_text = "I couldn't find any courses matching your criteria. Could you try refining your search?"
                    response["system_response"] = {
                        "generation_time": "0.0000s",
                        "content": final_response_text
                    }
                    
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                final_response_text = "I'm sorry, I encountered an error while searching for courses."
                response["processing_details"] = {"error": str(e)}
                response["system_response"] = {"content": final_response_text}
        else:
            # Handle non-course queries
            final_response_text = "Hello! I'm here to help you find courses. You can ask me about specific courses, instructors, or organizations offering courses."
            response["system_response"] = {"content": final_response_text}
        
        # Step 4: GPT Judge evaluation of response quality (with hallucination check)
        if final_response_text:
            response_evaluation = self.gpt_judge.evaluate_response_quality(
                test_case["query"],
                final_response_text,
                results_count,
                test_case["type"],
                found_results
            )
            response["response_evaluation"] = response_evaluation
        
        # Step 5: Automated checks against expected outcomes
        outcome_check = self.check_expected_outcomes(response, test_case["expected_outcomes"])
        response["outcome_validation"] = outcome_check
        
        return response
    
    def check_expected_outcomes(self, response: Dict, expected_outcomes: Dict) -> Dict:
        """Automated validation against expected outcomes"""
        validation = {
            "results_count_check": "pass",
            "keyword_check": "pass", 
            "overall_validation": "pass",
            "details": []
        }
        
        # Check results count
        actual_results = response.get("processing_details", {}).get("results_found", 0)
        expected_results = expected_outcomes.get("should_find_results", False)
        min_results = expected_outcomes.get("minimum_results", 0)
        
        if expected_results and actual_results < min_results:
            validation["results_count_check"] = "fail"
            validation["details"].append(f"Expected min {min_results} results, got {actual_results}")
        elif not expected_results and actual_results > 0:
            validation["results_count_check"] = "warning"  
            validation["details"].append(f"Expected no results, but got {actual_results}")
        
        # Check keywords in response
        system_response = response.get("system_response", {}).get("content", "").lower()
        expected_keywords = expected_outcomes.get("expected_keywords", [])
        
        missing_keywords = []
        for keyword in expected_keywords:
            if keyword.lower() not in system_response:
                missing_keywords.append(keyword)
        
        if missing_keywords:
            validation["keyword_check"] = "partial"
            validation["details"].append(f"Missing keywords: {missing_keywords}")
        
        # Overall validation
        if validation["results_count_check"] == "fail" or validation["keyword_check"] == "fail":
            validation["overall_validation"] = "fail"
        elif "warning" in [validation["results_count_check"], validation["keyword_check"]] or validation["keyword_check"] == "partial":
            validation["overall_validation"] = "partial"
        
        return validation

    def calculate_overall_metrics(self, all_responses: List[Dict]) -> Dict:
        """Calculate overall metrics across all test cases"""
        if not all_responses:
            return {}
        
        # Initialize metric accumulators
        intent_metrics = {
            "intent_accuracy": [],
            "confidence_appropriateness": [],
            "overall_intent_quality": []
        }
        
        response_metrics = {
            "relevance": [],
            "helpfulness": [],
            "clarity": [],
            "completeness": [],
            "professional_tone": [],
            "hallucination": [],
            "overall_score": []
        }
        
        validation_scores = []
        
        # Collect all metrics
        for resp in all_responses:
            # Intent metrics
            intent_eval = resp.get("intent_evaluation", {})
            for metric in intent_metrics:
                if metric in intent_eval:
                    intent_metrics[metric].append(intent_eval[metric])
            
            # Response metrics  
            response_eval = resp.get("response_evaluation", {})
            for metric in response_metrics:
                if metric in response_eval:
                    response_metrics[metric].append(response_eval[metric])
            
            # Validation score
            validation = resp.get("outcome_validation", {}).get("overall_validation", "unknown")
            if validation == "pass":
                validation_scores.append(10)
            elif validation == "partial":
                validation_scores.append(6)
            else:
                validation_scores.append(3)
        
        # Calculate overall statistics
        overall_stats = {}
        
        # Intent metrics
        overall_stats["intent_metrics"] = {}
        for metric, values in intent_metrics.items():
            if values:
                overall_stats["intent_metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        # Response metrics
        overall_stats["response_metrics"] = {}
        for metric, values in response_metrics.items():
            if values:
                overall_stats["response_metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        # Validation metrics
        if validation_scores:
            overall_stats["validation_metrics"] = {
                "mean": np.mean(validation_scores),
                "std": np.std(validation_scores),
                "min": np.min(validation_scores),
                "max": np.max(validation_scores),
                "median": np.median(validation_scores)
            }
        
        # Calculate system-wide overall score
        all_scores = []
        for resp in all_responses:
            intent_score = resp.get("intent_evaluation", {}).get("overall_intent_quality", 0)
            response_score = resp.get("response_evaluation", {}).get("overall_score", 0)
            validation = resp.get("outcome_validation", {}).get("overall_validation", "unknown")
            validation_score = 10 if validation == "pass" else 6 if validation == "partial" else 3
            
            overall_score = (intent_score + response_score + validation_score) / 3
            all_scores.append(overall_score)
        
        if all_scores:
            overall_stats["system_overall"] = {
                "mean": np.mean(all_scores),
                "std": np.std(all_scores),
                "min": np.min(all_scores),
                "max": np.max(all_scores),
                "median": np.median(all_scores)
            }
        
        return overall_stats

    def generate_comprehensive_report(self, response: Dict, test_case: Dict) -> str:
        """Generate comprehensive evaluation report"""
        
        # Calculate scores
        intent_eval = response.get("intent_evaluation", {})
        response_eval = response.get("response_evaluation", {})
        outcome_validation = response.get("outcome_validation", {})
        
        intent_score = intent_eval.get("overall_intent_quality", 0)
        response_score = response_eval.get("overall_score", 0)
        
        # Validation score
        validation_score = 10
        if outcome_validation.get("overall_validation") == "fail":
            validation_score = 3
        elif outcome_validation.get("overall_validation") == "partial":
            validation_score = 6
        
        overall_score = (intent_score + response_score + validation_score) / 3
        
        report = f"""
========================================
 GPT-3.5 TURBO LLM-AS-A-JUDGE EVALUATION
========================================
Test ID: {test_case['id']}
Query: "{test_case['query']}"
Type: {test_case['type']}
Timestamp: {response['timestamp']}
Overall Score: {overall_score:.2f}/10

1. INTENT CLASSIFICATION EVALUATION (GPT-3.5 Judge):
   Detected: {response['intent_analysis']['detected_intent']} (confidence: {response['intent_analysis']['confidence']})
   Expected: {test_case['expected_intent']}
   
   GPT-3.5 Turbo Scores:
   - Intent Accuracy: {intent_eval.get('intent_accuracy', 'N/A')}/10
   - Confidence Appropriateness: {intent_eval.get('confidence_appropriateness', 'N/A')}/10  
   - Overall Intent Quality: {intent_eval.get('overall_intent_quality', 'N/A')}/10
   - Is Correct: {intent_eval.get('is_correct', 'N/A')}
   
   GPT Reasoning: {intent_eval.get('reasoning', 'N/A')}

2. RESPONSE QUALITY EVALUATION (GPT-3.5 Judge):
   Results Found: {response.get('processing_details', {}).get('results_found', 0)}
   
   GPT-3.5 Turbo Scores:
   - Relevance: {response_eval.get('relevance', 'N/A')}/10
   - Helpfulness: {response_eval.get('helpfulness', 'N/A')}/10
   - Clarity: {response_eval.get('clarity', 'N/A')}/10
   - Completeness: {response_eval.get('completeness', 'N/A')}/10
   - Professional Tone: {response_eval.get('professional_tone', 'N/A')}/10
   - Hallucination Control: {response_eval.get('hallucination', 'N/A')}/10
   
   Overall Response Score: {response_eval.get('overall_score', 'N/A')}/10
   
   GPT Identified Strengths: {', '.join(response_eval.get('strengths', []))}
   GPT Identified Weaknesses: {', '.join(response_eval.get('weaknesses', []))}
   GPT Suggestions: {', '.join(response_eval.get('suggestions', []))}

3. OUTCOME VALIDATION:
   Results Count Check: {outcome_validation.get('results_count_check', 'N/A')}
   Keyword Check: {outcome_validation.get('keyword_check', 'N/A')}
   Overall Validation: {outcome_validation.get('overall_validation', 'N/A')}
   Validation Score: {validation_score}/10
   
   Details: {'; '.join(outcome_validation.get('details', ['No issues found']))}

4. SYSTEM RESPONSE:
{response.get('system_response', {}).get('content', 'No response generated')}

5. SYSTEM ARCHITECTURE:
   Main System: Qwen3:4b (Local LLM) for intent classification & response generation
   Judge System: GPT-3.5 Turbo (OpenAI API) for objective evaluation
   Embedding: all-MiniLM-L6-v2 (Local)
   Database: Neo4j (Local)

6. PERFORMANCE METRICS:
   Intent Processing: {response['intent_analysis']['processing_time']}
   KB Processing: {response.get('processing_details', {}).get('processing_time', 'N/A')}
   Response Generation: {response.get('system_response', {}).get('generation_time', 'N/A')}

========================================
"""
        return report

def main():
    """Main evaluation execution with GPT-3.5 Turbo judge"""
    print("Starting GPT-3.5 Turbo LLM-as-a-Judge Evaluation System...")
    print("Architecture: Qwen3:4b (System) + GPT-3.5 Turbo (Judge)")
    
    # Check for OpenAI API key
    api_key = "///***///***"
    if not api_key:
        print("\n❌ OpenAI API key not found!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load test cases from file or use default
    import sys
    if len(sys.argv) > 1:
        test_cases_file = sys.argv[1]
        print(f"Loading test cases from: {test_cases_file}")
        test_cases = load_test_cases_from_file(test_cases_file)
        if not test_cases:
            print(f"❌ Failed to load test cases from {test_cases_file}")
            print("Using default test cases instead...")
            test_cases = ENHANCED_TEST_CASES
    else:
        print("No test cases file specified, using default test cases")
        print("Usage: python test.py <test_cases.json>")
        test_cases = ENHANCED_TEST_CASES
    
    if not test_cases:
        print("❌ No test cases available!")
        return
    
    try:
        evaluator = EnhancedEvaluationSystem(api_key)
        
        # Test GPT connection
        print("Testing GPT-3.5 Turbo connection...")
        if evaluator.gpt_judge.test_connection():
            print("✅ GPT-3.5 Turbo connection successful")
        else:
            print("⚠️  GPT-3.5 Turbo connection test failed, but continuing...")
        
    except Exception as e:
        print(f"❌ Failed to initialize evaluation system: {e}")
        return
    
    print(f"\nSystem Status:")
    print(f"- Neo4j: {'✅' if evaluator.neo4j_conn else '❌'}")
    print(f"- Embedding Model: ✅ {evaluator.embedding_model.model_name}")
    print(f"- Main LLM (Qwen3:4b): ✅ Ready")
    print(f"- Judge LLM (GPT-3.5): ✅ Ready")
    
    # Create reports directory
    report_dir = "gpt_judge_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    all_scores = []
    all_responses = []  
    gpt_api_calls = 0
    start_time = time.perf_counter()
    
    print(f"\nRunning {len(test_cases)} enhanced test cases with GPT-3.5 Turbo judge...")
    print("Note: Each test case makes 2 GPT API calls (intent + response evaluation)")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Processing {test_case['id']}...")
        
        try:
            response = evaluator.run_enhanced_test_case(test_case)
            report = evaluator.generate_comprehensive_report(response, test_case)
            json_path = os.path.join(report_dir, f"{test_case['id']}_report.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(convert_numpy_types(response), jf, ensure_ascii=False, indent=2)
            
            all_responses.append(response)
            
            gpt_api_calls += 2  # Intent evaluation + Response evaluation
            
            # Print summary
            intent_score = response.get("intent_evaluation", {}).get("overall_intent_quality", 0)
            response_score = response.get("response_evaluation", {}).get("overall_score", 0)
            validation = response.get("outcome_validation", {}).get("overall_validation", "unknown")
            
            overall_score = (intent_score + response_score + (10 if validation == "pass" else 6 if validation == "partial" else 3)) / 3
            all_scores.append(overall_score)
            
            print(f"   GPT Intent Score: {intent_score}/10, Response Score: {response_score}/10")
            print(f"   Validation: {validation}, Overall: {overall_score:.2f}/10")
            
            # Save detailed report
            report_path = os.path.join(report_dir, f"{test_case['id']}_gpt_judge_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            
            # Brief pause to respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error evaluating {test_case['id']}: {e}")
            print(f"   ❌ Error: {e}")
    
    # Calculate overall metrics across all test cases
    overall_metrics = evaluator.calculate_overall_metrics(all_responses)
    
    # Final summary with detailed metrics
    total_time = time.perf_counter() - start_time
    avg_score = np.mean(all_scores) if all_scores else 0
    
    # Calculate detailed averages
    intent_metrics_avg = {}
    response_metrics_avg = {}
    
    if overall_metrics.get("intent_metrics"):
        for metric, stats in overall_metrics["intent_metrics"].items():
            intent_metrics_avg[metric] = stats["mean"]
    
    if overall_metrics.get("response_metrics"):
        for metric, stats in overall_metrics["response_metrics"].items():
            response_metrics_avg[metric] = stats["mean"]
    
    summary = f"""
========================================
GPT-3.5 TURBO JUDGE EVALUATION SUMMARY
========================================
Evaluation Architecture:
- Main System: Qwen3:4b (Local LLM)
- Judge System: GPT-3.5 Turbo (OpenAI API)

Results:
- Total Test Cases: {len(test_cases)}
- Successfully Evaluated: {len(all_scores)}
- GPT API Calls Made: {gpt_api_calls}
- Average Score: {avg_score:.2f}/10
- Total Time: {total_time:.2f}s
- Avg Time per Case: {total_time/len(all_scores) if all_scores else 0:.2f}s

INTENT CLASSIFICATION METRICS (Average):
- Intent Accuracy: {intent_metrics_avg.get('intent_accuracy', 0):.2f}/10
- Confidence Appropriateness: {intent_metrics_avg.get('confidence_appropriateness', 0):.2f}/10
- Overall Intent Quality: {intent_metrics_avg.get('overall_intent_quality', 0):.2f}/10

RESPONSE QUALITY METRICS (Average):
- Relevance: {response_metrics_avg.get('relevance', 0):.2f}/10
- Helpfulness: {response_metrics_avg.get('helpfulness', 0):.2f}/10
- Clarity: {response_metrics_avg.get('clarity', 0):.2f}/10
- Completeness: {response_metrics_avg.get('completeness', 0):.2f}/10
- Professional Tone: {response_metrics_avg.get('professional_tone', 0):.2f}/10
- Hallucination Control: {response_metrics_avg.get('hallucination', 0):.2f}/10
- Overall Response Score: {response_metrics_avg.get('overall_score', 0):.2f}/10

Score Distribution:
- Excellent (9-10): {sum(1 for s in all_scores if s >= 9)} cases
- Good (7-8.9): {sum(1 for s in all_scores if 7 <= s < 9)} cases
- Fair (5-6.9): {sum(1 for s in all_scores if 5 <= s < 7)} cases
- Poor (0-4.9): {sum(1 for s in all_scores if s < 5)} cases

Benefits of GPT-3.5 Turbo Judge:
✅ Objective evaluation independent of main system
✅ High-quality reasoning and detailed feedback
✅ Consistent scoring across test cases
✅ Professional assessment of conversational quality
✅ Comprehensive hallucination detection

Reports saved to: {report_dir}/
========================================
"""
    
    print(summary)
    
    # Create comprehensive summary JSON with all detailed metrics
    summary_json = {
        "evaluation_metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": "Qwen3:4b (System) + GPT-3.5 Turbo (Judge)",
            "total_test_cases": len(test_cases),
            "successful_evaluations": len(all_scores),
            "gpt_api_calls": gpt_api_calls,
            "total_time_seconds": total_time,
            "average_time_per_case": total_time/len(all_scores) if all_scores else 0
        },
        "overall_metrics": {
            "system_average_score": avg_score,
            "intent_metrics_average": intent_metrics_avg,
            "response_metrics_average": response_metrics_avg,
            "validation_metrics": overall_metrics.get("validation_metrics", {}),
            "system_overall_stats": overall_metrics.get("system_overall", {})
        },
        "score_distribution": {
            "excellent_9_to_10": sum(1 for s in all_scores if s >= 9),
            "good_7_to_8_9": sum(1 for s in all_scores if 7 <= s < 9),
            "fair_5_to_6_9": sum(1 for s in all_scores if 5 <= s < 7),
            "poor_0_to_4_9": sum(1 for s in all_scores if s < 5)
        },
        "test_results": []
    }
    
    # Add detailed results for each test case
    for resp in all_responses:
        test_id = resp["test_id"]
        intent_eval = resp.get("intent_evaluation", {})
        response_eval = resp.get("response_evaluation", {})
        outcome_validation = resp.get("outcome_validation", {})
        
        # Calculate validation score
        validation_score = 10
        if outcome_validation.get("overall_validation") == "fail":
            validation_score = 3
        elif outcome_validation.get("overall_validation") == "partial":
            validation_score = 6
        
        overall_score = (
            intent_eval.get("overall_intent_quality", 0) + 
            response_eval.get("overall_score", 0) + 
            validation_score
        ) / 3
        
        test_result = {
            "test_id": test_id,
            "query": resp["query"],
            "type": resp["type"],
            "overall_score": round(overall_score, 2),
            
            # Intent Classification Metrics
            "intent_metrics": {
                "detected_intent": resp["intent_analysis"]["detected_intent"],
                "expected_intent": resp["expected_intent"],
                "confidence": resp["intent_analysis"]["confidence"],
                "intent_accuracy": intent_eval.get("intent_accuracy", 0),
                "confidence_appropriateness": intent_eval.get("confidence_appropriateness", 0),
                "overall_intent_quality": intent_eval.get("overall_intent_quality", 0),
                "is_correct": intent_eval.get("is_correct", False),
                "gpt_reasoning": intent_eval.get("reasoning", "N/A")
            },
            
            # Response Quality Metrics
            "response_metrics": {
                "relevance": response_eval.get("relevance", 0),
                "helpfulness": response_eval.get("helpfulness", 0),
                "clarity": response_eval.get("clarity", 0),
                "completeness": response_eval.get("completeness", 0),
                "professional_tone": response_eval.get("professional_tone", 0),
                "hallucination": response_eval.get("hallucination", 0),
                "overall_score": response_eval.get("overall_score", 0),
                "strengths": response_eval.get("strengths", []),
                "weaknesses": response_eval.get("weaknesses", []),
                "suggestions": response_eval.get("suggestions", []),
                "gpt_reasoning": response_eval.get("reasoning", "N/A")
            },
            
            # Validation Metrics
            "validation_metrics": {
                "results_count_check": outcome_validation.get("results_count_check", "unknown"),
                "keyword_check": outcome_validation.get("keyword_check", "unknown"),
                "overall_validation": outcome_validation.get("overall_validation", "unknown"),
                "validation_score": validation_score,
                "validation_details": outcome_validation.get("details", [])
            },
            
            # Processing Details
            "processing_details": {
                "results_found": resp.get("processing_details", {}).get("results_found", 0),
                "intent_processing_time": resp["intent_analysis"]["processing_time"],
                "kb_processing_time": resp.get("processing_details", {}).get("processing_time", "N/A"),
                "response_generation_time": resp.get("system_response", {}).get("generation_time", "N/A")
            },
            
            # System Response
            "system_response": resp.get("system_response", {}).get("content", "No response generated")
        }
        
        summary_json["test_results"].append(test_result)
    
    # Save comprehensive summary
    summary_path = os.path.join(report_dir, "comprehensive_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(summary_json), f, ensure_ascii=False, indent=2)
    
    # Also save to the requested path
    os.makedirs("course_searching_refactor/evaluation/results", exist_ok=True)
    with open("course_searching_refactor/evaluation/results/summary_detailed.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(summary_json), f, ensure_ascii=False, indent=2)
    
    # Save text summary
    with open(os.path.join(report_dir, "gpt_evaluation_summary.txt"), "w") as f:
        f.write(summary)
    
    print(f"\n✅ GPT-3.5 Turbo evaluation completed successfully!")
    print(f"📊 Comprehensive results saved to: {summary_path}")
    print(f"📋 Text summary saved to: {os.path.join(report_dir, 'gpt_evaluation_summary.txt')}")
    print(f"💰 Estimated cost: ~${(gpt_api_calls * 0.002):.4f} (assuming $0.002 per API call)")
    
    # Print top insights
    if all_responses:
        best_test = max(all_responses, key=lambda x: (
            x.get("intent_evaluation", {}).get("overall_intent_quality", 0) + 
            x.get("response_evaluation", {}).get("overall_score", 0)
        ) / 2)
        
        worst_test = min(all_responses, key=lambda x: (
            x.get("intent_evaluation", {}).get("overall_intent_quality", 0) + 
            x.get("response_evaluation", {}).get("overall_score", 0)
        ) / 2)
        
        print(f"\n🏆 Best performing test: {best_test['test_id']} - \"{best_test['query']}\"")
        print(f"⚠️  Needs improvement: {worst_test['test_id']} - \"{worst_test['query']}\"")
    
    return avg_score, all_scores, summary_json

if __name__ == "__main__":
    main()
